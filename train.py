"""
train.py
========
Training loop for the ~1B-parameter 1.58-bit ternary ``WeatherNet``.

Runs on CPU, Apple MPS, or NVIDIA CUDA GPUs.  On GPU the forward/backward
passes are executed under ``torch.autocast`` (BF16 on Ampere+, optional
FP16 with a GradScaler) while the *latent* parameters are always updated in
FP32 by AdamW -- exactly the scheme BitNet b1.58 uses in practice.

Dummy data
----------
By default a synthetic, weather-shaped dataset is generated on the fly
(diurnal temperature cycle, synoptic pressure waves, etc.) so the loop is
fully runnable today.  Point ``--data`` at a zip of station CSVs (see
``download_data.py``) to train on real century-scale observations instead.

Example invocations
-------------------
    # CPU smoke test (small net):
    python train.py --smoke

    # Full ~1B budget, CPU, few steps (slow, proves the loop):
    python train.py --width 8192 --depth 14 --epochs 1 --batch-size 8

    # GPU (auto-detected), BF16 mixed precision, real data:
    python train.py --data data/raw/weather_zip --precision bf16 \
                    --batch-size 64 --grad-accum 4 --epochs 50
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.nn as nn

from data import SyntheticWeatherGenerator, Normalizer, statistics_context, WeatherZipCSV
from model import WeatherNet, WeatherNetConfig, WINDOW_HOURS, NUM_VARS, NUM_CONTEXT, OUT_FEATURES


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the 1B 1.58-bit ternary WeatherNet")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "cpu", "mps"],
                   help="device to run on (auto = first available)")
    p.add_argument("--precision", default="auto",
                   choices=["auto", "fp32", "bf16", "fp16"],
                   help="compute dtype; auto = bf16 on CUDA, fp32 elsewhere")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="gradient accumulation micro-batches (effective batch = bs*accum)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ada-lr", type=float, default=0.01,
                   help="learning rate used when --optim adafactor")
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--samples", type=int, default=8192,
                   help="synthetic samples generated if --data is not given")
    p.add_argument("--width", type=int, default=8192, help="hidden width (default = 1B budget)")
    p.add_argument("--depth", type=int, default=14, help="residual blocks (default = 1B budget)")
    p.add_argument("--chunk-size", type=int, default=0,
                   help="matmul output chunk for low-VRAM GPUs (0 = off)")
    p.add_argument("--ternary-threshold", type=float, default=0.1,
                   help="dead-zone factor (frac. of mean|W|); 0 = pure {-1,+1}")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true",
                   help="tiny net + 2 batches, ~seconds; for CI/dev boxes")
    p.add_argument("--data", default=None,
                   help="path to real-data zip of station CSVs (download_data.py)")
    p.add_argument("--stream", action="store_true",
                   help="normalize windows from the zip; cached in RAM by default "
                        "(fast); --no-cache re-streams the archive every epoch")
    p.add_argument("--no-cache", action="store_true",
                   help="with --stream: re-read + re-normalize the zip every epoch "
                        "(slow fallback)")
    p.add_argument("--no-cpu-offload", action="store_true",
                   help="keep all optimizer work on the GPU (default offloads the "
                        "tiny 1D bias/RMS-gain + small head tensors to CPU cores)")
    p.add_argument("--holdout", type=int, default=0,
                   help="hold back the LAST N windows (latest stations in archive) "
                        "for honest out-of-sample eval")
    p.add_argument("--resume", choices=["", "auto"], default="",
                   help="auto: continue from runs/<out>/latest.pt if present "
                        "(safe against crashes/reboots)")
    p.add_argument("--save-every", type=int, default=500,
                   help="checkpoint every N optimizer steps (crash recovery");
    p.add_argument("--out", default="runs/weather")
    p.add_argument("--optim", choices=["adamw", "adafactor"], default="adamw",
                   help="adafactor = factorized states, ~no optimizer memory "
                        "(needed to fit the 1B model in 8GB VRAM)")
    p.add_argument("--data-limit", type=int, default=0,
                   help="max windows read from --data zip (0 = all)")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Device + precision helpers
# ---------------------------------------------------------------------------
def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def resolve_precision(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def make_model(args: argparse.Namespace) -> WeatherNet:
    if args.smoke:
        cfg = WeatherNetConfig(d_embed=128, d_hidden=256, num_hidden_blocks=2, d_proj=64)
    else:
        cfg = WeatherNetConfig(
            d_hidden=args.width,
            num_hidden_blocks=args.depth,
            chunk_size=args.chunk_size or None,
            ternary_threshold=args.ternary_threshold,
            out_features=OUT_FEATURES,
        )
    net = WeatherNet(cfg)
    print(net.summary())
    return net


def make_optimizer(args: argparse.Namespace, model: nn.Module):
    """AdamW (default) or Adafactor for tiny optimizer memory on 8GB GPUs."""
    if args.optim == "adafactor":
        lr = args.ada_lr
        groups = []
        groups.append({"params": [p for p in model.parameters() if p.dim() >= 2],
                       "lr": lr, "weight_decay": args.weight_decay})
        groups.append({"params": [p for p in model.parameters() if p.dim() < 2],
                       "lr": lr, "weight_decay": 0.0})
        return torch.optim.Adafactor(
            groups, lr=lr, beta2_decay=-0.8, eps=(1e-16, 1e-3), d=1.0,
        )
    return torch.optim.AdamW(param_groups(model, args.lr, args.weight_decay))


def param_groups(model: nn.Module, lr: float, wd: float):
    """Separate args: weight decay only for the huge 2D weight tensors."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:  # 2D matrices (linear weights) get wd; norms/biases don't
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "lr": lr, "weight_decay": wd},
        {"params": no_decay, "lr": lr, "weight_decay": 0.0},
    ]


def make_lr(step: int, warmup: int, total: int, lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return lr * 0.5 * (1.0 + math.cos(math.pi * p))


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    err = pred - target
    abs_err = err.abs()
    quad = 0.5 * abs_err.square()
    lin = delta * (abs_err - 0.5 * delta)
    return torch.where(abs_err <= delta, quad, lin).mean()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    normalizer: Normalizer,
    device: torch.device,
    autocast_ctx,
) -> dict[str, float]:
    """RMSE (physical units) + per-forecast-hour Pearson correlation."""
    model.eval()
    se_sum = 0.0
    n = 0
    preds, targets = [], []
    for xw, xc, yt in loader:
        xw, xc, yt = xw.to(device), xc.to(device), yt.to(device)
        with autocast_ctx:
            y = model(xw, xc)
        y = normalizer.inverse_y(y.float())      # physical units
        yt = normalizer.inverse_y(yt.float())    # physical units (same space)
        se_sum += ((y - yt) ** 2).sum().item()
        n += yt.numel()
        preds.append(y)
        targets.append(yt)
    rmse = math.sqrt(se_sum / max(1, n))

    pred = torch.cat(preds)
    targ = torch.cat(targets)
    # Pearson correlation per forecast hour, then average.
    # NB: torch .norm(0) is the L0 'norm' (nonzero count) -- use L2 explicitly.
    p = pred - pred.mean(0)
    t = targ - targ.mean(0)
    pn = torch.linalg.vector_norm(p, dim=0)
    tn = torch.linalg.vector_norm(t, dim=0)
    corr = (p * t).sum(0) / (pn * tn + 1e-8)
    model.train()
    return {"rmse": rmse, "corr": corr.mean().item()}


@torch.no_grad()
def ternary_stats(model: nn.Module) -> tuple[float, float]:
    """Fraction of ternary weights that are +/-1 vs 0 (1.58-bit sparsity)."""
    from layers import BitLinear
    non, yes = 0, 0
    for m in model.modules():
        if isinstance(m, BitLinear):
            w = m.weight.detach()
            thr = m.ternary_threshold * w.abs().mean() if m.ternary_threshold > 0 else 0.0
            q = (w > thr).to(w.dtype) - (w < -thr).to(w.dtype)
            non += (q != 0).sum().item()
            yes += q.numel()
    return non / max(1, yes), 1.0 - non / max(1, yes)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_precision(args.precision, device)

    # ---- data --------------------------------------------------------------
    stream = args.stream
    total_samples = None
    hold_ds = None
    if args.data:
        ds = WeatherZipCSV(args.data)
        normalizer = Normalizer()
        if stream:
            if not args.no_cache:
                # memory-cached: read + normalize ONCE, train from RAM tensors
                # (300k windows ~ 150 MB; kills the per-epoch zip re-read &
                # Python normalize cost that capped throughput at ~2.4 smp/s)
                xw_all, ctx_all, y_all = ds.to_tensors(limit=args.data_limit or None)
                n_all = len(xw_all)
                print(f"[data] read {n_all:,} windows -> RAM cache ({args.data})")
                h = min(args.holdout, max(0, n_all - 1))
                if h > 0:
                    tr = slice(0, n_all - h)
                    ev = slice(n_all - h, n_all)
                    normalizer.fit(xw_all[tr], y_all[tr])
                    print(f"[data] holdout {h:,} windows (last/latest stations), "
                          f"train {n_all - h:,}")
                else:
                    tr = slice(0, n_all)
                    normalizer.fit(xw_all, y_all)
                xw_t = torch.tensor(normalizer.transform_x(xw_all), dtype=dtype)
                ctx_t = torch.tensor(normalizer.transform_ctx(ctx_all), dtype=dtype)
                y_t = torch.tensor(normalizer.transform_y(y_all), dtype=dtype)
                train_ds = torch.utils.data.TensorDataset(
                    xw_t[tr], ctx_t[tr], y_t[tr])
                hold_ds = (torch.utils.data.TensorDataset(
                    xw_t[ev], ctx_t[ev], y_t[ev]) if h > 0 else train_ds)
                total_samples = len(train_ds)
            else:
                xm, xs, ym, ys, n_win = ds.stats(limit=args.data_limit or None)
                normalizer.x_mean, normalizer.x_std = xm, xs
                normalizer.y_mean, normalizer.y_std = ym, ys
                total_samples = n_win
                print(f"[data] streamed stats over {n_win:,} windows from {args.data}")

                class _StreamDS(torch.utils.data.IterableDataset):
                    def __init__(self, gen_fn):
                        self.gen_fn = gen_fn

                    def __iter__(self):
                        yield from self.gen_fn()

                # yields (xw, ctx, yt) already bf16 on device; DataLoader only
                # wraps iteration (unbatched=false creates a new stream each epoch)
                train_ds = _StreamDS(lambda: ds.iter_batches(
                    args.batch_size, dtype, device, xm, xs, ym, ys,
                    limit=args.data_limit or None))
        else:
            xw_all, ctx_all, y_all = ds.to_tensors(limit=args.data_limit or None)   # raw physical units
            print(f"[data] read {len(xw_all):,} windows from {args.data}")
            normalizer.fit(xw_all, y_all)
            xw_t = torch.tensor(normalizer.transform_x(xw_all), dtype=dtype)
            ctx_t = torch.tensor(normalizer.transform_ctx(ctx_all), dtype=dtype)
            y_t = torch.tensor(normalizer.transform_y(y_all), dtype=dtype)
            train_ds = torch.utils.data.TensorDataset(xw_t, ctx_t, y_t)
            hold_ds = train_ds
    else:
        gen = SyntheticWeatherGenerator(n=args.samples, seed=args.seed)
        xw, y = gen.sample()
        ctx = statistics_context(xw)
        normalizer = Normalizer()
        normalizer.fit(xw, y)
        xw_t = torch.tensor(normalizer.transform_x(xw), dtype=dtype)
        ctx_t = torch.tensor(normalizer.transform_ctx(ctx), dtype=dtype)
        y_t = torch.tensor(normalizer.transform_y(y), dtype=dtype)
        train_ds = torch.utils.data.TensorDataset(xw_t, ctx_t, y_t)
        hold_ds = train_ds
    if total_samples is None:
        total_samples = len(train_ds)
    print(f"[data] {total_samples} samples on {device} | "
          f"var std {normalizer.x_std.round(2).tolist()}")

    # ---- model --------------------------------------------------------------
    model = make_model(args)
    if dtype == torch.bfloat16:
        model.bfloat16()   # halves weight memory so the 1B model fits 8GB VRAM
                            # (cast is done on CPU before .to(device)).
                            # NB: on this gfx1032 box, EVERY GPU run must set
                            # HSA_OVERRIDE_GFX_VERSION=10.3.0 or the first
                            # kernel launch segfaults in libamdhip64.
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # ---- optimizer + scheduler -----------------------------------------------
    cpu_opt = None
    small = []
    cpu_copies = []
    if args.optim == "adafactor" and not args.no_cpu_offload:
        # Offload the tiny 1D bias/RMS-gain + small head tensors onto CPU cores.
        # Their Adafactor update transfers only a few MB/step, so the host can
        # never bottleneck -- while sparing the GPU kernels and bypassing the
        # ROCm Adafactor 1D-NaN path entirely for those tensors.
        small_ids = {id(p) for p in model.parameters()
                     if p.dim() < 2 or p.numel() <= 1_000_000}
        big = [p for p in model.parameters() if id(p) not in small_ids]
        small = [p for p in model.parameters() if id(p) in small_ids]
        opt = torch.optim.Adafactor(
            [{"params": big, "lr": args.ada_lr, "weight_decay": args.weight_decay}],
            lr=args.ada_lr, beta2_decay=-0.8, eps=(1e-16, 1e-3), d=1.0)
        # fp32 CPU mirrors of the offloaded tensors
        cpu_copies = [p.detach().to("cpu").float().requires_grad_(False)
                      for p in small]
        cpu_opt = torch.optim.Adafactor(
            cpu_copies, lr=args.ada_lr, beta2_decay=-0.8,
            eps=(1e-16, 1e-3), d=1.0)
        print(f"[opt] GPU Adafactor {sum(p.numel() for p in big):,}p | "
              f"CPU Adafactor {sum(p.numel() for p in small):,}p (offloaded)")
    else:
        opt = make_optimizer(args, model)
    total_steps = math.ceil(total_samples / (args.batch_size * args.grad_accum)) * args.epochs
    step = 0

    # ---- resume support: continue from the last checkpoint if requested ----
    resume_epoch = 0
    done_marker = os.path.join(args.out, "DONE")
    if args.resume == "auto" and os.path.exists(os.path.join(args.out, "latest.pt")):
        st = torch.load(os.path.join(args.out, "latest.pt"), map_location="cpu",
                        weights_only=False)
        if st.get("config") == vars(model.cfg):
            model.load_state_dict(st["model_state"])
            normalizer.load_state_dict(st["normalizer"])
            if st.get("opt_state") is not None:
                opt.load_state_dict(st["opt_state"])
            if cpu_opt is not None and st.get("cpu_opt_state") is not None:
                cpu_opt.load_state_dict(st["cpu_opt_state"])
            resume_epoch = int(st.get("epoch", 0))
            step = int(st.get("step", 0))
            print(f"[resume] continuing from epoch {resume_epoch}, "
                  f"global step {step}")
        else:
            print("[resume] config mismatch with checkpoint -- ignoring")
    if os.path.exists(done_marker):
        print(f"[done] {done_marker} exists; run already finished.")
        return

    def save_ckpt(path: str, epoch: int, cur_step: int) -> None:
        ckpt = {
            "model_state": model.state_dict(),
            "config": vars(model.cfg),
            "normalizer": normalizer.state_dict(),
            "n_params": n_params,
            "dtype": str(dtype),
            "epoch": epoch,
            "step": cur_step,
            "opt_state": opt.state_dict(),
            "cpu_opt_state": cpu_opt.state_dict() if cpu_opt is not None else None,
        }
        tmp = path + ".tmp"
        torch.save(ckpt, tmp)
        os.replace(tmp, path)

    if isinstance(train_ds, torch.utils.data.IterableDataset):
        # uncached stream: batches re-normalized from the archive every epoch
        loader = train_ds
        eval_loader = train_ds
        shuffle = False
    else:
        loader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, drop_last=True,
        )
        eval_loader = torch.utils.data.DataLoader(
            (hold_ds or train_ds), batch_size=args.batch_size, shuffle=False,
            num_workers=0, drop_last=False,
        )
        shuffle = True

    use_scaler = dtype == torch.float16 and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler) if device.type == "cuda" else None
    autocast_ctx = torch.autocast(
        device_type=device.type, dtype=dtype,
        enabled=(dtype in (torch.bfloat16, torch.float16)),
    )

    print(f"[run] device={device} precision={str(dtype).split('.')[-1]} "
          f"steps={total_steps} effective_batch={args.batch_size * args.grad_accum}")

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    for p in model.parameters():
        p.grad = None

    for epoch in range(resume_epoch, args.epochs):
        ep_loss = 0.0
        nb = 0
        te = time.time()
        micro_total = math.ceil(total_samples / max(1, args.batch_size))
        for bi, (xw, xc, yt) in enumerate(loader):
            nb += 1
            xw, xc, yt = xw.to(device), xc.to(device), yt.to(device)

            with autocast_ctx:
                y = model(xw, xc)
            if dtype == torch.float32:
                loss = huber_loss(y.float(), yt.float())
            else:
                # stay in bf16 end-to-end; fp32 loss graphs cost ~4GB VRAM
                loss = huber_loss(y, yt)

            # scale the micro-batch loss so grad-accumulated sums are correct
            loss = loss / args.grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (bi + 1) % args.grad_accum == 0:
                # LR schedule (global optimizer step)
                lr = make_lr(step, args.warmup_steps, total_steps, args.lr)
                for g in opt.param_groups:
                    g["lr"] = lr
                if cpu_opt is not None:
                    for g in cpu_opt.param_groups:
                        g["lr"] = lr

                if scaler is not None:
                    scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                if cpu_opt is not None:
                    # CPU-side Adafactor step for the offloaded tiny tensors
                    grads = [p.grad.detach().to("cpu", dtype=torch.float32)
                             if p.grad is not None else None for p in small]
                    for p in small:
                        p.grad = None
                    opt.step()
                    for cp, g in zip(cpu_copies, grads):
                        cp.grad = g
                    cpu_opt.step()
                    for p, cp in zip(small, cpu_copies):
                        p.data.copy_(cp.data.to(dtype=p.dtype, device=p.device))
                else:
                    if scaler is not None:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                for p in model.parameters():
                    p.grad = None
                step += 1
                if args.save_every > 0 and step % args.save_every == 0:
                    save_ckpt(os.path.join(args.out, "latest.pt"), epoch, step)
                    print(f"[save] step {step} -> latest.pt", flush=True)

            if (bi + 1) % 250 == 0:
                el = time.time() - te
                rate = (bi + 1) * args.batch_size / max(el, 1e-9)
                rem = (micro_total - (bi + 1)) * args.batch_size / max(rate, 1e-9) / 60
                print(f"[e{epoch} {bi + 1}/{micro_total}] {rate:5.1f} smp/s  "
                      f"~{rem:4.0f} min left", flush=True)

            ep_loss += loss.item() * args.grad_accum

        ep_loss /= (nb + 1e-9)
        # --- end of epoch evaluation --------------------------------------
        if device.type == "cuda":
            torch.cuda.synchronize()
        metrics = {}
        if epoch % max(1, args.epochs // 4) == 0 or epoch == args.epochs - 1:
            metrics = evaluate(model, eval_loader, normalizer, device, autocast_ctx)
        nnz, sp = ternary_stats(model)
        secs = time.time() - t0
        print(f"[epoch {epoch:02d}] loss={ep_loss:.4f}  "
              + (f"rmse={metrics['rmse']:.3f} corr={metrics['corr']:.3f}  " if metrics else "")
              + f"ternary={nnz:.1%} active  samples/s={nb*args.batch_size/max(1,secs):.1f}")

        if metrics:
            import json
            mfile = os.path.join(args.out, "metrics.json")
            hist = []
            if os.path.exists(mfile):
                try:
                    hist = json.load(open(mfile))
                except Exception:
                    hist = []
            nds = getattr(eval_loader, "dataset", None)
            hist.append({
                "epoch": epoch,
                "train_loss": round(ep_loss, 4),
                "rmse_c": round(metrics["rmse"], 3),
                "corr": round(metrics["corr"], 4),
                "train_windows": int(nb * args.batch_size),
                "holdout_windows": int(len(nds)) if nds is not None else -1,
            })
            with open(mfile, "w") as f:
                json.dump(hist, f, indent=1)
            print(f"[metrics] logged -> {mfile}")

        # --- incremental checkpoint so visualizers/drivers see live weights --
        save_ckpt(os.path.join(args.out, "latest.pt"), epoch, step)
        print(f"[ckpt] epoch {epoch} -> latest.pt")

    # ---- final checkpoint ----------------------------------------------------
    save_ckpt(os.path.join(args.out, "final.pt"), args.epochs, step)
    with open(done_marker, "w") as f:
        f.write(f"epochs={args.epochs} step={step}\n")
    print(f"[done] checkpoint -> {os.path.join(args.out, 'final.pt')} "
          f"({secs:.0f}s), marker {done_marker}")


if __name__ == "__main__":
    train(parse_args())