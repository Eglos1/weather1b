#!/usr/bin/env python3
"""train_weather_cpu.py
======================
CPU-only, multicore training loop for the ~1B-parameter 1.58-bit ternary
WeatherNet, designed to run *alongside* the GPU LLM trainer.

Why a separate trainer?
------------------------
``train.py`` can already run on CPU, but its real-data path reads a single
zip of per-station hourly CSVs.  The new "all the world's data" archive is
the global ISD-Lite HOURLY harvest (25k+ stations, 1973..2025), which
``build_hourly_cache.py`` pre-aggregates into daily windows stored as npy
shards.  This trainer consumes that cache with:

  * CPU only -- never touches the GPU the LLM is hogging;
  * multicore -- ``--threads`` (default ``nproc - 4``) maps to
    ``torch.set_num_threads``; no DataLoader workers (shard iteration is
    pure numpy);
  * politest mode -- launch via ``supervise_weather_cpu.sh`` which runs
    ``nice -n 15 ionice -c 3``; the trainer also yields with
    ``os.sched_yield`` periodically;
  * low memory -- windows are read shard-by-shard (50k ~= 25 MB each);
    AdamW on 954M fp32 params would need ~11 GB of optimizer state, so the
    default optimizer is **Adafactor** (factorized, ~no extra memory);
  * resume-safe -- ``--resume auto`` continues from ``runs/<out>/latest.pt``
    exactly like the LLM trainer, and ``--init-ckpt`` warm-starts from a
    previous WeatherNet checkpoint (e.g. ``runs/weather-gpu/final.pt``).

Usage
-----
    # supervise_weather_cpu.sh handles nice/ionice + auto-restart:
    bash supervise_weather_cpu.sh

    # or run directly (already-niced):
    python3 train_weather_cpu.py --init-ckpt runs/weather-gpu/final.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from data import Normalizer
from model import WeatherNet, WeatherNetConfig
from train import make_lr, param_groups, evaluate, ternary_stats


def huber(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    err = pred - target
    ab = err.abs()
    quad = 0.5 * ab.square()
    lin = delta * (ab - 0.5 * delta)
    return torch.where(ab <= delta, quad, lin).mean()


class ShardTrainer:
    """Streams (x, ctx, y) batches out of the npy shard cache.

    Each shard holds ~50k windows.  We load ONE shard into RAM at a time
    (~25 MB), shuffle its row order, and emit normalized FP32 CPU batches.
    After every shard is exhausted we move to the next; once all shards have
    been seen we reshuffle the shard list (a "shard epoch").
    """

    def __init__(self, cache_dir: str, batch_size: int, rng: np.random.Generator) -> None:
        with open(os.path.join(cache_dir, "manifest.json")) as f:
            man = json.load(f)
        self.cache = cache_dir
        self.shards = list(man["shards"])
        self.batch_size = batch_size
        self.rng = rng
        self.idx = 0
        self.perm = None
        self.x = self.c = self.y = None

    def _load(self, s: str):
        x = np.load(os.path.join(self.cache, f"x_{s}.npy"))
        c = np.load(os.path.join(self.cache, f"ctx_{s}.npy"))
        y = np.load(os.path.join(self.cache, f"y_{s}.npy"))
        return x, c, y

    def stream(self, xm, xs, ym, ys):
        """Yield normalized (xw (B,24,4), ctx (B,4), yt (B,24)) float32."""
        while True:
            if self.x is None:
                if self.idx % len(self.shards) == 0:
                    self.rng.shuffle(self.shards)      # reshuffle each epoch
                s = self.shards[self.idx % len(self.shards)]
                self.idx += 1
                self.x, self.c, self.y = self._load(s)
                self.perm = self.rng.permutation(len(self.x))
            n = len(self.x)
            if self.perm.size < self.batch_size:
                self.x = self.c = self.y = self.perm = None
                continue
            b = self.perm[:self.batch_size]
            self.perm = self.perm[self.batch_size:]
            xw = ((self.x[b] - xm) / xs).astype(np.float32)
            ctx = ((self.c[b] - xm) / xs).astype(np.float32)
            yw = ((self.y[b] - ym) / ys).astype(np.float32)
            yield (torch.from_numpy(xw), torch.from_numpy(ctx),
                   torch.from_numpy(yw))


def make_optimizer(args, model: nn.Module):
    if args.optim == "adafactor":
        groups = [{"params": [p for p in model.parameters() if p.dim() >= 2],
                   "lr": args.ada_lr, "weight_decay": args.weight_decay},
                  {"params": [p for p in model.parameters() if p.dim() < 2],
                   "lr": args.ada_lr, "weight_decay": 0.0}]
        return torch.optim.Adafactor(
            groups, lr=args.ada_lr, beta2_decay=-0.8, eps=(1e-16, 1e-3), d=1.0,
        )
    return torch.optim.AdamW(param_groups(model, args.lr, args.weight_decay))


def parse_args():
    p = argparse.ArgumentParser(description="CPU WeatherNet trainer (hourly cache)")
    p.add_argument("--cache-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "hourly_cache"))
    p.add_argument("--out", default="runs/weather-cpu")
    p.add_argument("--threads", type=int, default=max(4, os.cpu_count() - 4),
                   help="intra-op threads for the multicore matmul path")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--optim", choices=["adamw", "adafactor"], default="adafactor")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ada-lr", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=250000,
                   help="long-running background budget, like the LLM")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--width", type=int, default=8192)
    p.add_argument("--depth", type=int, default=14)
    p.add_argument("--chunk-size", type=int, default=0)
    p.add_argument("--init-ckpt", default=None,
                   help="warm-start from a prior WeatherNet checkpoint "
                        "(e.g. runs/weather-gpu/final.pt); adopts its config")
    p.add_argument("--eval-limit", type=int, default=2000,
                   help="cap the eval set at this many windows (CPU evals are "
                        "slow; the cache may hold up to 20k)")
    p.add_argument("--resume", choices=["", "auto"], default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true",
                   help="~20 steps + tiny eval, verifies the whole path")
    return p.parse_args()


def train(args) -> None:
    torch.manual_seed(args.seed)
    if args.smoke:
        args.max_steps = 25
    torch.set_num_threads(max(1, args.threads))
    try:
        torch.set_num_interop_threads(1)    # only safe once, before any compute
    except RuntimeError:
        pass

    rng = np.random.default_rng(args.seed)

    # ---- normalizer from the cache manifest -------------------------------
    with open(os.path.join(args.cache_dir, "manifest.json")) as f:
        man = json.load(f)
    nm = Normalizer()
    nm.x_mean = np.asarray(man["stats"]["x_mean"], dtype=np.float64)
    nm.x_std = np.asarray(man["stats"]["x_std"], dtype=np.float64)
    nm.y_mean = np.asarray(man["stats"]["y_mean"], dtype=np.float64)
    nm.y_std = np.asarray(man["stats"]["y_std"], dtype=np.float64)
    n_shards = len(man["shards"])
    n_train = int(man["train_windows"])
    print(f"[cache] {n_train:,} train windows across {n_shards} shards | "
          f"eval {man.get('eval_windows', 0):,}", flush=True)

    # ---- model --------------------------------------------------------------
    if args.init_ckpt and args.resume == "auto" and \
            os.path.exists(os.path.join(args.out, "latest.pt")):
        pass   # resume takes priority; --init-ckpt ignored below on resume
    if args.init_ckpt and not os.path.exists(os.path.join(args.out, "latest.pt")):
        st = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        cfg = WeatherNetConfig(**st["config"])
        cfg.chunk_size = None      # GPU-run chunking is for 8GB VRAM; CPU has RAM
        print(f"[init] warm start from {args.init_ckpt} (config "
              f"d_embed={cfg.d_embed} d_hidden={cfg.d_hidden} "
              f"blocks={cfg.num_hidden_blocks})", flush=True)
    else:
        cfg = WeatherNetConfig(
            d_hidden=args.width,
            num_hidden_blocks=args.depth,
            chunk_size=args.chunk_size or None,
        )
    model = WeatherNet(cfg)
    if args.init_ckpt and not os.path.exists(os.path.join(args.out, "latest.pt")):
        if st["config"]["d_hidden"] == cfg.d_hidden and \
                st["config"]["num_hidden_blocks"] == cfg.num_hidden_blocks:
            model.load_state_dict(st["model_state"])
            print(f"[init] loaded weights from {args.init_ckpt} "
                  f"(n_params={st.get('n_params', 0):,})", flush=True)
        else:
            print("[init] config mismatch -- starting fresh", flush=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {total_params:,} params ({total_params/1e9:.2f}B), "
          f"threads={args.threads}", flush=True)

    # ---- optimizer -----------------------------------------------------------
    opt = make_optimizer(args, model)
    step = 0
    resume_epoch = 0

    # ---- resume --------------------------------------------------------------
    if args.resume == "auto":
        latest = os.path.join(args.out, "latest.pt")
        if os.path.exists(latest):
            st = torch.load(latest, map_location="cpu", weights_only=False)
            if st.get("config") == vars(model.cfg):
                model.load_state_dict(st["model_state"])
                nm.load_state_dict(st["normalizer"])
                if st.get("opt_state") is not None:
                    opt.load_state_dict(st["opt_state"])
                step = int(st.get("step", 0))
                resume_epoch = int(st.get("epoch", 0))
                print(f"[resume] continuing from step {step} (epoch {resume_epoch})",
                      flush=True)
            else:
                print("[resume] config mismatch -- ignoring", flush=True)
    done_marker = os.path.join(args.out, "DONE")
    if os.path.exists(done_marker):
        print(f"[done] {done_marker} exists; not re-running.", flush=True)
        return
    os.makedirs(args.out, exist_ok=True)

    def save_ckpt(path: str, cur_step: int) -> None:
        ckpt = {
            "model_state": model.state_dict(),
            "config": vars(model.cfg),
            "normalizer": nm.state_dict(),
            "n_params": total_params,
            "dtype": "float32",
            "epoch": resume_epoch,
            "step": cur_step,
            "opt_state": opt.state_dict(),
        }
        tmp = path + ".tmp"
        torch.save(ckpt, tmp)
        os.replace(tmp, path)

    # ---- eval set ------------------------------------------------------------
    ev_x = np.load(os.path.join(args.cache_dir, "eval_x.npy"))
    ev_c = np.load(os.path.join(args.cache_dir, "eval_ctx.npy"))
    ev_y = np.load(os.path.join(args.cache_dir, "eval_y.npy"))
    ev_x = ev_x[:args.eval_limit]; ev_c = ev_c[:args.eval_limit]
    ev_y = ev_y[:args.eval_limit]
    from torch.utils.data import TensorDataset, DataLoader
    ev_ds = TensorDataset(
        torch.from_numpy(nm.transform_x(ev_x)),
        torch.from_numpy(nm.transform_ctx(ev_c)),
        torch.from_numpy(nm.transform_y(ev_y)),
    )
    ev_loader = DataLoader(ev_ds, batch_size=256, shuffle=False, num_workers=0)
    del ev_x, ev_c, ev_y
    print(f"[eval] {len(ev_ds):,} held-out windows", flush=True)

    # ---- streaming shard source ------------------------------------------------
    st = ShardTrainer(args.cache_dir, args.batch_size, rng)
    gen = st.stream(nm.x_mean, nm.x_std, nm.y_mean, nm.y_std)

    autocast_ctx = torch.autocast(device_type="cpu", enabled=False)

    print(f"[run] bf16=off fp32=on batch={args.batch_size} "
          f"grad_accum={args.grad_accum} max_steps={args.max_steps}", flush=True)

    t0 = time.time()
    mem_logged = 0
    for p in model.parameters():
        p.grad = None

    for gs in range(step, args.max_steps):
        opt.zero_grad(set_to_none=True)
        mb_loss = 0.0
        for mi in range(args.grad_accum):
            try:
                xw, ctx, yt = next(gen)
            except StopIteration:
                gen = st.stream(nm.x_mean, nm.x_std, nm.y_mean, nm.y_std)
                xw, ctx, yt = next(gen)
            with autocast_ctx:
                pred = model(xw, ctx)
            loss = huber(pred.float(), yt.float()) / args.grad_accum
            loss.backward()
            mb_loss += loss.item()

        lr = make_lr(gs, args.warmup_steps, args.max_steps, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        opt.step()
        for p in model.parameters():
            p.grad = None

        now = time.time()
        dt = now - t0
        if (gs + 1) % 50 == 0 or gs == step:
            rate = (gs - step + 1) * args.batch_size * args.grad_accum / max(dt, 1e-9)
            print(f"[step {gs+1}/{args.max_steps}] loss={mb_loss:.4f} "
                  f"{rate:5.1f} smp/s  {dt:6.0f}s", flush=True)
        if args.save_every > 0 and (gs + 1) % args.save_every == 0:
            save_ckpt(os.path.join(args.out, "latest.pt"), gs + 1)
            print(f"[save] step {gs+1} -> latest.pt", flush=True)
        if args.val_every > 0 and (gs + 1) % args.val_every == 0:
            model.eval()
            with torch.no_grad():
                m = evaluate(model, ev_loader, nm, torch.device("cpu"),
                             autocast_ctx)
            model.train()
            nnz, sp = ternary_stats(model)
            print(f"[val] step {gs+1} rmse={m['rmse']:.3f}C "
                  f"corr={m['corr']:.3f} ternary_active={nnz:.1%}", flush=True)
            import json as _json
            mfile = os.path.join(args.out, "metrics.json")
            hist = []
            if os.path.exists(mfile):
                try:
                    hist = _json.load(open(mfile))
                except Exception:
                    hist = []
            hist.append({"step": int(gs + 1), "rmse_c": round(m["rmse"], 3),
                         "corr": round(m["corr"], 4), "train_windows": int(gs + 1)})
            with open(mfile, "w") as f:
                _json.dump(hist, f, indent=1)
        if os.sched_yield and (gs + 1) % 10 == 0:
            os.sched_yield()   # politest mode: give the scheduler a breather

    save_ckpt(os.path.join(args.out, "final.pt"), args.max_steps)
    with open(done_marker, "w") as f:
        f.write(f"steps={args.max_steps}\n")
    print(f"[done] final.pt saved, {time.time()-t0:.0f}s total", flush=True)


if __name__ == "__main__":
    train(parse_args())