#!/usr/bin/env python3
"""Train the ~500M People-Speech-Text LLM. bf16 on GPU (low VRAM),
Adafactor, crash-proof resume (latest.pt + DONE marker), metrics.json."""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.gpt import GPT, GPTConfig  # noqa: E402


def resolve(args):
    if args.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return args.device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-bin", default="data/bin/train.bin")
    ap.add_argument("--val-bin", default="data/bin/val.bin")
    ap.add_argument("--out", default="runs/llm500")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--seq", "--seq-len", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=250_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--resume", default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--width", type=int, default=0)   # 0 = default config
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = resolve(args)
    print(f"[llm] device={dev}")

    HERE = os.path.dirname(os.path.abspath(__file__))
    data = None
    for c in (args.data_bin,
              os.path.join(HERE, args.data_bin),
              os.path.join(HERE, "data", "bin", "train.bin")):
        if os.path.exists(c):
            data = np.memmap(c, dtype=np.uint16, mode="r")
            break
    if data is None:
        sys.exit(f"[llm] no train.bin found ({args.data_bin})")
    ntoks = data.shape[0]
    print(f"[llm] train tokens: {ntoks/1e6:.2f}M")

    val_t = None
    for v in (args.val_bin,
              os.path.join(HERE, args.val_bin),
              os.path.join(HERE, "data", "bin", "val.bin")):
        if os.path.exists(v):
            val_t = np.memmap(v, dtype=np.uint16, mode="r")
            break
    if val_t is not None:
        print(f"[llm] val tokens: {val_t.shape[0]/1e6:.2f}M")

    cfg = GPTConfig()
    # derive vocab size from our BPE tokenizer (json), not the default
    vocab_json = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "data", "vocab", "bpe.json")
    if os.path.exists(vocab_json):
        with open(vocab_json) as f:
            tj = json.load(f)
        vsize = len(tj["model"]["vocab"]) + len(tj.get("added_tokens", []))
        cfg.vocab_size = vsize
        print(f"[llm] tokenizer vocab size {vsize}")
    # bins MUST use ids < vocab or the GPU embedding aborts the whole driver
    mx = int(np.fromfile("data/bin/train.bin", dtype=np.uint16).max())
    if mx >= cfg.vocab_size:
        sys.exit(f"[llm] FATAL: train.bin max id {mx} >= vocab {cfg.vocab_size}; "
                 f"run build_bpe.py to rebuild bins")
    print(f"[llm] verified train.bin max id {mx} < vocab {cfg.vocab_size}")
    model = GPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    ck_path = os.path.join(args.out, "latest.pt")
    done = os.path.join(args.out, "DONE")
    step = 0
    if args.resume == "auto" and os.path.exists(ck_path):
        st = torch.load(ck_path, map_location="cpu", weights_only=False)
        old_vocab = st.get("config", {}).get("vocab_size", -1)
        if old_vocab != cfg.vocab_size:
            print(f"[llm] ckpt vocab {old_vocab} != current {cfg.vocab_size}; "
                  f"starting fresh")
        else:
            model.load_state_dict(st["model"])
            step = int(st.get("step", 0))
            print(f"[llm] resumed at step {step}")
    if os.path.exists(done):
        print(f"[llm] DONE marker present -- finished earlier.")
        return

    model = model.to(dev)
    if dev == "cuda":
        model = model.to(torch.bfloat16)

    opt = torch.optim.Adafactor(
        model.parameters(), lr=args.lr,
        weight_decay=0.0, eps=(1e-16, 1e-3))
    if os.path.exists(ck_path) and step > 0:
        opt.load_state_dict(st.get("opt", {}))

    B, T = args.batch_size, args.seq
    gbs = B * args.grad_accum
    steps = args.max_steps
    t0 = time.time()
    last = t0
    metrics = []

    def save_ckpt(s):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "step": s, "n_params": n_params,
                    "config": vars(cfg)}, ck_path + ".tmp")
        os.replace(ck_path + ".tmp", ck_path)

    def eval_loss():
        if val_t is None:
            return float("nan")
        model.eval()
        nt = val_t.shape[0]
        nwin = (nt // T) // B
        b = max(1, min(8, nwin))
        idxs = np.random.default_rng(step).choice(nwin, size=b, replace=False)
        loss = []
        with torch.no_grad():
            for i in idxs:
                xb = torch.tensor(
                    np.asarray(val_t[i * B * T:(i + 1) * B * T]).reshape(B, T),
                    dtype=torch.long).to(dev)
                _, l = model(xb, xb)
                loss.append(l.item())
        model.train()
        return float(np.mean(loss))

    print(f"[llm] gbs={gbs} seq={T} steps={steps}")
    try:
        v0 = eval_loss()
        print(f"[llm] startup eval_loss {v0:.3f}", flush=True)
    except Exception as e:
        print(f"[llm] startup eval FAILED: {type(e).__name__} {str(e)[:100]}",
              flush=True)
    while step < steps:
        model.train()
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for mb in range(args.grad_accum):
            # random contiguous windows from the big memmap
            off = np.random.randint(0, ntoks - B * T - 1)
            xb = torch.tensor(
                np.asarray(data[off:off + B * T]).reshape(B, T).copy(),
                dtype=torch.long).to(dev)
            # NOTE: casting the input to bf16 breaks the embedding (needs int);
            # model is bf16 so matmuls still run in bf16
            _, loss = model(xb, xb)
            acc += loss.item()
            (loss / args.grad_accum).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        loss_ = acc / args.grad_accum
        tok = gbs * T

        if step % 25 == 0 or step == steps:
            el = time.time() - last
            last = time.time()
            rate = tok / max(el, 1e-6)
            print(f"[llm {step}/{steps}] loss {loss_:.4f}  "
                  f"{rate:.0f} tok/s  {time.time()-t0:.0f}s", flush=True)

        if step % args.save_every == 0:
            save_ckpt(step)
            vl = eval_loss() if step % args.val_every == 0 else float("nan")
            metrics.append({"step": step, "train_loss": loss_,
                            "val_loss": vl,
                            "tok_s": tok / max(time.time() - last, 1e-6)})
            with open(os.path.join(args.out, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=1)
            print(f"[ckpt] step {step} -> latest.pt "
                  f"(val {vl:.4f})", flush=True)

    save_ckpt(step)
    with open(done, "w") as f:
        f.write(f"step={step}\n")
    print(f"[done] trained {step} steps in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()