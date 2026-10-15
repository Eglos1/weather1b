#!/usr/bin/env python3
"""Train/enforce a ByteLevel BPE vocab and maintain train.bin + val.bin.

Modes:
  default       - full rebuild: snapshot ALL shard text into RAM, tokenize the
                  complete corpus, write *.new, verify, os.replace.
  --incremental - absorb only shards NOT already recorded in the manifest
                  (data/bin/.manifest.json); append atomically with the same
                  verify step. Never re-adds old shards, never drops them.

A write lock (data/bin/.build.lock) prevents racing builders. Any failure
leaves the old bins untouched. Exit 0 only when bins match the vocab."""
import argparse, glob, json, os, sys, time
import numpy as np
from tokenizers import Tokenizer, models, trainers, decoders, normalizers, pre_tokenizers

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
TXTD = os.path.join(DATA, "text")
bindir = os.path.join(DATA, "bin")
vocdir = os.path.join(DATA, "vocab")
os.makedirs(bindir, exist_ok=True)
os.makedirs(vocdir, exist_ok=True)

VOCAB = 16384
SEP = "<|endoftext|>"
VOCAB_PATH = os.path.join(vocdir, "bpe.json")
TRAIN_BIN = os.path.join(bindir, "train.bin")
VAL_BIN = os.path.join(bindir, "val.bin")
MANI = os.path.join(bindir, ".manifest.json")
LOCK = os.path.join(bindir, ".build.lock")


def shards(split):
    return sorted(
        p for p in glob.glob(os.path.join(TXTD, "*.txt"))
        if f"_{split}_" in os.path.basename(p))


def acquire_lock():
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL)
        os.close(fd)
        return True
    except FileExistsError:
        age = time.time() - os.path.getmtime(LOCK)
        if age > 3600:  # stale (>1h) -> steal
            try:
                os.remove(LOCK)
                fd = os.open(LOCK, os.O_CREAT | os.O_EXCL)
                os.close(fd)
                return True
            except Exception:
                return False
        return False


def release_lock():
    try:
        os.remove(LOCK)
    except FileNotFoundError:
        pass


def load_manifest():
    try:
        with open(MANI) as f:
            return json.load(f)
    except Exception:
        return {}


def save_manifest(m):
    with open(MANI + ".new", "w") as f:
        json.dump(m, f)
    os.replace(MANI + ".new", MANI)


def stamp(path):
    st = os.stat(path)
    return [st.st_mtime_ns * 1.0, st.st_size]


def train_tokenizer(texts):
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.Sequence(
        [normalizers.NFKC(), normalizers.Replace("’", "'"),
         normalizers.Lowercase()])
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB,
                                  min_frequency=2,
                                  special_tokens=[SEP],
                                  show_progress=False)
    tok.train(texts, trainer)
    tok.save(VOCAB_PATH)
    return tok


def encode_shards(tok, paths, sep_id, chunk_lines=200_000):
    """Encode files in bounded chunks so peak memory stays flat
    (a 778MB shard as ONE encode_batch can transiently spike into
    multi-GB line-object lists)."""
    ids = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            lines = f.read().splitlines()
        for i in range(0, len(lines), chunk_lines):
            enc = tok.encode_batch(lines[i:i + chunk_lines],
                                   add_special_tokens=False)
            for e in enc:
                ids.extend(e.ids)
        ids.append(sep_id)
    return ids


def verify_and_publish(tok, tmp, final, what):
    a = np.fromfile(tmp, dtype=np.uint16)
    vsz = tok.get_vocab_size()
    if a.size:
        mx = int(a.max())
        if mx >= vsz:
            os.remove(tmp)
            raise SystemExit(
                f"[tok] FATAL {what}: max id {mx} >= vocab {vsz}; NOT published")
    os.replace(tmp, final)
    print(f"[tok] {what}: {a.size/1e6:.2f}M tokens, max id {a.max() if a.size else 0} "
          f"< vocab {vsz}", flush=True)


def publish(split, tok, arr, mani, what):
    final = TRAIN_BIN if split == "train" else VAL_BIN
    tmp = final + ".new"
    arr.tofile(tmp)
    verify_and_publish(tok, tmp, final, what)

def full_rebuild(tok):
    """Snapshot ALL text, encode the whole corpus, publish both bins."""
    train_shards = shards("train")
    val_shards = shards("validation") + shards("test")
    if not train_shards:
        print("[tok] no train text yet")
        return
    snap = {}
    for p in train_shards + val_shards:
        with open(p, encoding="utf-8") as f:
            snap[p] = f.read()

    if os.path.exists(VOCAB_PATH):
        tok = Tokenizer.from_file(VOCAB_PATH)
        print(f"[tok] loaded vocab {tok.get_vocab_size()}")
    else:
        sample = [snap[p] for p in train_shards[:4] + val_shards[:2]]
        tmp_txt = os.path.join(bindir, "_train_sample.txt")
        with open(tmp_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(sample))
        print(f"[tok] training fresh BPE on {len(sample)} shards...")
        tok = train_tokenizer([tmp_txt])
        os.remove(tmp_txt)
        print(f"[tok] vocab {tok.get_vocab_size()}")

    sep_id = tok.token_to_id(SEP)
    ta = np.asarray(encode_shards(tok, train_shards, sep_id), dtype=np.uint16)
    va = np.asarray(encode_shards(tok, val_shards, sep_id), dtype=np.uint16)

    publish("train", tok, ta, None, "train.bin")
    publish("val", tok, va, None, "val.bin")

    # seed / refresh the manifest with every shard now in the corpus
    m = {}
    for split in ("train", "validation", "test"):
        for p in shards(split):
            m[p] = stamp(p)
    save_manifest(m)


def incremental(tok):
    """Append shards missing from the manifest (newest corpora first)."""
    mani = load_manifest()
    added = 0
    for split in ("train", "validation", "test"):
        final = TRAIN_BIN if split == "train" else VAL_BIN
        label = os.path.basename(final)
        if not os.path.exists(final):
            print(f"[tok] incremental {label} missing -> full rebuild needed")
            return False
        fresh = [p for p in shards(split)
                 if mani.get(p) != stamp(p)]
        if not fresh:
            print(f"[tok] incremental {label}: nothing new", flush=True)
            continue
        txt = sum(os.path.getsize(p) for p in fresh) / 1e6
        print(f"[tok] incremental {label}: +{len(fresh)} shards, "
              f"{txt:.1f} MB of new text", flush=True)
        old = np.fromfile(final, dtype=np.uint16)
        sep_id = tok.token_to_id(SEP)
        extra = np.asarray(encode_shards(tok, fresh, sep_id), dtype=np.uint16)
        if extra.size < 100:
            print(f"[tok] incremental {label}: {extra.size} tok (skipping)", flush=True)
            continue
        combined = np.concatenate([old, extra])
        publish(split, tok, combined, None, label)
        added += len(extra)
        for p in fresh:
            mani[p] = stamp(p)
    save_manifest(mani)
    print(f"[tok] incremental done, +{added/1e6:.2f}M tokens total")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incremental", action="store_true",
                    help="absorb only shards missing from the manifest")
    args = ap.parse_args()

    if not acquire_lock():
        print("[tok] another build is running; skipping")
        return
    try:
        if args.incremental and os.path.exists(VOCAB_PATH):
            tok = Tokenizer.from_file(VOCAB_PATH)
            print(f"[tok] incremental mode, vocab {tok.get_vocab_size()}")
            if incremental(tok):
                return
            print("[tok] falling back to full rebuild")
        full_rebuild(None)
    finally:
        release_lock()


if __name__ == "__main__":
    main()