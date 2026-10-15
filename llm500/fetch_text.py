#!/usr/bin/env python3
"""Fetch ONLY the text part of MLCommons/peoples_speech via the HF
datasets-server /rows endpoint (audio is just a URL there, so no audio
bytes are downloaded). Writes one transcript per line to shard files.
Resumable: state is kept in data/fetch_state.json and we append shards."""
import json, os, random, re, sys, time, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
TXTD = os.path.join(DATA, "text")
os.makedirs(TXTD, exist_ok=True)
STATE = os.path.join(DATA, "fetch_state.json")

CONFIG = os.environ.get("PS_CONFIG", "clean")
SPLIT = os.environ.get("PS_SPLIT", "train")
MAX_ROWS = int(os.environ.get("PS_MAX_ROWS", str(60_000_000)))  # cap (whole split)
STEP = 100  # max rows per datasets-server request

BASE = "https://datasets-server.huggingface.co/rows?dataset=MLCommons/peoples_speech&config={cfg}&split={sp}&offset={off}&length={len}"

WS = re.compile(r"\s+")


def state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {}


def save_state(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.replace(tmp, STATE)


def fetch(off):
    url = BASE.format(cfg=CONFIG, sp=SPLIT, off=off, len=STEP)
    for attempt in range(14):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                d = json.load(r)
            rows = d.get("rows") or []
            ntotal = d.get("num_rows_total")
            return rows, ntotal
        except Exception as e:
            wait = min(2 ** attempt, 60) + random.random() * 0.4
            print(f"[fetch] offset {off} error {e!r}; retry in {wait:.0f}s",
                  flush=True)
            time.sleep(wait)
    return None, None


def main():
    st = state()
    key = f"{CONFIG}:{SPLIT}"
    off = st.get(key, {}).get("offset", 0)
    total = st.get(key, {}).get("total")

    shard_idx = off // 20000
    fpath = os.path.join(TXTD, f"{CONFIG}_{SPLIT}_{shard_idx:06d}.txt")
    fh = open(fpath, "a", encoding="utf-8")
    written = 0
    last = time.time()

    while off < MAX_ROWS:
        rows, ntotal = fetch(off)
        if rows is None:
            print("[fetch] giving up at offset", off, flush=True)
            break
        if total is None and ntotal:
            total = ntotal
            st[key] = st.get(key, {}); st[key]["total"] = total
        wrote = 0
        for r in rows:
            t = (r.get("row") or {}).get("text")
            if t and t.strip():
                t = " ".join(t.split())
                fh.write(t + "\n")
                wrote += 1
        fh.flush()
        written += wrote
        off += len(rows)
        # roll into a new shard every 20k rows
        nshard = off // 20000
        if nshard != shard_idx:
            fh.close()
            shard_idx = nshard
            fpath = os.path.join(TXTD, f"{CONFIG}_{SPLIT}_{shard_idx:06d}.txt")
            fh = open(fpath, "a", encoding="utf-8")
        # persist progress every ~50 requests
        if off % (STEP * 50) == 0:
            st[key] = st.get(key, {})
            st[key]["offset"] = off
            save_state(st)
            el = time.time() - last
            print(f"[fetch] {CONFIG}/{SPLIT} offset {off}/{total or '?'} "
                  f"rows, lines {written} last-50req {el:.1f}s", flush=True)
            last = time.time()
        time.sleep(0.09)  # polite rate limit
    st[key] = st.get(key, {})
    st[key]["offset"] = off
    save_state(st)
    fh.close()
    print(f"[fetch] done {CONFIG}/{SPLIT}: reached offset {off}, "
          f"rows written {written}", flush=True)


if __name__ == "__main__":
    main()