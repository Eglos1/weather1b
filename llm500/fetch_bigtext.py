#!/usr/bin/env python3
"""Big plain-text corpora for the LLM, staged as *_train_*.txt shards.

  --source enwik9     mattmahoney.net Wikipedia dump (1GB raw; classic bench set)
  --source gutenberg  www.gutenberg.org cache endpoint (curated + today's feed)
  --rebuild           run build_bpe.py --incremental after fetching

Both sources are resumable (skip existing shards) and polite (slow pacing).
Copyright note: enwik9 = Wikipedia CC BY-SA text; Gutenberg books = public
domain. Both fine for a private research corpus.
"""
import argparse, gzip, io, json, os, re, subprocess, sys, time, urllib.request, zipfile, zlib

ROOT = os.path.dirname(os.path.abspath(__file__))
TXTD = os.path.join(ROOT, "data", "text")
os.makedirs(TXTD, exist_ok=True)
UA = {"User-Agent": "weather1b-llm/1.0 (local research corpus)"}


def get(url, timeout=900):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parse_enwik(text):
    """Raw Wikipedia XML -> de-marked-up article bodies."""
    out = []
    for m in re.finditer(r"<text[^>]*>(.*?)</text>", text, re.S):
        body = m.group(1)
        body = re.sub(r"<!--.*?-->", " ", body, flags=re.S)
        body = re.sub(r"\{\{[^{}]*\}\}", " ", body)          # templates (1 pass)
        body = re.sub(r"\[\[(Category|File|Image|Media|Help|Portal):.*?\]\]", " ", body, flags=re.S)
        body = re.sub(r"\[\[([^\]|]*)\|([^\]]*)\]\]", r"\2", body)  # links -> display txt
        body = re.sub(r"\[\[([^\]]*)\]\]", r"\1", body)
        body = re.sub(r"'{2,}", "", body)                    # wiki bold/italic
        body = re.sub(r"(===?+)", r"===", body)              # normalize headings
        body = re.sub(r"<[^>]+>", " ", body)                 # remaining tags
        out.append(body)
    return "\n\n".join(out)


def fetch_enwik9():
    dest = os.path.join(TXTD, "wiki_train_0.txt")
    if os.path.exists(dest) and os.path.getsize(dest) > 5_000_000:
        print(f"[big] enwik9 shard exists ({os.path.getsize(dest)/1e6:.0f} MB), skipping")
        return
    print("[big] downloading enwik9.zip (~1GB) ...", flush=True)
    data = get("https://mattmahoney.net/dc/enwik9.zip")
    print(f"[big] got {len(data)/1e6:.0f} MB compressed; decompressing", flush=True)
    if data[:2] == b"PK":  # it's a real zip
        zf = zipfile.ZipFile(io.BytesIO(data))
        name = next((n for n in zf.namelist() if "enwik9" in n), zf.namelist()[0])
        raw = zf.read(name)
    elif data[:2] == b"\x1f\x8b":
        raw = gzip.decompress(data)
    else:
        raw = zlib.decompress(data, 16 + zlib.MAX_WBITS)
    print(f"[big] decompressed {len(raw)/1e6:.0f} MB raw XML; stripping markup", flush=True)
    text = parse_enwik(raw.decode("utf-8", errors="replace"))
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[big] wrote {dest} ({os.path.getsize(dest)/1e6:.0f} MB)", flush=True)


CURATED_GUTENBERG = [  # famous public-domain titles by PG id
    11, 84, 1342, 2701, 74, 76, 1400, 98, 1661, 345, 174, 36, 46, 1952,
    2600, 30254, 100, 43, 1232, 1998, 219, 120, 2554, 2542, 1727, 1260,
    160, 236, 5200, 6130, 4763, 41, 17617, 2383, 245, 1080, 1404, 1184,
    1497, 145, 1934, 2852, 244, 32, 25494, 935, 955, 215, 6593, 8387,
    6420, 846, 2814, 1585, 4300, 141, 1909, 2680, 3825, 2500, 5, 2591,
    19942, 2409, 5827, 550, 2160, 30360, 28885, 195, 4078, 37106, 285,
    1073, 14725, 10031, 766, 1250, 2895, 2204, 60, 17405, 3160, 16, 2255,
    113, 471, 2344, 103, 1695, 131, 92, 12581, 1971, 480, 9947, 861,
]
STATE = os.path.join(TXTD, "gutenberg_state.json")


def fetch_gutenberg(limit):
    try:
        with open(STATE) as f:
            done = set(json.load(f))
    except Exception:
        done = set()
    # today's new-books feed, then curated books — both deduped against `done`
    todo = []
    try:
        feed = get("https://www.gutenberg.org/cache/epub/feeds/today.json", timeout=120).decode("utf-8")
        for m in re.finditer(r'"id":\s*"?(\d+)"?', feed):
            pid = int(m.group(1))
            if pid not in done:
                todo.append(pid)
        print(f"[big] today feed adds {len(todo)} new ids", flush=True)
    except Exception as e:
        print(f"[big] today feed failed ({e}); using curated list", flush=True)
    for pid in CURATED_GUTENBERG:
        if pid not in done:
            todo.append(pid)
    got = 0
    for pid in todo:
        if got >= limit:
            break
        dest = os.path.join(TXTD, f"gutenberg_train_{pid}.txt")
        if os.path.exists(dest):
            done.add(pid)
            continue
        url = f"https://www.gutenberg.org/cache/epub/{pid}/pg{pid}.txt"
        try:
            raw = get(url, timeout=60).decode("utf-8", errors="replace")
        except Exception as e:
            print(f"[big] pg{pid} skip: {e}", flush=True)
            continue
        m = re.search(r"\*\*\* START OF .+?\*\*\*(.*?)\*\*\* END OF .+?\*\*\*", raw, re.S)
        if m:
            raw = m.group(1)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(raw)
        done.add(pid)
        with open(STATE, "w") as f:
            json.dump(sorted(done), f)
        got += 1
        print(f"[big] pg{pid} -> {os.path.getsize(dest)/1e3:.0f} KB (total {len(done)})", flush=True)
        time.sleep(1.2)  # polite pace
    print(f"[big] gutenberg done: {got} new this run", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["enwik9", "gutenberg"], required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if args.source == "enwik9":
        fetch_enwik9()
    else:
        fetch_gutenberg(args.limit)

    if args.rebuild:
        print("[big] incremental corpus rebuild ...", flush=True)
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "build_bpe.py"), "--incremental"])
        sys.exit(r.returncode)
    sys.exit(0)


if __name__ == "__main__":
    main()