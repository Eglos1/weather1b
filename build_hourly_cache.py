#!/usr/bin/env python3
"""build_hourly_cache.py
========================
Build a training-window cache from the FULL global ISD-Lite HOURLY archive
for WeatherNet CPU training.

Unlike `map_from_hourly.py` (which only looks at the tail of each station's
record to make maps), this makes a *training set*:

  - every hourly observation record for every station with valid coordinates
    is read from the yearly ``USAF-WBAN-year.gz`` files (1973..2025);
  - all hours of a calendar day are aggregated to a daily mean of
    [T, P, RH, wind] (same math as map_from_hourly.daily_rows);
  - a sliding 48-day window (24 input + 24 target) is walked over each
    station's chronological record, stride 24 days; any window that straddles
    a gap in the record is discarded (same semantics as WeatherZipCSV);
  - windows are saved as shards of float32 .npy arrays,
  - online mean/std stats (per input variable + per forecast step) are saved
    to a manifest.json so the trainer can build its Normalizer once,
  - the first ~EVAL_N windows are held out as an eval set.

The whole job is embarrassingly parallel per station; run it with
`nice -n 15 ionice -c 3 python3` so it never fights the user's foreground
apps or the LLM trainer.

Output layout (CACHE_DIR):
    x_<shard>.npy      (n, 24, 4) float32   input windows
    ctx_<shard>.npy    (n, 4)     float32   per-window rolling means
    y_<shard>.npy      (n, 24)    float32   target = next 24 daily temps
    eval_{x,ctx,y}.npy                      held-out windows (small)
    manifest.json                            stats + shard list + counts
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import sys
import time
from datetime import date as _date

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

HOUR_DIR = "/mnt/biggiecheese/weather1b/hourly/isd-lite"
HIST = os.path.join(ROOT, "data", "isd-history.csv")

WIN = 24                # input window length (days)
TGT = 24                # forecast length (days)
STRIDE = WIN            # non-overlapping window walk
MIN_OBS_PER_DAY = 4     # hourly obs needed to trust a calendar day
SHARD = 50_000          # windows per .npy shard (50k * 496 B ~= 25 MB)
EVAL_N = 20_000         # held-out windows for RMSE/corr reporting

F64 = np.float64


def rh_from_dew(tc: float, tdc: float) -> float:
    a = (17.625 * tdc) / (243.04 + tdc)
    b = (17.625 * tc) / (243.04 + tc)
    return 100.0 * (math.exp(a) / math.exp(b))


def daily_rows(path: str):
    """Yield (date_string, [t, p, rh, wind]) daily means from one hourly .gz.

    ISD-Lite fixed/space cols: Y M D H T(deciC) TD(deciC) SLP(deciPa)
    WDIR(deci?) WSPD(deci m/s) SKY P1 P6   (-9999 = missing).
    """
    sums: dict = {}
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            p = line.split()
            if len(p) < 12:
                continue
            try:
                y, m, d = int(p[0]), int(p[1]), int(p[2])
                t = float(p[4]); td = float(p[5])
                slp = float(p[6]); ws = float(p[8])
            except ValueError:
                continue
            if t <= -9000 or td <= -9000 or slp <= -9000 or ws <= -9000:
                continue
            key = (y, m, d)
            s = sums.get(key)
            if s is None:
                sums[key] = [t, td, slp, ws, 1]
            else:
                s[0] += t; s[1] += td; s[2] += slp; s[3] += ws; s[4] += 1
    out = []
    for (y, m, d), (st, std, sp, sw, n) in sums.items():
        if n < MIN_OBS_PER_DAY:
            continue
        tc = st / n / 10.0
        tdc = std / n / 10.0
        out.append((f"{y:04d}-{m:02d}-{d:02d}",
                    [tc, sp / n / 10.0, rh_from_dew(tc, tdc), sw / n / 10.0]))
    return out


def _station_windows(args) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Aggregate one station's yearly files and slide daily windows.

    args = (sid, path_list).  Returns stacked (x, ctx, y) float32 arrays or
    None when the station has no usable 48-day window.
    """
    import gzip
    sid, paths = args
    rows: list = []
    for p in paths:
        rows.extend(daily_rows(p))
    if len(rows) < WIN + TGT:
        return None

    rows.sort(key=lambda r: r[0])
    recs = []
    for ds, vals in rows:
        y, m, d = map(int, ds.split("-"))
        recs.append((_date(y, m, d), np.asarray(vals, dtype=np.float32)))

    # split the chronological record into runs of *consecutive* calendar days
    runs = []
    cur: list[np.ndarray] = []
    prev = None
    for dt, v in recs:
        if prev is not None and (dt - prev).days != 1:
            if len(cur) >= WIN + TGT:
                runs.append(cur)
            cur = []
        cur.append(v)
        prev = dt
    if len(cur) >= WIN + TGT:
        runs.append(cur)

    outs = []
    for run in runs:
        n = len(run)
        for i in range(0, n - (WIN + TGT) + 1, STRIDE):
            blk = np.stack(run[i:i + WIN + TGT])   # (48, 4)
            x = blk[:WIN]
            outs.append((x, x.mean(axis=0).astype(np.float32),
                         blk[WIN:, 0].copy()))
    if not outs:
        return None
    xs = np.stack([o[0] for o in outs])
    cs = np.stack([o[1] for o in outs])
    ys = np.stack([o[2] for o in outs])
    return xs, cs, ys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, "hourly_cache"))
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-stations", type=int, default=0,
                    help="restrict processing to the first N shuffled stations "
                         "(0 = all; useful for smoke tests)")
    ap.add_argument("--eval-n", type=int, default=EVAL_N,
                    help="how many held-out eval windows to reserve "
                         "(keep small for smoke runs)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny test: 40 stations, 2 shards max")
    args = ap.parse_args()
    if args.smoke and args.eval_n >= EVAL_N:
        args.eval_n = 200   # smoke runs are tiny; keep some windows for training

    cache = args.cache_dir
    os.makedirs(cache, exist_ok=True)
    t0 = time.time()

    # ---- station catalog ---------------------------------------------------
    stations: list[str] = []
    with open(HIST, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                sa, wb, la, lo = (row["USAF"].strip(), row["WBAN"].strip(),
                                  float(row["LAT"]), float(row["LON"]))
            except (KeyError, ValueError):
                continue
            if not (-90 <= la <= 90 and -180 <= lo <= 180
                    and abs(la) > 0.01 and abs(lo) > 0.01):
                continue
            stations.append(f"{sa.zfill(6)}-{wb.zfill(5)}")
    stations = list(dict.fromkeys(stations))        # dedupe, keep order
    rng = random.Random(args.seed)
    rng.shuffle(stations)
    print(f"[catalog] {len(stations)} stations with valid coords", flush=True)
    if args.max_stations:
        stations = stations[:args.max_stations]

    # ---- file index: year -> set of filenames (one os.listdir each) -------
    years = sorted((y for y in os.listdir(HOUR_DIR) if y.isdigit()))
    files_by_year = {y: set(os.listdir(os.path.join(HOUR_DIR, y)))
                     for y in years}
    total_files = sum(len(s) for s in files_by_year.values())
    print(f"[archive] years {years[0]}..{years[-1]}, {total_files:,} files",
          flush=True)

    tasks: list[tuple[str, list[str]]] = []
    for sid in stations:
        paths = []
        for y, fset in files_by_year.items():
            fn = f"{sid}-{y}.gz"
            if fn in fset:
                paths.append(os.path.join(HOUR_DIR, y, fn))
        if paths:
            tasks.append((sid, paths))
    print(f"[archive] {len(tasks)} stations with >=1 yearly file "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- parallel aggregation ----------------------------------------------
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    pool = ctx.Pool(args.workers)

    # online stats accumulators (over the TRAIN windows only)
    # NOTE: use longdouble so large-mean variables (pressure ~1020 hPa) don't
    # lose their variance to float64 cancellation across ~76M rows.
    n_tr = 0
    xs_sum = np.zeros(4, dtype=np.longdouble)
    xs_sq = np.zeros(4, dtype=np.longdouble)
    ys_sum = np.zeros(TGT, dtype=np.longdouble)
    ys_sq = np.zeros(TGT, dtype=np.longdouble)

    n_eval = 0
    eval_n = args.eval_n
    n_win = 0
    used_shards = 0
    shard_idx = 0

    ebuf = [[], [], []]
    tbuf = [[], [], []]

    def nonlocal_shard(i: int, x, c, y) -> None:
        # write shard i into cache dir
        np.save(os.path.join(cache, f"x_{i:06d}.npy"), x)
        np.save(os.path.join(cache, f"ctx_{i:06d}.npy"), c)
        np.save(os.path.join(cache, f"y_{i:06d}.npy"), y)

    stations_done = 0
    windows_from_station = 0
    for res in pool.imap_unordered(_station_windows, tasks, chunksize=1):
        stations_done += 1
        if res is None:
            if stations_done % 200 == 0:
                print(f"[agg] {stations_done}/{len(tasks)} stations, "
                      f"{n_win:,} windows [{time.time()-t0:.0f}s]", flush=True)
            continue
        xs, cs, ys = res
        n_win += len(xs)
        # hold out the first EVAL_N windows seen
        rem = eval_n - n_eval
        if rem > 0:
            take = min(rem, len(xs))
            ebuf[0].append(xs[:take]); ebuf[1].append(cs[:take])
            ebuf[2].append(ys[:take]); n_eval += take
            xs, cs, ys = xs[take:], cs[take:], ys[take:]
        if len(xs):
            tbuf[0].append(xs); tbuf[1].append(cs); tbuf[2].append(ys)
            while sum(len(a) for a in tbuf[0]) >= SHARD:
                # split tbuf into one shard + remainder
                flat = (np.concatenate(tbuf[0]), np.concatenate(tbuf[1]),
                        np.concatenate(tbuf[2]))
                x0, c0, y0 = (flat[0][:SHARD], flat[1][:SHARD], flat[2][:SHARD])
                if len(x0) >= SHARD:
                    nonlocal_shard(shard_idx, x0, c0, y0)
                    shard_idx += 1
                    used_shards += 1
                    n_tr += SHARD
                    xs_sum += x0.reshape(-1, 4).astype(np.longdouble).sum(axis=0)
                    xs_sq += (x0.reshape(-1, 4).astype(np.longdouble) ** 2).sum(axis=0)
                    ys_sum += y0.astype(np.longdouble).sum(axis=0)
                    ys_sq += (y0.astype(np.longdouble) ** 2).sum(axis=0)
                # keep the tail for the next shard
                tbuf[0] = [flat[0][SHARD:]]; tbuf[1] = [flat[1][SHARD:]]
                tbuf[2] = [flat[2][SHARD:]]
        if stations_done % 200 == 0:
            print(f"[agg] {stations_done}/{len(tasks)} stations, "
                  f"{n_win:,} windows [{time.time()-t0:.0f}s]", flush=True)
        if args.smoke and used_shards >= 2:
            break
    pool.close(); pool.join()

    # ---- flush the final partial train shard -------------------------------
    if tbuf[0] and len(tbuf[0][0]) > 0:
        x = np.concatenate(tbuf[0]); c = np.concatenate(tbuf[1])
        y = np.concatenate(tbuf[2])
        nonlocal_shard(shard_idx, x, c, y)
        shard_idx += 1
        used_shards += 1
        n_tr += len(x)
        xs_sum += x.reshape(-1, 4).astype(np.longdouble).sum(axis=0)
        xs_sq += (x.reshape(-1, 4).astype(np.longdouble) ** 2).sum(axis=0)
        ys_sum += y.astype(np.longdouble).sum(axis=0)
        ys_sq += (y.astype(np.longdouble) ** 2).sum(axis=0)

    # ---- eval set ----------------------------------------------------------
    if n_eval > 0:
        np.save(os.path.join(cache, "eval_x.npy"), np.concatenate(ebuf[0]))
        np.save(os.path.join(cache, "eval_ctx.npy"), np.concatenate(ebuf[1]))
        np.save(os.path.join(cache, "eval_y.npy"), np.concatenate(ebuf[2]))

    # ---- stats (train windows only) ---------------------------------------
    if n_tr == 0:
        raise SystemExit("no training windows produced")
    nx = n_tr * WIN
    x_mean = xs_sum / nx
    x_std = np.clip(xs_sq / nx - x_mean * x_mean, 1e-12, None) ** 0.5
    y_mean = ys_sum / n_tr
    y_std = np.clip(ys_sq / n_tr - y_mean * y_mean, 1e-12, None) ** 0.5

    manifest = {
        "source": "ISD-Lite hourly global (1973..2025)",
        "shards": [f"{i:06d}" for i in range(used_shards)],
        "train_windows": int(n_tr),
        "eval_windows": int(n_eval),
        "stations_done": stations_done,
        "stats": {
            "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
            "y_mean": y_mean.tolist(), "y_std": y_std.tolist(),
        },
        "seed": args.seed,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(cache, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    print(f"[done] {n_win:,} windows ({n_tr:,} train / {n_eval:,} eval) -> "
          f"{cache} in {time.time()-t0:.0f}s", flush=True)
    print(f"[stats] x_mean={x_mean.round(2).tolist()} "
          f"x_std={x_std.round(2).tolist()}", flush=True)
    print(f"[stats] y_mean={y_mean.round(2).tolist()} "
          f"y_std={y_std.round(2).tolist()}", flush=True)


if __name__ == "__main__":
    main()