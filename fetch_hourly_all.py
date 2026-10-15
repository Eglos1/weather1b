#!/usr/bin/env python3
"""Bulk-harvest NOAA ISD-Lite HOURLY observations for EVERY station in the
isd-history catalog (29k+ stations worldwide, 1973-present).

Downloads raw per-station-per-year .gz files into:
    data/hourly/isd-lite/{YYYY}/{SA}-{WBAN}-{YYYY}.gz
(mirroring NOAA's own layout). State (completed id-year pairs + bytes +
rate) is persisted in data/hourly/isdfetch_state.json so any interruption
resumes exactly where it left off. Throttled + retried; polite to NCEI.

Line format (fixed width, see isd-lite-format.txt):
  year month day hour  tempC*10 dewC*10 presHPa*10 wdir ws*10 sky p1h*10 p6h*10
  (-9999 = missing; ws is m/s*10, pres is hPa*10)
"""
import csv, json, os, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite"
ROOT = os.path.dirname(os.path.abspath(__file__))
# Hourly harvest lives on the 8 TB btrfs drive (biggiecheese), not the SSD.
BIG = "/mnt/biggiecheese/weather1b"
OUT = os.path.join(BIG, "hourly", "isd-lite")
STATE = os.path.join(BIG, "hourly", "isdfetch_state.json")
LOG = "/tmp/hourly_fetch.log"
MIN_YEAR, MAX_YEAR = 1973, 2026
MIN_END = 1973          # only stations active at/after ISD-Lite start
WORKERS = 4


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%FT%T')} {msg}\n")


def load_stations():
    stations = []
    with open(os.path.join(ROOT, "data", "isd-history.csv"), newline="",
              encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                lat, lon = float(row["LAT"]), float(row["LON"])
                beg, end = int(row["BEGIN"][:4]), int(row["END"][:4])
            except (ValueError, KeyError):
                continue
            # skip bogus pseudo-stations and non-meteorological entries
            if not (-90 <= lat <= 90 and -180 <= lon <= 180 and abs(lat) > 0.01
                    and abs(lon) > 0.01):
                continue
            if end < MIN_END:
                continue
            sa, wb = row["USAF"].strip(), row["WBAN"].strip()
            # (some IDs are alphanumeric, e.g. ships A00002 -- pad, don't parse)
            sid = f"{sa.zfill(6)}-{wb.zfill(5)}"
            stations.append({
                "id": sid, "name": row["STATION NAME"], "lat": lat, "lon": lon,
                "begin": max(MIN_YEAR, beg), "end": min(MAX_YEAR, end)})
    return stations


def file_url(st, yr):
    return f"{BASE}/{yr}/{st['id']}-{yr}.gz"


def stash_path(st, yr):
    return os.path.join(OUT, str(yr), f"{st['id']}-{yr}.gz")


def fetch(st, yr):
    """Download one station-year .gz; returns size bytes or error marker."""
    dest = stash_path(st, yr)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return ("exists", st, yr)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = file_url(st, yr)
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "weather1b-data/0.3+isd-lite"})
            with urllib.request.urlopen(req, timeout=60) as r, \
                    open(dest, "wb") as fh:
                data = r.read()
                if len(data) < 20 and b"404" in data[:64]:
                    raise urllib.error.HTTPError(url, 404, "not found", {}, None)
                fh.write(data)
            return ("ok", st, yr)
        except Exception as e:
            last = e
            try:
                if os.path.exists(dest):
                    os.remove(dest)
            except OSError:
                pass
            wait = 1.5 * (attempt + 1) * (attempt + 1)  # 1.5, 6, 13.5 s
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                return ("404", st, yr)   # station inactive that year: not an error
            time.sleep(wait)
    return ("err", st, yr, str(last)[:80])


def main():
    stations = load_stations()
    # all (station, year) tasks in FIFO order, stable across resumes
    tasks = []
    for st in stations:
        for yr in range(st["begin"], st["end"] + 1):
            tasks.append((st, yr))
    state = {}
    if os.path.exists(STATE):
        with open(STATE) as f:
            state = json.load(f)
    done = set(state.get("done", []))
    total = len(tasks)
    log(f"[start] {len(stations)} stations, {total:,} station-years to harvest"
        f" (already done {len(done):,})")
    pending = [(st, yr) for st, yr in tasks if f"{st['id']}|{yr}" not in done]
    log(f"[start] {len(pending):,} remaining")
    if not pending:
        log("[done] nothing left to fetch")
        return
    stats_lock = threading.Lock()
    stats = {"n": 0, "bytes": state.get("bytes", 0), "err": 0, "skip": 0,
             "t0": time.time()}
    done_set = set(done)

    def report():
        el = max(time.time() - stats["t0"], 1e-6)
        rate = stats["n"] / el
        eta = (len(pos) - stats["n"]) / max(rate, 1e-6) / 3600
        log(f"[~] {stats['n']:,}/{len(pos):,} fetched · {rate:.2f} files/s · "
            f"{stats['bytes']/2**30:.1f} GiB · ETA {eta:.1f} h · "
            f"errs {stats['err']} skips {stats['skip']}")

    from concurrent.futures import as_completed
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        pos = list(pending)
        futs = {ex.submit(fetch, st, yr): (st, yr) for st, yr in pos}
        for i, fu in enumerate(as_completed(futs), 1):
            st, yr = futs[fu]
            try:
                res = fu.result()
            except Exception as e:
                res = ("err", st, yr, str(e)[:60])
            with stats_lock:
                if res[0] == "ok":
                    stats["n"] += 1
                    try:
                        stats["bytes"] += os.path.getsize(stash_path(st, yr))
                    except OSError:
                        pass
                    done_set.add(f"{st['id']}|{yr}")
                elif res[0] == "404":
                    stats["skip"] += 1
                    done_set.add(f"{st['id']}|{yr}")
                else:
                    stats["err"] += 1
                if i % 40 == 0:
                    fp = STATE + ".tmp"
                    with open(fp, "w") as f:
                        json.dump({"done": sorted(done_set),
                                   "bytes": stats["bytes"],
                                   "stations": len(stations)}, f)
                    os.replace(fp, STATE)
                    report()
    with open(STATE + ".tmp", "w") as f:
        json.dump({"done": sorted(done_set), "bytes": stats["bytes"],
                   "stations": len(stations)}, f)
    os.replace(STATE + ".tmp", STATE)
    report()
    log(f"[done] harvested {stats['n']:,} new files · {stats['bytes']/2**30:.1f} GiB total"
        f" · {stats['err']} errors · {stats['skip']} inactive-year skips")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        log(f"[FATAL] unhandled: {__import__('traceback').format_exc()}")
        sys.exit(1)