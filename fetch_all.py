"""
fetch_all.py
============
Bulk acquisition of the widest weather record we can pull:
  * every catalog station, hourly, 1940 -> present (Open-Meteo ERA5 reanalysis)
  * NOAA GSOD daily dailies 1929 -> 1979 (the pre-1940 history hole)

State is persisted in ``data/raw/.openmeteo_state.json`` so an interrupted
run can resume.  The output is one CSV per station:

    date,temperature_c,pressure_hpa,humidity_pct,wind_ms

Hourly CSVs land in ``data/raw/`` (zip: weather_hourly_archive.zip)
GSOD daily CSVs land in ``data/raw/gsod/`` (zip: weather_gsod_daily.zip)

Usage:
    python fetch_all.py                # everything
    python fetch_all.py --only hourly  # hourly only
    python fetch_all.py --only gsod    # 1929-1979 dailies only
"""

from __future__ import annotations

import argparse
import csv
import gzip  # noqa: F401  (kept for parity with future .op retries)
import json
import math
import os
import shutil
import tarfile
import time
import urllib.parse
import urllib.request
import urllib.error
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ---------------------------------------------------------------------------
# Station catalogue -- 72 stations spanning every climate regime
# ---------------------------------------------------------------------------
STATIONS: dict[str, tuple[float, float]] = {
    # ---- Americas ----
    "New York Central Park": (40.7794, -73.9692),
    "Chicago OHare": (41.9818, -87.9066),
    "Los Angeles": (34.0522, -118.2437),
    "San Francisco": (37.7749, -122.4194),
    "Seattle": (47.6062, -122.3321),
    "Denver": (39.7392, -104.9903),
    "Miami": (25.7617, -80.1918),
    "Washington DC": (38.9072, -77.0369),
    "Houston": (29.7604, -95.3698),
    "Anchorage": (61.2181, -149.9003),
    "Fairbanks": (64.8378, -147.7164),
    "Vancouver": (49.2827, -123.1207),
    "Toronto": (43.6532, -79.3832),
    "Montreal": (45.5017, -73.5673),
    "Edmonton": (53.5461, -113.4938),
    "Winnipeg": (49.8951, -97.1384),
    "Mexico City": (19.4326, -99.1332),
    "Havana": (23.1136, -82.3666),
    "Guatemala City": (14.6349, -90.5069),
    "Panama City": (8.9824, -79.5199),
    "San Juan PR": (18.4655, -66.1057),
    "Caracas": (10.4806, -66.9036),
    "Bogota": (4.7110, -74.0721),
    "Quito": (-0.1807, -78.4678),
    "Lima": (-12.0464, -77.0428),
    "La Paz": (-16.4897, -68.1193),
    "Sao Paulo": (-23.5505, -46.6333),
    "Rio de Janeiro": (-22.9068, -43.1729),
    "Buenos Aires": (-34.6037, -58.3816),
    "Montevideo": (-34.9011, -56.1645),
    "Santiago": (-33.4489, -70.6693),
    "Brasilia": (-15.7939, -47.8828),
    "Ushuaia": (-54.8019, -68.3030),
    "Punta Arenas": (-53.1638, -70.9171),
    # ---- Europe ----
    "London Heathrow": (51.4700, -0.4543),
    "Paris Orly": (48.7233, 2.3794),
    "Berlin Tempelhof": (52.4733, 13.4044),
    "Madrid Barajas": (40.4719, -3.5626),
    "Rome Fiumicino": (41.8003, 12.2389),
    "Lisbon": (38.7223, -9.1393),
    "Amsterdam": (52.3676, 4.9041),
    "Brussels": (50.8503, 4.3517),
    "Vienna": (48.2082, 16.3738),
    "Warsaw": (52.2297, 21.0122),
    "Prague": (50.0755, 14.4378),
    "Budapest": (47.4979, 19.0402),
    "Athens": (37.9838, 23.7275),
    "Istanbul": (41.0082, 28.9784),
    "Oslo": (59.9139, 10.7522),
    "Stockholm": (59.3293, 18.0686),
    "Helsinki": (60.1699, 24.9384),
    "Copenhagen": (55.6761, 12.5683),
    "Dublin": (53.3498, -6.2603),
    "Edinburgh": (55.9533, -3.1883),
    "Zurich": (47.3769, 8.5417),
    "Kyiv": (50.4501, 30.5234),
    "Bucharest": (44.4268, 26.1025),
    "Belgrade": (44.7866, 20.4489),
    "Reykjavik": (64.1466, -21.9426),
    "Moscow VDNKh": (55.8233, 37.6347),
    "Saint Petersburg": (59.9311, 30.3609),
    "Murmansk": (68.9585, 33.0827),
    # ---- Africa & Middle East ----
    "Cairo": (30.0444, 31.2357),
    "Cape Town": (-33.9249, 18.4241),
    "Johannesburg": (-26.2041, 28.0473),
    "Nairobi": (-1.2921, 36.8219),
    "Lagos": (6.5244, 3.3792),
    "Accra": (5.6037, -0.1870),
    "Algiers": (36.7538, 3.0588),
    "Casablanca": (33.5731, -7.5898),
    "Addis Ababa": (9.0320, 38.7469),
    "Dakar": (14.7167, -17.4677),
    "Khartoum": (15.5007, 32.5599),
    "Tunis": (36.8065, 10.1815),
    "Tehran": (35.6892, 51.3890),
    "Riyadh": (24.7136, 46.6753),
    "Tel Aviv": (32.0853, 34.7818),
    # ---- Asia ----
    "Beijing Capital": (40.0801, 116.5846),
    "Shanghai": (31.2304, 121.4737),
    "Hong Kong": (22.3193, 114.1694),
    "Chengdu": (30.5728, 104.0668),
    "Urumqi": (43.8256, 87.6168),
    "Tokyo": (35.6762, 139.6503),
    "Seoul": (37.5665, 126.9780),
    "Delhi Safdarjung": (28.5892, 77.2113),
    "Mumbai": (19.0760, 72.8777),
    "Karachi": (24.8607, 67.0011),
    "Dhaka": (23.8103, 90.4125),
    "Bangkok": (13.7563, 100.5018),
    "Singapore Changi": (1.3644, 103.9915),
    "Jakarta": (-6.2088, 106.8456),
    "Manila": (14.5995, 120.9842),
    "Ho Chi Minh City": (10.8231, 106.6297),
    "Taipei": (25.0330, 121.5654),
    "Kuala Lumpur": (3.1390, 101.6869),
    "Colombo": (6.9271, 79.8612),
    "Yangon": (16.8409, 96.1735),
    "Irkutsk": (52.2870, 104.2810),
    "Novosibirsk": (55.0302, 82.9204),
    "Vladivostok": (43.1198, 131.8869),
    "Astana": (51.1605, 71.4704),
    "Tashkent": (41.2995, 69.2401),
    "Kathmandu": (27.7172, 85.3240),
    # ---- Oceania & Pacific ----
    "Sydney Observatory Hill": (-33.8599, 151.2053),
    "Melbourne": (-37.8136, 144.9631),
    "Brisbane": (-27.4698, 153.0251),
    "Perth": (-31.9505, 115.8605),
    "Adelaide": (-34.9285, 138.6007),
    "Hobart": (-42.8821, 147.3272),
    "Darwin": (-12.4634, 130.8456),
    "Auckland": (-36.8509, 174.7645),
    "Wellington": (-41.2866, 174.7756),
    "Honolulu": (21.3069, -157.8583),
    "Suva": (-18.1248, 178.4501),
    # ---- Polar ----
    "Longyearbyen": (78.2232, 15.6267),
    "Alert Nunavut": (82.5012, -62.3464),
}
# dedupe by lat/lon
_seen, _uniq = set(), {}
for k, v in STATIONS.items():
    if v not in _seen:
        _uniq[k] = v
        _seen.add(v)
STATIONS = _uniq
print(f"[catalog] {len(STATIONS)} unique stations")

# ---------------------------------------------------------------------------
# Open-Meteo hourly backend
# ---------------------------------------------------------------------------
HOURLY_VARS = "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m"
SCHEMA = ("date", "temperature_c", "pressure_hpa", "humidity_pct", "wind_ms")


def _http_get(url: str, retries: int = 120, timeout: int = 180,
              politeness: float = 1.5) -> bytes:
    """GET with patient handling of the archive API's rate limiter.

    Retries effectively indefinitely on 429/5xx with a capped backoff, so a
    long cooldown just stalls the background job instead of failing chunks.
    """
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "weather1b-bulk/0.3"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
                time.sleep(politeness)
                return data
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429:
                delay = min(300, 20 + 10 * i)
                if i in (0, 2, 5, 10, 20) or i % 20 == 19:
                    print(f"    429: backing off {delay}s", flush=True)
                time.sleep(delay)
            elif e.code in (500, 502, 503, 504):
                time.sleep(min(120, 10 * (i + 1)))
            else:
                raise              # hard error (404/400/403): let it bubble
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(60.0, 2.0 * (2 ** i)))
    raise RuntimeError(f"download failed after {retries} tries: {url}: {last}")


def fetch_station_chunk(station: str, lat: float, lon: float,
                        start: str, end: str, out_csv: str) -> int:
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": HOURLY_VARS, "wind_speed_unit": "ms", "timezone": "GMT",
    })
    url = f"https://archive-api.open-meteo.com/v1/archive?{q}"
    j = json.loads(_http_get(url))
    h = j.get("hourly") or {}
    times, rows = h.get("time") or [], 0
    with open(out_csv, "a", newline="") as fh:
        w = csv.writer(fh)
        for i, tm in enumerate(times):
            rec = [tm]
            for key in ("temperature_2m", "pressure_msl",
                        "relative_humidity_2m", "wind_speed_10m"):
                try:
                    v = (h.get(key) or [None] * len(times))[i]
                    rec.append("" if v is None else f"{v:f}")
                except Exception:  # noqa: BLE001
                    rec.append("")
            w.writerow(rec)
            rows += 1
    return rows


def fetch_all_hourly(out_dir: str, start_year: int = 1940, end_year: int = 2026,
                     chunk_years: int = 10, workers: int = 4) -> None:
    os.makedirs(out_dir, exist_ok=True)
    state_path = os.path.join(out_dir, ".openmeteo_state.json")
    state: dict = {}
    if os.path.exists(state_path):
        with open(state_path) as fh:
            state = json.load(fh)
    lock = Lock()

    def chunks():
        from datetime import date
        for y in range(start_year, end_year + 1, chunk_years):
            s = date(y, 1, 1)
            e = date(min(y + chunk_years - 1, end_year), 12, 31)
            if s.year > e.year:
                break
            yield f"{s.isoformat()}|{e.isoformat()}"

    def one_station(station: str, lat: float, lon: float, spans: list[str]) -> tuple:
        """Fetch one station's chunks *sequentially* (chronological rows).

        Different stations run on different worker threads; a single station's
        rows are never interleaved, so each station CSV stays time-ordered.
        """
        freekey = f"{station}::DONE_ALL"
        if state.get(freekey):
            return station, -1, 0
        out_csv = os.path.join(out_dir, f"{station.replace(' ', '_')}.csv")
        with open(out_csv, "w", newline="") as fh:
            csv.writer(fh).writerow(SCHEMA)
        total_rows, failed = 0, 0
        for span in spans:
            key = f"{station}::{span}"
            with lock:
                cached = state.get(key)
            if cached:
                total_rows += cached if isinstance(cached, int) else 87600
                continue
            start, end = span.split("|")
            for attempt in range(6):
                try:
                    rows = fetch_station_chunk(station, lat, lon, start, end, out_csv)
                    with lock:
                        state[key] = rows
                        with open(state_path, "w") as fh:
                            json.dump(state, fh)
                    total_rows += rows
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"    retry {station} {span}: {type(e).__name__} {e}",
                          flush=True)
                    time.sleep(2 * (attempt + 1))
            else:
                failed += 1
        with lock:
            state[freekey] = total_rows
            with open(state_path, "w") as fh:
                json.dump(state, fh)
        return station, total_rows, failed

    # group every span of a station together so its chunks run sequentially
    jobs = [(s, lat, lon, list(chunks())) for s, (lat, lon) in STATIONS.items()]
    t0 = time.time()
    done = 0
    for station, rows, failed in one_parallel(one_station, jobs, workers):
        done += 1
        print(f"[{done}/{len(jobs)}] {station}: {rows:,} rows (fail={failed}) "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"[hourly] finished {len(jobs)} stations in {(time.time()-t0)/60:.1f} min")


def one_parallel(fn, jobs, workers: int):
    if workers <= 1:
        for j in jobs:
            yield fn(*j)
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, *j): j for j in jobs}
        for f in as_completed(futs):
            yield f.result()


# ---------------------------------------------------------------------------
# NOAA GSOD daily backend (1929-1979 tarballs)
# ---------------------------------------------------------------------------
def fetch_gsod_tarballs(start_year: int, end_year: int, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, "_tarballs")
    os.makedirs(cache, exist_ok=True)
    for year in range(start_year, end_year + 1):
        tgz = os.path.join(cache, f"{year}.tar.gz")
        if not os.path.exists(tgz) or os.path.getsize(tgz) < 1000:
            url = f"https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive/{year}.tar.gz"
            print(f"[gsod] downloading {year}...", flush=True)
            data = _http_get(url, retries=3)
            with open(tgz, "wb") as fh:
                fh.write(data)
        else:
            print(f"[gsod] {year}.tar.gz cached", flush=True)
        convert_tarball(tgz, out_dir, year)


def _rh_gsod(temp_c: float, dewp_c: float) -> float:
    es = 6.1094 * math.exp(17.625 * temp_c / (243.04 + temp_c))
    ea = 6.1094 * math.exp(17.625 * dewp_c / (243.04 + dewp_c))
    return min(100.0, max(0.0, 100.0 * ea / es)) if es > 0 else 100.0


def convert_tarball(tgz: str, out_dir: str, year: int) -> int:
    stations_dir = os.path.join(out_dir, "gsod")
    os.makedirs(stations_dir, exist_ok=True)
    td = os.path.join(out_dir, "_tmp", str(year))
    os.makedirs(td, exist_ok=True)
    rows = 0
    with tarfile.open(tgz, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".csv"):
                continue
            station = os.path.basename(member.name).replace(".csv", "")
            src = tar.extractfile(member)
            if src is None:
                continue
            lines = io_text_lines(src)
            header = next(lines, None)
            if header is None:
                continue
            out_csv = os.path.join(stations_dir, f"{station}.csv")
            new = not os.path.exists(out_csv)
            with open(out_csv, "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(SCHEMA)
                for line in lines:
                    r = csv.reader([line]).__next__()
                    if len(r) < 28:
                        continue
                    try:
                        temp_f = float(r[6])
                        if temp_f > 900:
                            continue
                        dewp_f = float(r[8]) if r[8].strip() else float("nan")
                        slp = float(r[10]) if r[10].strip() else float("nan")
                        wdsp_k = float(r[16]) if r[16].strip() else float("nan")
                    except (ValueError, IndexError):
                        continue
                    t_c = (temp_f - 32.0) * 5.0 / 9.0
                    d_c = (dewp_f - 32.0) * 5.0 / 9.0 if dewp_f <= 900 else t_c
                    rh = _rh_gsod(t_c, d_c)
                    press = slp if slp < 9900 else float("nan")
                    wind = wdsp_k * 0.514444 if wdsp_k <= 900 else float("nan")
                    w.writerow([f"{r[1]}T00:00", f"{t_c:.2f}", f"{press:.1f}",
                                f"{rh:.1f}", f"{wind:.2f}"])
                    rows += 1
    shutil.rmtree(td, ignore_errors=True)
    print(f"[gsod] {year}: {rows:,} daily rows", flush=True)
    return rows


def io_text_lines(fobj):
    import io
    for raw in fobj:
        yield raw.decode("utf-8", errors="replace").rstrip("\r\n")


# ---------------------------------------------------------------------------
def make_archives(raw: str) -> None:
    hourly_csvs = [os.path.join(raw, n) for n in os.listdir(raw)
                   if n.endswith(".csv")]
    hourly_zip = os.path.join(raw, "weather_hourly_archive.zip")
    with zipfile.ZipFile(hourly_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for csv_path in hourly_csvs:
            z.write(csv_path, arcname=os.path.basename(csv_path))
    print(f"[archive] {hourly_zip} ({sum(os.path.getsize(p) for p in hourly_csvs)//1048576} MB raw)")

    gsod_dir = os.path.join(raw, "gsod")
    gsod_csvs = [os.path.join(gsod_dir, n) for n in os.listdir(gsod_dir)
                 if n.endswith(".csv")]
    if gsod_csvs:
        gsod_zip = os.path.join(raw, "weather_gsod_daily.zip")
        with zipfile.ZipFile(gsod_zip, "w", zipfile.ZIP_DEFLATED) as z:
            for csv_path in gsod_csvs:
                z.write(csv_path, arcname=os.path.basename(csv_path))
        print(f"[archive] {gsod_zip} ({sum(os.path.getsize(p) for p in gsod_csvs)//1048576} MB raw)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["hourly", "gsod"], default=None)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--gsod-years", nargs=2, type=int, default=[1929, 1979])
    ap.add_argument("--no-archive", action="store_true")
    args = ap.parse_args()

    if args.only in (None, "hourly"):
        fetch_all_hourly(args.out, workers=args.workers)
    if args.only in (None, "gsod"):
        fetch_gsod_tarballs(args.gsod_years[0], args.gsod_years[1], args.out)
    if not args.no_archive:
        make_archives(args.out)


if __name__ == "__main__":
    main()