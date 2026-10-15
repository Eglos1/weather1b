"""
download_data.py
================
Acquire century-scale weather observations and reanalysis for training the
1.58-bit ternary network on *real* data.

Three operational backends are provided (all free / open):

  1. **Open-Meteo Historical API**  (``fetch_open_meteo``)
     Global ERA5 / ERA5-Land reanalysis.  **Hourly, gap-free, 1940 -> today**.
     No API key.  Covers every grid point on Earth; a single request returns
     a full continuous hourly series for one coordinate pair.
        URL  : https://archive-api.open-meteo.com/v1/archive
        Source: ECMWF ERA5 / ERA5-Land (Copernicus)

  2. **NOAA GSOD**  (``fetch_gsod``)
     Global Summary of the Day, 9000+ stations, **1929 -> 2025** (daily).
     Available per-station as ``access/{year}/{station}.csv`` on NCEI.
        URL  : https://www.ncei.noaa.gov/data/global-summary-of-the-day/

  3. **ERA-20C / pre-1940 reanalysis**  (``fetch_era20c``)
     20th-Century Reanalysis (1900-2010, 3-hourly), plus ERA5 *back-extension*
     1950-1978 -- useful to push dense hourly-ish data before 1940.
     Requires a free Copernicus CDS account + ``cdsapi`` package.

The station-archive layout produced here is exactly what ``WeatherZipCSV``
in ``data.py`` consumes:
    station.csv  ->  date,temperature_c,pressure_hpa,humidity_pct,wind_ms
so ``python train.py --data <archive.zip>`` trains on the real records.

Usage
-----
    # default: 18 long-record stations, 1940-2025 hourly, 5-year chunks:
    python download_data.py --years 1940 2025

    # a quick probe (2 stations, 2 years) for CI:
    python download_data.py --stations "New York Central Park" London --years 1940 1941

    # NOAA daily GSOD for stations of interest (daily-resolution records):
    python download_data.py --source gsod --years 1929 1940 --stations "New York"

Raw CSVs land in ``data/raw/`` and are finally zipped into
``data/raw/weather_archive.zip``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import time
import urllib.parse
import urllib.request
import zipfile
from typing import Callable

# ---------------------------------------------------------------------------
# Station catalogue (coordinates of long, well-observed records)
# ---------------------------------------------------------------------------
STATIONS: dict[str, tuple[float, float]] = {
    "New York Central Park":    (40.7794, -73.9692),
    "London Heathrow":          (51.4700,  -0.4543),
    "Berlin Tempelhof":         (52.4733,  13.4044),
    "Paris Orly":               (48.7233,   2.3794),
    "Moscow VDNKh":             (55.8233,  37.6347),
    "Beijing Capital":          (40.0801, 116.5846),
    "Tokyo":                    (35.6762, 139.6503),
    "Sydney Observatory Hill":  (-33.8599, 151.2053),
    "Buenos Aires":             (-34.6037, -58.3816),
    "Nairobi":                  (-1.2921,  36.8219),
    "Cairo":                    (30.0444,  31.2357),
    "Delhi Safdarjung":         (28.5892,  77.2113),
    "Rio de Janeiro":           (-22.9068, -43.1729),
    "Mexico City":              (19.4326, -99.1332),
    "Cape Town":                (-33.9249,  18.4241),
    "Anchorage":                (61.2181, -149.9003),
    "Reykjavik":                (64.1466, -21.9426),
    "Santiago":                 (-33.4489, -70.6693),
}

GSOD_STATIONS = {
    # 11-digit ISD station ids (USAF+WBAN).  A few long records:
    "New York Central Park": "725033-94728",
    "London Heathrow":       "037720-99999",
    "Berlin Tempelhof":      "103840-99999",
    "Daily Station":         "0",
}


def _http_get(url: str, retries: int = 4, timeout: int = 120) -> bytes:
    """Very small GET helper with exponential backoff and a polite UA."""
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "weather1b-downloader/0.1"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(30.0, 1.5 * (2 ** i)))
    raise RuntimeError(f"download failed: {url}: {last}")


# ---------------------------------------------------------------------------
# Scheme 1: Open-Meteo (hourly, 1940->present, no key)
# ---------------------------------------------------------------------------
HOURLY_VARS = "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m"
SCHEMA = ("date", "temperature_c", "pressure_hpa", "humidity_pct", "wind_ms")


def fetch_open_meteo(
    lat: float, lon: float, start: str, end: str, out_csv: str, chunk_years: int = 5
) -> int:
    """Download hourly series for one coordinate and append to ``out_csv``.

    Requests are split into ``chunk_years``-long windows so a single JSON
    payload stays modest.  Returns the number of rows written.
    """
    from datetime import date

    def split(start: str, end: str, step: int):
        s = date.fromisoformat(start)
        e = date.fromisoformat(end)
        while s <= e:
            nxt = min(e, date(s.year + step - 1, 12, 31))
            yield s.isoformat(), nxt.isoformat()
            s = date(nxt.year + 1, 1, 1)

    rows = 0
    first = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as fh:
        w = csv.writer(fh)
        if first:
            w.writerow(SCHEMA)
        for s, e in split(start, end, chunk_years):
            q = urllib.parse.urlencode(
                {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": s,
                    "end_date": e,
                    "hourly": HOURLY_VARS,
                    "wind_speed_unit": "ms",
                    "timezone": "GMT",
                }
            )
            url = f"https://archive-api.open-meteo.com/v1/archive?{q}"
            j = json.loads(_http_get(url))
            h = j.get("hourly") or {}
            times = h.get("time") or []
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
            print(f"    {s}..{e}: {len(times)} hours")
            time.sleep(0.4)  # politeness
    return rows


# ---------------------------------------------------------------------------
# Scheme 2: NOAA GSOD (daily, 1929->present)
# ---------------------------------------------------------------------------
def _gsod_rh(temp_c: float, dewp_c: float) -> float:
    """Relative humidity (%) from T/Td using the August-Roche-Magnus formula."""
    import math
    es = 6.1094 * math.exp(17.625 * temp_c / (243.04 + temp_c))
    ea = 6.1094 * math.exp(17.625 * dewp_c / (243.04 + dewp_c))
    return min(100.0, max(0.0, 100.0 * ea / es)) if es > 0 else 100.0


def fetch_gsod(station: str, year: int, out_csv: str) -> int:
    """Append one GSOD daily station-year to ``out_csv`` (physical units).

    Data returned by NOAA NCEI is in tenths of Fahrenheit / knots / mb;
    converted to C / hPa / % / m/s.
    """
    url = f"https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/{station}.csv"
    raw = _http_get(url).decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(raw))
    first = not os.path.exists(out_csv)
    rows = 0
    with open(out_csv, "a", newline="") as fh:
        w = csv.writer(fh)
        if first:
            w.writerow(SCHEMA)
        for r in reader:
            def f(key: str):
                try:
                    return float(r[key])
                except (KeyError, ValueError):
                    return float("nan")
            temp_f = f("TEMP")
            dewp_f = f("DEWP")
            slp = f("SLP")
            wdsp_k = f("WDSP")
            if temp_f > 900:      # 9999.9 = missing flag
                continue
            # access CSV values are decimal degrees F (e.g. 70.2 = 70.2 F)
            t_c = (temp_f - 32.0) * 5.0 / 9.0
            d_c = (dewp_f - 32.0) * 5.0 / 9.0 if dewp_f <= 900 else t_c
            rh = _gsod_rh(t_c, d_c)
            press = slp if slp < 9900 else float("nan")   # 9999.9 = missing
            wind = wdsp_k * 0.514444 if wdsp_k <= 900 else float("nan")
            w.writerow([r["DATE"], f"{t_c:.2f}", f"{press:.1f}",
                        f"{rh:.1f}", f"{wind:.2f}"])
            rows += 1
    return rows


# ---------------------------------------------------------------------------
# Scheme 3: ERA-20C pre-1940 reanalysis (needs free CDS account)
# ---------------------------------------------------------------------------
def fetch_era20c(lat: float, lon: float, start: int, end: int, out_csv: str) -> int:
    """1900-2010 3-hourly reanalysis through the Copernicus CDS API.

    Requires ``pip install cdsapi`` and a ``~/.cdsapirc`` key from
    https://cds.climate.copernicus.eu  (free registration).
    """
    try:
        import cdsapi
    except ImportError:
        raise SystemExit(
            "[era20c] install the Copernicus client first:  pip install cdsapi\n"
            "set up ~/.cdsapirc with your free CDS key (cds.climate.copernicus.eu)"
        )
    client = cdsapi.Client()
    import numpy as np
    rows = 0
    first = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as fh:
        w = csv.writer(fh)
        if first:
            w.writerow(SCHEMA)
        for year in range(start, end + 1):
            # single gridpoint extraction, 3-hourly (the finest ERA-20C step)
            out = client.retrieve(
                "reanalysis-era20c-single-levels",
                {
                    "product_type": "ensemble_mean",
                    "variable": ["2m_temperature", "mean_sea_level_pressure",
                                 "10m_wind_speed"],
                    "year": str(year),
                    "month": [f"{m:02d}" for m in range(1, 13)],
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "time": ["00:00", "03:00", "06:00", "09:00",
                             "12:00", "15:00", "18:00", "21:00"],
                    "area": [lat + 0.25, lon - 0.25, lat - 0.25, lon + 0.25],
                    "data_format": "csv",
                },
            )
            tgt = f"/tmp/era20c_{lat}_{lon}_{year}.csv"
            out.download(tgt)
            import pandas as pd
            df = pd.read_csv(tgt)
            t2 = df["t2m"].values - 273.15
            wkp = (df["pz"].values * 0.1)          # Pa -> hPa
            msr = df["si10"].values
            for i, _ in enumerate(range(len(df))):
                w.writerow([str(df["valid_time"].iloc[i]), f"{t2[i]:.2f}",
                            f"{wkp[i]:.1f}", '""', f"{msr[i]:.2f}"])
                rows += 1
            print(f"    {year}: {rows_ := rows} rows so far")
    return rows


# ---------------------------------------------------------------------------
# Zipping & drivers
# ---------------------------------------------------------------------------
def zip_out_dir(raw_dir: str, zip_path: str) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(os.listdir(raw_dir)):
            if name.endswith(".csv"):
                z.write(os.path.join(raw_dir, name), arcname=name)
    print(f"[+archive] {zip_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="century-scale weather acquisition")
    ap.add_argument("--source", choices=["openmeteo", "gsod", "era20c"],
                    default="openmeteo")
    ap.add_argument("--stations", nargs="*", default=None,
                    help="station names; default = all in catalogue")
    ap.add_argument("--years", nargs=2, type=int, default=[1940, 2025])
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.source == "openmeteo":
        for name, (lat, lon) in STATIONS.items():
            if args.stations and name not in args.stations:
                continue
            out_csv = os.path.join(args.out, f"{name.replace(' ', '_')}.csv")
            print(f"[openmeteo] {name} ({lat}, {lon}) "
                  f"{args.years[0]}-{args.years[1]}")
            fetch_open_meteo(lat, lon, f"{args.years[0]}-01-01",
                             f"{args.years[1]}-12-31", out_csv)

    elif args.source == "gsod":
        for name, station in GSOD_STATIONS.items():
            if args.stations and name not in args.stations:
                continue
            if station == "0":
                continue
            out_csv = os.path.join(args.out, f"{name.replace(' ', '_')}.csv")
            for year in range(args.years[0], args.years[1] + 1):
                try:
                    n = fetch_gsod(station, year, out_csv)
                    print(f"[gsod] {name} {year}: {n} days")
                except Exception as e:  # noqa: BLE001
                    print(f"[gsod] {name} {year}: unavailable ({e})")

    elif args.source == "era20c":
        for name, (lat, lon) in ([
            ("New York Central Park", (40.7794, -73.9692)),
            ("London Heathrow", (51.4700, -0.4543)),
        ] if not args.stations else [(s, STATIONS[s]) for s in args.stations]):
            out_csv = os.path.join(args.out, f"{name.replace(' ', '_')}.csv")
            fetch_era20c(lat, lon, args.years[0], args.years[1], out_csv)

    if not args.no_zip:
        zip_out_dir(args.out, os.path.join(args.out, "weather_archive.zip"))


if __name__ == "__main__":
    main()