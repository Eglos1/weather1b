#!/usr/bin/env python3
"""Live WeatherNet forecast for a point location.

Fetches the most recent 24 days of daily means (T, P, RH, wind) from
Open-Meteo's archive API, feeds them through the trained WeatherNet,
and prints the next 24 daily-temperature forecasts (we read off any
lead day, e.g. day 7).

Usage:
  python forecast_live.py --location "33.3854,-91.0519" [--days 7]
                          [--ckpt runs/weather-cpu/latest.pt]
"""
import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np
import torch

from model import WeatherNet, WeatherNetConfig, WINDOW_HOURS, NUM_VARS
from data import Normalizer

# Greenville, MS (Mid Delta Regional Airport), same point NWS uses
DEFAULT_LOC = "33.3854,-91.0519"
DEFAULT_CKPT = "runs/weather-cpu/latest.pt"


def fetch_daily(lat: float, lon: float, days: int = WINDOW_HOURS):
    """Return list of (date, [t, p, rh, wind]) oldest->newest, one per day."""
    import datetime as _dt
    end = _dt.date.today() - _dt.timedelta(days=1)   # yesterday (complete day)
    start = end - _dt.timedelta(days=days - 1)
    q = ("https://archive-api.open-meteo.com/v1/archive?latitude={}&longitude={}"
         "&start_date={}&end_date={}"
         "&daily=temperature_2m_mean,pressure_msl_mean,relative_humidity_2m_mean,"
         "wind_speed_10m_mean&timezone=America%2FChicago").format(
             lat, lon, start.isoformat(), end.isoformat())
    with urllib.request.urlopen(q, timeout=45) as r:
        d = json.load(r)
    daily = d["daily"]
    rows = []
    for i, date in enumerate(daily["time"]):
        t = daily["temperature_2m_mean"][i]
        p = daily["pressure_msl_mean"][i]
        rh = daily["relative_humidity_2m_mean"][i]
        w = daily["wind_speed_10m_mean"][i]
        if t is None or p is None or rh is None or w is None:
            last = rows[-1] if rows else None
            if last is None:
                sys.exit(f"missing data on {date}, cannot build window")
            rows.append((date, list(last[1])))       # carry-forward fill
        else:
            rows.append((date, [float(t), float(p), float(rh), float(w)]))
    if len(rows) < WINDOW_HOURS:
        sys.exit(f"got {len(rows)} days, need {WINDOW_HOURS}")
    rows = rows[-WINDOW_HOURS:]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", default=DEFAULT_LOC,
                    help="lat,lon of the forecast point")
    ap.add_argument("--days", type=int, default=7,
                    help="how many forecast days out (<=24)")
    ap.add_argument("--lead", type=int, default=1,
                    help="which output day to lead with (1 = day after window end)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--max-window", type=int, default=0,
                    help="only use this many trailing days (debug)")
    args = ap.parse_args()
    if args.days > 24 or args.lead + args.days - 1 > 24:
        sys.exit("days/lead must fit within the 24-day output")
    lat, lon = [float(v) for v in args.location.split(",")]
    if not os.path.exists(args.ckpt):
        sys.exit(f"checkpoint not found: {args.ckpt}")

    t0 = time.time()
    rows = fetch_daily(lat, lon)
    if args.max_window:
        rows = rows[-args.max_window:]
    x = np.asarray([r[1] for r in rows], dtype=np.float32)          # (24,4)
    x = x[-WINDOW_HOURS:]                                            # newest 24
    dates = [r[0] for r in rows][-WINDOW_HOURS:]
    ctx = x.mean(axis=0).astype(np.float32).reshape(1, NUM_VARS)     # (1,4)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = WeatherNetConfig(**ck["config"])
    net = WeatherNet(cfg).eval()
    net.load_state_dict(ck["model_state"])
    nm = Normalizer(); nm.load_state_dict(ck["normalizer"])
    print(f"[ckpt] step={ck.get('step')} n_params={ck.get('n_params', 0):,} "
          f"loaded in {time.time()-t0:.0f}s", flush=True)

    xt = torch.tensor(nm.transform_x(x.reshape(-1, WINDOW_HOURS, NUM_VARS)))
    ct = torch.tensor(nm.transform_ctx(ctx))
    with torch.no_grad():
        y = net(xt, ct)                                              # (1,24)
    pred = nm.inverse_y(y).squeeze(0).numpy()                        # deg C

    # window = dates[-1] is the newest observed day; output day k -> next day
    import datetime as _dt
    base = _dt.date.fromisoformat(dates[-1])
    print("\n   date   | mean T model |  mean T NWS*(hi/lo)  |  notes")
    print("----------+--------------+------------------------+----------------")
    for k in range(args.lead - 1, args.lead - 1 + args.days):
        day = base + _dt.timedelta(days=1 + k)
        c = float(pred[k])
        f = c * 9 / 5 + 32
        tag = ""
        if k == 0:
            tag = "<- day+1"
        elif k == len(pred) - 1:
            tag = "<- day+24"
        print(f" {day.isoformat()} |    {c:5.1f} C ({f:.0f} F) |            | {tag}")
    print("\n* NWS hi/lo are point forecasts; model predicts daily MEAN temp only.")


if __name__ == "__main__":
    main()