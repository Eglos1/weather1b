#!/usr/bin/env python3
"""Render gradient forecast maps from the FULL ISD-Lite HOURLY archive.

For every CONUS station in the hourly harvest:
  - read every hourly .gz file for the station's most recent years in full,
  - aggregate all hourly observations -> daily means (T, P, RH, wind),
  - take the last GAP-FREE run of >=24 consecutive days as the input window,
  - predict the next 24 daily temperatures with WeatherNet,
then interpolate the field over CONUS and paint classic gradient PNGs.

The previous version used the GSOD daily zip (which stops ~Aug 4 2025).
This one uses the hourly archive (reaches ~Aug 28 2025) and honors the
full per-year hourly record instead of a small tail of pre-aggregated data.

CPU inference, chunked, runs intentionally NICE ('nice -n 15') so it never
fights the user's foreground apps.
"""
import gzip, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "4")

from model import WeatherNet, WeatherNetConfig, WINDOW_HOURS, NUM_VARS, NUM_CONTEXT
from data import Normalizer
# NOTE: geo loading is inline below so we can do WORLD or CONUS.

ROOT = os.path.dirname(os.path.abspath(__file__))
HOUR_DIR = "/mnt/biggiecheese/weather1b/hourly/isd-lite"
CKPT = os.path.join(ROOT, "runs", "weather-gpu", "final.pt")
OUTDIR = os.path.join(ROOT, "maps")
os.makedirs(OUTDIR, exist_ok=True)

# Geo source + country borders for the chosen region
HIST = os.path.join(ROOT, "data", "isd-history.csv")
STATES = os.path.join(ROOT, "data", "us_states_20m.json")
WORLD = os.path.join(ROOT, "data", "world_countries_110m.json")

MODE = os.environ.get("MAP_MODE", "WORLD")   # "WORLD" renders the whole planet
if MODE == "WORLD":
    LAT0, LAT1, LON0, LON1 = -60.0, 84.0, -180.0, 180.0
elif MODE == "CONUS":
    LAT0, LAT1, LON0, LON1 = 22.0, 50.0, -128.0, -64.0
else:
    raise SystemExit(f"unknown MAP_MODE {MODE!r}")

# coherent forecast window: keep stations whose window end is within N days
# of the newest window end (so the map shows ONE date range, not 80 years)
RECENT_DAYS = 90
MIN_OBS_PER_DAY = 4          # at least this many hourly obs to trust a day
MAX_YEARS_BACK = 3           # how many recent yearly files to read per station


def load_geo():
    """isd-history.csv -> station_id -> (lat, lon); all stations in MODE bounds."""
    import csv
    geo = {}
    with open(HIST, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                sa, wb, la, lo = row["USAF"].strip(), row["WBAN"].strip(), \
                                 float(row["LAT"]), float(row["LON"])
            except (KeyError, ValueError):
                continue
            if not (-90 <= la <= 90 and -180 <= lo <= 180 and abs(la) > 0.01 and abs(lo) > 0.01):
                continue
            if not (LAT0 <= la <= LAT1 and LON0 <= lo <= LON1):
                continue
            sid = f"{sa.zfill(6)}-{wb.zfill(5)}"   # matches hourly .gz naming
            geo[sid] = (la, lo)
    print(f"[geo] {len(geo)} stations in {MODE} bounds", flush=True)
    return geo


def rh_from_dew(tc, tdc):
    """Relative humidity (%) from air temp and dew point (both °C)."""
    a = (17.625 * tdc) / (243.04 + tdc)
    b = (17.625 * tc) / (243.04 + tc)
    return 100.0 * (math.exp(a) / math.exp(b))


def daily_rows(path):
    """Yield daily means (date_string, [t, p, rh, wind]) from one hourly .gz.

    ISD-Lite fixed/space cols: Y M D H T(deciC) TD(deciC) SLP(deciPa)
    WDIR(deci?) WSPD(deci m/s) SKY P1 P6   (-9999 = missing).
    """
    sums = {}  # date -> [sum_t, sum_td, sum_p, sum_w, count]
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            p = line.split()
            if len(p) < 12:
                continue
            try:
                y, m, d = int(p[0]), int(p[1]), int(p[2])
                t = float(p[4]); td = float(p[5]); slp = float(p[6]); ws = float(p[8])
            except ValueError:
                continue
            if t <= -9000 or td <= -9000 or slp <= -9000 or ws <= -9000:
                continue
            date = (y, m, d)
            s = sums.get(date)
            if s is None:
                sums[date] = [t, td, slp, ws, 1]
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


def last_gapfree_window(daily):
    """Given list (date,[t,p,rh,w]) sorted, find the newest run of >=24
    consecutive calendar days. Returns (base_date, list-of-24-vals) or None."""
    if not daily:
        return None
    daily.sort(key=lambda r: r[0])
    by_day = {d: v for d, v in daily}
    from datetime import date as _date, timedelta as _td
    days = [r[0] for r in daily]
    # scan from newest backwards; collect (date, vals) pairs newest-first
    d0 = _date.fromisoformat(days[-1])
    run = []
    d = d0
    for _ in range(len(days) + 60):
        key = d.isoformat()
        v = by_day.get(key)
        if v is None:
            if len(run) >= WINDOW_HOURS:
                break
            run = []
        else:
            run.append((key, v))
            if len(run) >= WINDOW_HOURS:
                break
        d -= _td(days=1)
    if len(run) < WINDOW_HOURS:
        return None
    run = run[:WINDOW_HOURS]          # newest-first now
    run.reverse()                      # oldest -> newest
    base = run[-1][0]                  # newest day WITH data = context end
    vals = [v for _, v in run]
    return base, vals


def station_files(usa, wban, nyears=MAX_YEARS_BACK):
    """Most recent existing yearly paths for a station (newest first)."""
    import glob
    root = os.path.join(HOUR_DIR)
    years = sorted((int(y) for y in os.listdir(root) if y.isdigit()), reverse=True)
    got = []
    for y in years[:nyears]:
        p = os.path.join(root, str(y), f"{usa}-{wban}-{y}.gz")
        if os.path.exists(p):
            got.append(p)
    return got


def load_country_rings():
    """Borders for the active MODE: world countries or US state lines."""
    import json
    path = WORLD if MODE == "WORLD" else STATES
    rings = []
    with open(path) as f:
        gj = json.load(f)
    for feat in gj["features"]:
        prop = feat.get("properties", {})
        if MODE == "CONUS" and (prop.get("STATE") in ("02", "15")
                                or prop.get("NAME") in ("Alaska", "Hawaii", "Puerto Rico")):
            continue
        geom = feat["geometry"]
        if geom is None or geom["type"] not in ("Polygon", "MultiPolygon"):
            continue
        polys = geom["coordinates"] if geom["type"] == "Polygon" else \
                [p for mp in geom["coordinates"] for p in mp]
        for poly in polys:
            ring = [(c[0], c[1]) for c in poly]
            lons = [p[0] for p in ring]; lats = [p[1] for p in ring]
            if MODE == "CONUS" and (max(lats) < LAT0 or min(lats) > LAT1
                                    or max(lons) < LON0 or min(lons) > LON1):
                continue
            rings.append(ring)
    return rings


def main():
    t0 = time.time()
    geo = load_geo()              # station "SA-WBAN" -> (lat, lon)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = WeatherNetConfig(**ck["config"])
    net = WeatherNet(cfg).eval(); net.load_state_dict(ck["model_state"])
    nm = Normalizer(); nm.load_state_dict(ck["normalizer"])
    print(f"[ckpt] n_params={ck['n_params']:,} step={ck.get('step')}", flush=True)

    xs, cs, geos, bases = [], [], [], []
    n_scan = 0
    for key, (lat, lon) in geo.items():
        usa, wban = key.split("-")
        files = station_files(usa, wban)
        if not files:
            continue
        n_scan += 1
        rows = []
        for p in files:                      # read each yearly file in FULL
            rows.extend(daily_rows(p))
        w = last_gapfree_window(rows)
        if w is None:
            continue
        base, vals = w
        if len(vals) != WINDOW_HOURS:
            continue
        x = np.asarray(vals, dtype=np.float32)
        xs.append(x); cs.append(x.mean(0)); geos.append((lat, lon)); bases.append(base)
        if n_scan % 500 == 0:
            print(f"[scan] {n_scan} stations, {len(xs)} windows "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    if not xs:
        raise SystemExit("no usable windows found")
    xs = np.stack(xs); cs = np.stack(cs)
    geo_arr = np.asarray(geos, dtype=np.float64)

    from datetime import date as _date, timedelta as _td
    base_dates = np.array([_date.fromisoformat(b) for b in bases])
    newest = base_dates.max()
    keep = base_dates >= newest - _td(days=RECENT_DAYS)
    xs = xs[keep]; cs = cs[keep]; geo_arr = geo_arr[keep]
    base_dates = base_dates[keep]
    print(f"[data] {n_scan} stations scanned, {len(xs)} windows in coherent "
          f"range {newest - _td(days=RECENT_DAYS)}..{newest} "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- forward in CPU chunks --------------------------------------------
    xt = torch.tensor(nm.transform_x(xs))
    ct = torch.tensor(nm.transform_ctx(cs))
    preds = []
    B = 256
    with torch.no_grad():
        for i in range(0, len(xt), B):
            y = net(xt[i:i + B], ct[i:i + B])
            preds.append(nm.inverse_y(y).float().numpy())
            print(f"[fwd] {min(i+B, len(xt))}/{len(xt)} [{time.time()-t0:.0f}s]", flush=True)
    pred = np.concatenate(preds)
    del xt, ct

    # ---- interpolation + rendering ----------------------------------------
    import scipy.interpolate as si
    from PIL import Image, ImageDraw, ImageFont

    GRID = 0.5 if MODE == "WORLD" else 0.15
    lon_g = np.arange(LON0, LON1, GRID); lat_g = np.arange(LAT0, LAT1, GRID)
    GL, GT = np.meshgrid(lon_g, lat_g)
    pts_ll = np.stack([GT.ravel(), GL.ravel()], axis=1)
    # memory-safe nearest-station mask via KD-tree (avoids huge cross product)
    from scipy.spatial import cKDTree
    tree = cKDTree(geo_arr)
    dist = tree.query(pts_ll, k=1)[0]
    mask = (dist.reshape(GL.shape) <= 1.7)

    stops = [(1.00, 0.02, 0.27), (0.96, 0.34, 0.13), (0.95, 0.86, 0.22),
             (0.34, 0.88, 0.42), (0.22, 0.69, 0.55), (0.18, 0.42, 0.83),
             (0.31, 0.11, 0.66), (0.48, 0.06, 0.50), (0.85, 0.45, 0.60)]
    tvals = np.linspace(-30, 45, len(stops))
    def maps(temp):
        q = np.clip((temp - tvals[0]) / (tvals[-1] - tvals[0]), 0, 1) * (len(stops) - 1)
        return np.stack([np.interp(q, np.arange(len(stops)), [s[k] for s in stops])
                         for k in range(3)], axis=-1)

    leads = [(0, "day+1"), (6, "day+7"), (13, "day+14"), (20, "day+21")]
    borders = load_country_rings()
    for li, (ld, tag) in enumerate(leads):
        vals = pred[:, ld]
        f = si.griddata(geo_arr, vals, pts_ll, method="linear").reshape(GL.shape)
        fnn = si.griddata(geo_arr, vals, pts_ll, method="nearest").reshape(GL.shape)
        field = np.where(np.isfinite(f), f, fnn)
        field = np.where(mask, field, np.nan)
        nf = np.isfinite(field).sum()
        if nf < 1000:
            raise SystemExit(f"[render] degenerate interpolation: only {nf} cells")
        print(f"[render] lead {tag}: finite={nf} mean={np.nanmean(field):.2f} "
              f"min={np.nanmin(field):.2f} max={np.nanmax(field):.2f}", flush=True)
        body = np.where(np.isnan(field), 0.0, field)
        rgb = maps(body) * 255.0
        rgb = np.where(np.isnan(field)[..., None], 220, rgb).astype(np.uint8)
        img = Image.fromarray(rgb, "RGB")
        img = img.resize((img.width * 3, img.height * 3), Image.LANCZOS)
        d = ImageDraw.Draw(img)
        def proj(ring):
            return [(round((lon - LON0) / (LON1 - LON0) * img.width),
                     round((LAT1 - lat) / (LAT1 - LAT0) * img.height))
                    for lon, lat in ring]
        for ring in borders:
            d.line(proj(ring), fill=(40, 40, 40), width=2)
        px = ((geo_arr[:, 1] - LON0) / (LON1 - LON0) * img.width).astype(int)
        py = ((LAT1 - geo_arr[:, 0]) / (LAT1 - LAT0) * img.height).astype(int)
        for ax, ay in zip(px, py):
            if 0 <= ax < img.width and 0 <= ay < img.height:
                d.ellipse([ax - 1, ay - 1, ax + 1, ay + 1], fill=(30, 30, 30))
        fc_min = (base_dates + _td(days=ld)).min()
        fc_max = (base_dates + _td(days=ld)).max()
        d.text((8, 8), f"WeatherNet forecast {fc_min}..{fc_max} "
               f"~mean {np.nanmean(vals):6.1f} C  min {np.nanmin(vals):5.1f} "
               f"max {np.nanmax(vals):5.1f}", fill=(255, 255, 255))
        d.text((8, 26), f"stations={len(vals)} lead={tag} ({MODE}: hourly ISD-Lite archive)",
               fill=(200, 200, 200))
        p = os.path.join(OUTDIR, f"gradient_{MODE}_lead_{tag}.png")
        bar = np.zeros((420, 60, 3), dtype=np.float64)
        ramp = np.linspace(-30, 45, 420)
        bar[:, :, :] = maps(ramp)[:, None, :]
        bimg = Image.fromarray((bar * 255).astype(np.uint8).repeat(2, axis=1), "RGB")\
                .resize((40, 320), Image.NEAREST)
        canvas = Image.new("RGB", (img.width + 70, img.height), (245, 245, 245))
        canvas.paste(img, (0, 0))
        canvas.paste(bimg, (img.width + 20, (img.height - 320) // 2))
        canvas.save(p)
        print(f"[map] {p}", flush=True)
    print(f"[done] images in {OUTDIR} [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()