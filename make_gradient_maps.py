#!/usr/bin/env python3
"""Render old-school "gradient forecast maps" from the trained WeatherNet.

For every decade->recent station in the GSOD archive: build its last
gap-free 24-day input window, predict the next 24 days of temperature,
then interpolate the predicted field over CONUS into smooth color
gradient PNGs (classic isotherm look) at several forecast leads.

CPU inference, chunked. GPU left alone (LLM owns it).
"""
import csv, io, math, os, sys, time, zipfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from model import WeatherNet, WeatherNetConfig, WINDOW_HOURS, NUM_VARS, NUM_CONTEXT
from data import Normalizer

ROOT = os.path.dirname(os.path.abspath(__file__))
ZIP = os.path.join(ROOT, "data", "gsod_daily_full.zip")
HIST = os.path.join(ROOT, "data", "isd-history.csv")
CKPT = os.path.join(ROOT, "runs", "weather-gpu", "final.pt")
STATES = os.path.join(ROOT, "data", "us_states_20m.json")
OUTDIR = os.path.join(ROOT, "maps")
os.makedirs(OUTDIR, exist_ok=True)

# CONUS bounds
LAT0, LAT1, LON0, LON1 = 22.0, 50.0, -128.0, -64.0

# Coherent-time filter: only use stations whose last window is within this many
# days of the newest record. Otherwise the map mixes 1943 forecasts with 2025
# ones (which is what produced the nonsense-looking picture before).
RECENT_DAYS = 90


def load_station_geo():
    """sisd-history.csv -> filename-key -> (lat, lon)."""
    geo = {}
    with open(HIST, encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                usa, wban, lat, lon = row["USAF"], row["WBAN"], float(row["LAT"]), float(row["LON"])
            except (KeyError, ValueError):
                continue
            key = (usa.zfill(6) + wban.zfill(5)) + ".csv"
            if LAT0 <= lat <= LAT1 and LON0 <= lon <= LON1:
                geo[key] = (lat, lon)
    print(f"[geo] {len(geo)} CONUS stations from catalog", flush=True)
    return geo


def parse_last_window(fh, cols):
    """Tail-parse one station CSV; return last gap-free 24-day window (24,4)
    plus its base date. Fast: only the tail rows are kept."""
    need = ["temperature_c", "pressure_hpa", "humidity_pct", "wind_ms"]
    idx = [cols[c] for c in need]
    tail = []
    ti = cols["date"]
    for row in fh:
        row = row.decode("utf-8", errors="replace").split(",")
        if len(row) < max(idx) + 1:
            continue
        try:
            v = [float(row[i].strip()) for i in idx]
        except ValueError:
            continue
        d = row[ti].strip()
        tail.append((d, v))
        if len(tail) > 400:
            tail = tail[-400:]
    # find the LAST run of >=24 consecutive (daily) rows with no NaN values:
    # scan newest->oldest; a clean run reaching 24 rows is the answer.
    run = []
    best = None
    for d, v in reversed(tail):
        if any(math.isnan(x) for x in v):
            if len(run) >= WINDOW_HOURS:
                break
            run = []
            continue
        run.append(v)
        if len(run) >= WINDOW_HOURS:
            best = (d, list(reversed(run[:WINDOW_HOURS])))
            break
    if best is None:
        return None
    base, vals = best
    if len(vals) < WINDOW_HOURS:
        return None
    arr = np.asarray(vals[-WINDOW_HOURS:], dtype=np.float32)
    return base, arr


def load_states_rings():
    """us_states_20m.json -> list of polygon rings [(lon, lat), ...] within CONUS."""
    import json
    rings = []
    with open(STATES) as f:
        gj = json.load(f)
    for feat in gj["features"]:
        prop = feat.get("properties", {})
        if prop.get("STATE") in ("02", "15") or prop.get("NAME") in ("Alaska", "Hawaii", "Puerto Rico"):
            continue
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "Polygon" else \
                [p for mp in geom["coordinates"] for p in mp]
        for poly in polys:
            ring = [(c[0], c[1]) for c in poly]
            lons = [p[0] for p in ring]; lats = [p[1] for p in ring]
            if max(lats) < LAT0 or min(lats) > LAT1 or max(lons) < LON0 or min(lons) > LON1:
                continue
            rings.append(ring)
    return rings


def main():
    t0 = time.time()
    geo = load_station_geo()

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = WeatherNetConfig(**ck["config"])
    net = WeatherNet(cfg).eval()
    net.load_state_dict(ck["model_state"])
    nm = Normalizer()
    nm.load_state_dict(ck["normalizer"])
    print(f"[ckpt] n_params={ck['n_params']:,} step={ck.get('step')} "
          f"epoch={ck.get('epoch')}", flush=True)

    # ---- collect inputs for every CONUS station in the archive ------------
    xs, cs, geos, bases = [], [], [], []
    n_stations = 0
    with zipfile.ZipFile(ZIP) as z:
        entries = [n for n in z.namelist() if n.endswith(".csv")]
        for name in entries:
            key = os.path.basename(name)
            if key not in geo:
                continue
            n_stations += 1
            with z.open(name) as fh:
                first = fh.readline()
                cols = {c: k for k, c in enumerate(first.decode("utf-8", errors="replace").strip().split(","))}
                if "date" not in cols:
                    continue
                parsed = parse_last_window(fh, cols)
            if parsed is None:
                continue
            base, x = parsed
            xs.append(x)
            cs.append(x.mean(axis=0))
            geos.append(geo[key])
            bases.append(base)
    if not xs:
        raise SystemExit("no usable CONUS windows found")
    # ---- keep only stations in one coherent forecast window ----------------
    from datetime import date as _date, timedelta as _td
    base_dates = np.array([_date.fromisoformat(b[:10]) for b in bases])
    newest = base_dates.max()
    keep = base_dates >= newest - _td(days=RECENT_DAYS)
    xs = np.stack(xs)[keep]; cs = np.stack(cs)[keep]
    geo_arr = np.asarray(geos, dtype=np.float64)[keep]
    bases = [b for b, k in zip(bases, keep) if k]
    print(f"[data] {n_stations} CONUS stations scanned, "
          f"{len(xs)} windows in coherent window "
          f"({keep.sum()} from {newest - _td(days=RECENT_DAYS)}..{newest})"
          f"  [{time.time()-t0:.0f}s]", flush=True)

    # ---- forward in CPU chunks --------------------------------------------
    xt = torch.tensor(nm.transform_x(xs))
    ct = torch.tensor(nm.transform_ctx(cs))
    preds = []
    B = 256
    with torch.no_grad():
        for i in range(0, len(xt), B):
            y = net(xt[i:i + B], ct[i:i + B])
            preds.append(nm.inverse_y(y).float().numpy())
            print(f"[fwd] {min(i+B, len(xt))}/{len(xt)}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    pred = np.concatenate(preds)  # (N, 24) predicted daily temps
    del xt, ct
    print(f"[fwd] done. pred shape {pred.shape} "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- interpolation + gradient rendering -------------------------------
    import scipy.interpolate as si
    from PIL import Image, ImageDraw

    lon_g = np.arange(LON0, LON1, 0.15)
    lat_g = np.arange(LAT0, LAT1, 0.15)
    GL, GT = np.meshgrid(lon_g, lat_g)

    # colormap: classic forecast ramp blue..green..red
    stops = [(1.00, 0.02, 0.27), (0.96, 0.34, 0.13), (0.95, 0.86, 0.22),
             (0.34, 0.88, 0.42), (0.22, 0.69, 0.55), (0.18, 0.42, 0.83),
             (0.31, 0.11, 0.66), (0.48, 0.06, 0.50), (0.85, 0.45, 0.60)]
    tvals = np.linspace(-30, 45, len(stops))
    def maps(temp):
        q = np.clip((temp - tvals[0]) / (tvals[-1] - tvals[0]), 0, 1) * (len(stops) - 1)
        lo = np.floor(q).astype(int); hi = np.minimum(lo + 1, len(stops) - 1)
        w = q - lo
        rgb = np.stack([np.interp(q, np.arange(len(stops)), [s[k] for s in stops])
                        for k in range(3)], axis=-1)
        return rgb

    # distance mask: keep cells within ~1.7 deg (in (lat,lon) space) of a station
    dgrid = ((GT[:, :, None] - geo_arr[:, 0][None, None, :]) ** 2 +
             (GL[:, :, None] - geo_arr[:, 1][None, None, :]) ** 2) ** 0.5
    nearest_station = dgrid.min(axis=2)
    mask = nearest_station <= 1.7
    # points AND queries both in (lat, lon) order
    pts_ll = np.stack([GT.ravel(), GL.ravel()], axis=1)

    leads = [(0, "day+1"), (6, "day+7"), (13, "day+14"), (20, "day+21")]
    state_rings = load_states_rings()
    from datetime import date as _date, timedelta as _td
    base_dates = np.array([_date.fromisoformat(b[:10]) for b in bases])
    fc = base_dates + np.array([_td(days=ld) for ld, _ in leads])[:, None]
    for li, (ld, tag) in enumerate(leads):
        vals = pred[:, ld]
        f = si.griddata(geo_arr, vals, pts_ll, method="linear").reshape(GL.shape)
        fnn = si.griddata(geo_arr, vals, pts_ll, method="nearest").reshape(GL.shape)
        field = np.where(np.isfinite(f), f, fnn)
        field = np.where(mask, field, np.nan)
        nf = np.isfinite(field).sum()
        if nf < 1000:
            raise SystemExit(f"[render] interpolation degenerate: only {nf} finite cells")
        print(f"[render] lead {tag}: finite={nf} mean={np.nanmean(field):.2f} "
              f"min={np.nanmin(field):.2f} max={np.nanmax(field):.2f}", flush=True)
        body = np.where(np.isnan(field), 0.0, field)
        rgb = maps(body) * 255.0
        rgb = np.where(np.isnan(field)[..., None], 220, rgb).astype(np.uint8)
        img = Image.fromarray(rgb, "RGB")
        img = img.resize((img.width * 3, img.height * 3), Image.LANCZOS)
        img = img.convert("RGB")
        d = ImageDraw.Draw(img)
        # US state boundary overlay (coastline + inter-state borders)
        def proj_pts(ring):
            return [(round((lon - LON0) / (LON1 - LON0) * img.width),
                     round((LAT1 - lat) / (LAT1 - LAT0) * img.height))
                    for lon, lat in ring]
        for ring in state_rings:
            d.line(proj_pts(ring), fill=(40, 40, 40), width=2)
        # station dots
        px = ((geo_arr[:, 1:2] - LON0) / (LON1 - LON0) * img.width).astype(int).ravel()
        py = ((LAT1 - geo_arr[:, 0:1]) / (LAT1 - LAT0) * img.height).astype(int).ravel()
        for (ax, ay) in zip(px, py):
            if 0 <= ax < img.width and 0 <= ay < img.height:
                d.ellipse([ax - 1, ay - 1, ax + 1, ay + 1], fill=(30, 30, 30))
        d.text((8, 8), f"WeatherNet forecast {fc[li].min()}..{fc[li].max()}"
               f"  ~mean {np.nanmean(vals):6.1f} C  min {np.nanmin(vals):5.1f}"
               f"  max {np.nanmax(vals):5.1f}", fill=(255, 255, 255))
        d.text((8, 26), f"stations={len(vals)} lead={tag}", fill=(200, 200, 200))
        p = os.path.join(OUTDIR, f"gradient_lead_{tag}.png")
        # vertical colorbar
        bar = np.zeros((420, 60, 3), dtype=np.float64)
        ramp = np.linspace(-30, 45, 420)
        bar[:, :, :] = maps(ramp)[:, None, :]
        bimg = Image.fromarray((bar * 255).astype(np.uint8).repeat(2, axis=1), "RGB").resize((40, 320), Image.NEAREST)
        canvas = Image.new("RGB", (img.width + 70, img.height), (245, 245, 245))
        canvas.paste(img, (0, 0))
        canvas.paste(bimg, (img.width + 20, (img.height - 320) // 2))
        canvas.save(p)
        print(f"[map] {p}", flush=True)
    print(f"[done] images in {OUTDIR} [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()