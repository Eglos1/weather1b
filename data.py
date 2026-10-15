"""
data.py
=======
Data plumbing for the 1.58-bit ternary weather network.

* :class:`SyntheticWeatherGenerator`   -- realistic-looking dummy series
  (diurnal temperature cycle, synoptic pressure waves, AR(1) noise) so the
  training loop runs out-of-the-box without any downloads.

* :class:`Normalizer`  -- per-variable z-scoring used before the model and
  its algebraic inverse when reporting RMSE in physical units later.

* :class:`WeatherZipCSV`  -- reader for the *real* station archive produced
  by ``download_data.py`` (a zip of ``*.csv`` files, one hourly series per
  station, columns ``date,temperature_c,pressure_hpa,humidity_pct,wind_ms``).

All tensors flowing into the network are standardised to roughly N(0,1);
the FP32 input head then re-scales them before the ternary body.
"""

from __future__ import annotations

import csv
import io
import zipfile

import numpy as np

from model import WINDOW_HOURS, NUM_VARS, NUM_CONTEXT, OUT_FEATURES

F64 = np.float64


# ---------------------------------------------------------------------------
# Synthetic (dummy) data
# ---------------------------------------------------------------------------
class SyntheticWeatherGenerator:
    """Draw ``n`` synthetic 48-step windows of 4 weather variables.

    The first 24 steps are the model input window; the second 24 steps are
    the forecast target (the same physical process continued forward), so
    the plain continuation task is exactly solvable *in principle* -- useful
    for validating gradients, the STE and the training loop on this box.
    """

    def __init__(self, n: int, seed: int = 0) -> None:
        self.n = int(n)
        self.seed = int(seed)

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        horizon = WINDOW_HOURS + OUT_FEATURES                 # 48 hours
        n = self.n
        n_t = np.arange(horizon, dtype=F64)[None, :, None]    # (1, 48, 1) hours

        # ---- per-variable base dynamics (time in hours) ------------------
        #   temperature:  15 C mean, diurnal ±8 C (24 h) + seasonal drift
        #   pressure:     ~1013 hPa + synoptic waves (96 h / 264 h)
        #   humidity:      ~62 % anti-correlated with temperature anomaly
        #   wind:          ~5 m/s with a ~72 h travelling wave
        base_mu = np.array([15.0, 1013.0, 62.0, 5.0], dtype=F64)
        base_amp = np.array([8.0, 7.0, 18.0, 3.0], dtype=F64)
        base_per = np.array([24.0, 96.0, 24.0, 72.0], dtype=F64)
        phase = rng.uniform(0.0, 2 * np.pi, size=(n, 1, NUM_VARS))

        series = base_mu[None, None, :] + base_amp[None, None, :] * \
            np.sin(2.0 * np.pi * n_t / base_per + phase)      # (n, 48, 4)

        # ---- AR(1) persistent noise tails (weather "memory") --------------
        rho = np.array([0.85, 0.90, 0.80, 0.60], dtype=F64)
        sig = np.array([1.2, 2.5, 6.0, 0.9], dtype=F64)
        z = rng.standard_normal((n, horizon, NUM_VARS)) * sig
        for h in range(1, horizon):                            # cheap 48-step AR
            z[:, h] += rho * z[:, h - 1]
        series = series + z

        # humidity anti-correlates with the temperature anomaly
        series[:, :, 2] += -0.35 * (series[:, :, 0] - series[:, :, 0].mean(axis=1, keepdims=True))

        x = series[:, :WINDOW_HOURS, :]                        # 24h look-back
        y = series[:, WINDOW_HOURS:, 0].copy()                 # 24h temperature lead
        y += 0.15 * (series[:, WINDOW_HOURS:, 1].mean(axis=1, keepdims=True) - 1013.0)  # pressure mix-in
        y += 0.02 * (series[:, WINDOW_HOURS:, 2].mean(axis=1, keepdims=True) - 62.0)    # humidity mix-in
        # A *learnable* signal: a warm window today -> a warm next day.
        # (The unlearnable AR-noise continuation keeps an irreducible floor,
        #  so correlation rises but never trivially saturates.)
        y = y + 1.2 * (x[..., 0].mean(axis=1, keepdims=True) - x[..., 0].mean())
        return x.astype(np.float32), y.astype(np.float32)


def statistics_context(x_window: np.ndarray) -> np.ndarray:
    """Rolling per-variable means of the window -> ``(N, NUM_CONTEXT)``."""
    return x_window.mean(axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------
class Normalizer:
    """Per-variable mean/std statistics with physical-unit round-trips."""

    def __init__(self) -> None:
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_std: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Normalizer":
        x = x.reshape(-1, x.shape[-1])
        self.x_mean = x.mean(axis=0).astype(F64)
        self.x_std = x.std(axis=0).astype(F64).clip(min=1e-6)
        y = y.reshape(-1)
        self.y_mean = y.mean(axis=0).astype(F64)
        self.y_std = y.std(axis=0).astype(F64).clip(min=1e-6)
        return self

    # -- input transforms -------------------------------------------------
    def transform_x(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.x_mean) / self.x_std).astype(np.float32)

    def transform_ctx(self, c: np.ndarray) -> np.ndarray:
        return ((c - self.x_mean) / self.x_std).astype(np.float32)

    # -- target transforms ------------------------------------------------
    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.y_mean) / self.y_std).astype(np.float32)

    def inverse_y(self, y: np.ndarray | "torch.Tensor"):
        import torch
        if isinstance(y, torch.Tensor):
            return y * torch.tensor(self.y_std, dtype=y.dtype, device=y.device) \
                + torch.tensor(self.y_mean, dtype=y.dtype, device=y.device)
        return y * self.y_std + self.y_mean

    def state_dict(self) -> dict:
        return {
            "x_mean": np.asarray(self.x_mean),
            "x_std": np.asarray(self.x_std),
            "y_mean": np.asarray(self.y_mean),
            "y_std": np.asarray(self.y_std),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.x_mean = np.asarray(sd["x_mean"])
        self.x_std = np.asarray(sd["x_std"])
        self.y_mean = np.asarray(sd["y_mean"])
        self.y_std = np.asarray(sd["y_std"])


# ---------------------------------------------------------------------------
# Real archive reader
# ---------------------------------------------------------------------------
class WeatherZipCSV:
    """Streaming reader over a zip of per-station hourly CSVs.

    Archive layout expected (produced by ``download_data.py``):

        station.csv          ->  date,temperature_c,pressure_hpa,humidity_pct,wind_ms

    ``date`` is ISO ``YYYY-MM-DDTHH:MM``.  Missing *hourly* values become
    NaN, and windows that straddle a gap are discarded by the sliding sampler.
    """

    def __init__(self, archive_path: str) -> None:
        self.archive_path = archive_path
        with zipfile.ZipFile(archive_path) as z:
            self.entries = [n for n in z.namelist() if n.endswith(".csv")]

    def __len__(self) -> int:
        return sum(self._rows_per_entry(e) for e in self.entries)

    def _rows_per_entry(self, name: str) -> int:
        i = 0
        with zipfile.ZipFile(self.archive_path) as z:
            with z.open(name) as fh:
                for i, _ in enumerate(fh):
                    pass
        return i

    def iter_windows(self, limit: int | None = None):
        """Yield ``(window(24,4), ctx(4), target(24))`` numpy tuples.

        A sliding 48-step window is moved over each station series; any NaN
        (gap in the hourly record) ends the current run and the window
        restarts after the gap.  ~1e6 windows per 100 station-years.
        """
        produced = 0
        with zipfile.ZipFile(self.archive_path) as z:
            for name in self.entries:
                with z.open(name) as fh:
                    text = io.TextIOWrapper(fh, encoding="utf-8")
                    reader = csv.reader(text)
                    header = next(reader, None)
                    if header is None:
                        continue
                    cols = {c: i for i, c in enumerate(header)}
                    need = ["temperature_c", "pressure_hpa", "humidity_pct", "wind_ms"]
                    idx = [cols[c] for c in need]
                    t, vals = [], []
                    for row in reader:
                        if len(row) < max(idx) + 1:
                            continue
                        try:
                            v = []
                            for i in idx:
                                s = row[i].strip()
                                v.append(float(s) if s != "" else np.nan)
                            t.append(row[cols["date"]])
                            vals.append(v)
                        except (ValueError, IndexError):
                            continue
                series = np.asarray(vals, dtype=np.float32)     # (T, 4)
                # slide over the series of this station
                n = len(series)
                for i in range(0, n - (WINDOW_HOURS + OUT_FEATURES) + 1, WINDOW_HOURS):
                    blk = series[i : i + WINDOW_HOURS + OUT_FEATURES]
                    if np.isnan(blk).any():
                        continue
                    x = blk[:WINDOW_HOURS]
                    y = blk[WINDOW_HOURS:, 0]                    # lead var = temperature
                    yield x, x.mean(axis=0), y
                    produced += 1
                    if limit is not None and produced >= limit:
                        return

    def stats(self, limit: int | None = None):
        """Online per-variable mean/std over windows (x: 4 vars, y: 24-step).

        Matches ``Normalizer.fit`` semantics (x per-variable over all window
        rows; y per-step over all windows).  One pass, tiny memory.
        """
        w = 0
        xsum = np.zeros(4, dtype=F64)
        xsq = np.zeros(4, dtype=F64)
        ysum = np.zeros(OUT_FEATURES, dtype=F64)
        ysq = np.zeros(OUT_FEATURES, dtype=F64)
        for x, c, y in self.iter_windows(limit=limit):
            w += 1
            xv = x.reshape(-1, 4)
            xsum += xv.sum(axis=0)
            xsq += (xv * xv).sum(axis=0)
            ysum += y
            ysq += y * y
        if w == 0:
            raise RuntimeError("no usable windows in archive")
        nx = w * WINDOW_HOURS
        xm = xsum / nx
        xs = np.clip(xsq / nx - xm * xm, 1e-12, None) ** 0.5
        ym = ysum / w
        ys = np.clip(ysq / w - ym * ym, 1e-12, None) ** 0.5
        return xm, xs, ym, ys, w

    def iter_batches(self, batch_size: int, dtype, device, xm, xs, ym, ys,
                     limit: int | None = None):
        """Stream normalized windows as bf16/fp16 GPU batches -- no dataset.

        Yields ``(xw (B,24,4), ctx (B,4), yt (B,24))`` tensors already on
        ``device`` in ``dtype``, so the GPU never materialises full-precision
        copies of the training archive.
        """
        import torch
        xb, cb, yb = [], [], []
        for x, c, y in self.iter_windows(limit=limit):
            xb.append(((x - xm) / xs).astype(np.float32))
            cb.append(((c - xm) / xs).astype(np.float32))
            yb.append(((y - ym) / ys).astype(np.float32))
            if len(xb) == batch_size:
                yield (torch.tensor(np.stack(xb), dtype=dtype, device=device),
                       torch.tensor(np.stack(cb), dtype=dtype, device=device),
                       torch.tensor(np.stack(yb), dtype=dtype, device=device))
                xb, cb, yb = [], [], []
        if xb:
            yield (torch.tensor(np.stack(xb), dtype=dtype, device=device),
                   torch.tensor(np.stack(cb), dtype=dtype, device=device),
                   torch.tensor(np.stack(yb), dtype=dtype, device=device))

    def to_tensors(self, limit: int | None = None):
        """Materialise the whole (filtered) archive in memory.

        For very large century-scale archives prefer ``iter_windows`` and a
        separate statistics pass (see README).
        """
        xs, cs, ys = [], [], []
        for x, c, y in self.iter_windows(limit=limit):
            xs.append(x)
            cs.append(c)
            ys.append(y)
        if not xs:
            raise RuntimeError("no usable windows in archive")
        return np.stack(xs), np.stack(cs), np.stack(ys)


if __name__ == "__main__":
    gen = SyntheticWeatherGenerator(n=64, seed=1)
    xw, y = gen.sample()
    print("synthetic:", xw.shape, y.shape, "| temp range", xw[..., 0].min(), "..", xw[..., 0].max())
    nm = Normalizer().fit(xw, y)
    print("norm stats", nm.x_mean.round(1), nm.x_std.round(1))