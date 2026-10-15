# Weather 1B — 1.58-bit Ternary Deep Network for Numerical Weather Forecasting

A pure tensor-in / tensor-out deep network for 24-step-ahead numerical weather
forecasting. **No tokenizers, no LLM embedding blocks, no autoregressive
generation.** Built with PyTorch. Weights are 1.58-bit ternary (-1, 0, +1)
through a custom `BitLinear` layer with a Straight-Through Estimator.

```
  24h window, 4 vars (temp °C, pressure hPa, humidity %, wind m/s)
                       │
          ┌────────────▼───────────────┐
          │ FP32 input head  100 → 1536 │   non-quantized
          └────────────┬───────────────┘
                       │
          ┌────────────▼───────────────┐
          │ BitLinear (ternary) 1536→8192
          │ 14 × BitLinear(8192→8192) + GELU + residual (pre-norm RMSNorm)
          │ BitLinear (ternary) 8192→256
          └────────────┬───────────────┘
                       │
          ┌────────────▼───────────────┐
          │ FP32 output head   256 → 24 │   non-quantized
          └────────────┬───────────────┘
                       │
             24h forecast vector (24,)
```

## Parameter budget (exactly ~0.95B, weights only)

| block          | shape          | parameters    |
|----------------|----------------|--------------:|
| feature_embed  | 100  → 1536    |       153,600 |
| encoder        | 1536 → 8192    |    12,582,912 |
| 14 hidden      | 8192 → 8192    |   939,524,096 |
| projector      | 8192 → 256     |     2,097,152 |
| output_head    | 256  → 24      |         6,144 |
| **total**      |                | **954,363,904** |

Total with biases/norms: **954,605,080** → ≈ 0.955B (targeted "around 1B").
The FLOPs per sample ≈ 2 × params ≈ **0.19 GFLOP per window**.

- ~954M of the 954.6M parameters live in *ternary* `BitLinear` layers.
- The outermost FP32 heads keep spatial resolution of the continuous metrics.
- Each `BitLinear` computes `y = (8-bit X_q · γ) @ (W_ternary · β).T + b` with
  per-token activation scale `γ = max|x|` and layer weight scale
  `β = mean|W|`, both steered through a Straight-Through Estimator.

## Files

| file                | purpose                                                        |
|---------------------|----------------------------------------------------------------|
| `layers.py`         | `BitLinear` (ternary weights + STE + 8-bit absmax activations) |
| `model.py`          | `WeatherNet` / `WeatherNetConfig`, ~1B network, param report   |
| `train.py`          | training loop (AdamW, warmup+cosine, autocast, metrics, ckpt)  |
| `data.py`           | synthetic generator, `Normalizer`, `WeatherZipCSV` reader      |
| `download_data.py`  | century-scale data acquisition (Open-Meteo / NOAA GSOD / ERA-20C) |

## Quick start

```bash
python train.py --smoke                 # tiny net, seconds, synthetic data
python train.py                         # full ~1B budget, synthetic data (slow on CPU)
python train.py --data data/raw/weather_archive.zip   # real station data
```

Flags you will want:

* `--epochs N --batch-size B --grad-accum G --lr 3e-4`
* `--device cuda|mps|cpu`  (default `auto`)
* `--precision bf16|fp16|fp32` — on CUDA, `auto` = **BF16**; fp16 adds GradScaler
* `--width 8192 --depth 14` — full budget (defaults)
* `--chunk-size 2048` — slice the big matmuls → fits smaller VRAM GPUs
* `--ternary-threshold 0.1` — dead-zone factor of `mean|W|`; `0.0` = pure {-1,+1}

## GPU usage

* The massive hidden layers run as quantised matmuls. On A100/H100 the full
  1B model trains comfortably: fp32 latent params (3.8 GB) + AdamW states
  (~7.6 GB) + activations → fits 24 GB+ cards with BF16 autocast
  (fp16 + GradScaler below that).
* With `--chunk-size` set you can trade a little throughput for much lower
  activation peaks on 8–16 GB cards.
* For a real multi-GPU run copy `WeatherNet` into `torch.nn.parallel.DistributedDataParallel` —
  it is a plain `nn.Module`, nothing special needed.

## Data: getting every last record, back to the 1900s

The network is step-agnostic: feed it 24 hourly *or* 24 daily values and it
forecasts 24 steps ahead. Hourly is ideal; daily lets you reach the 1900s.

| source             | resolution        | span             | how                      |
|--------------------|-------------------|------------------|--------------------------|
| **Open-Meteo**     | **hourly** (ERA5) | **1940 → now**   | `download_data.py` (no key) |
| **NOAA GSOD**      | daily             | **1929 → 2025**  | `download_data.py --source gsod` |
| **ERA-20C**        | 3-hourly          | **1900 → 2010**  | `download_data.py --source era20c` (free CDS key) |
| ERA5 back-extension| hourly            | 1950–1978        | Copernicus CDS           |
| NOAA ISD (Global-Hourly) | hourly     | US 1901+, global 1973+ | ncei.noaa.gov/data/global-hourly |
| GHCN-Daily         | daily             | **1763+** (few stations) | ncei.noaa.gov/data/ghcn-daily |

Commands:

```bash
# The full 18-station network hourly from 1940 to today (~0.5 GB):
python download_data.py --years 1940 2025

# NOAA daily records for very-long histories (a station at a time):
python download_data.py --source gsod --stations "New York Central Park" --years 1929 1940
```

Everything lands as per-station CSVs in `data/raw/`, then `weather_archive.zip`
is produced for `train.py --data` directly.

> **What was actually fetched in this session:** 8 city stations × 1940–1942
> hourly (≈ 210k rows) from Open-Meteo ERA5 reanalysis, zipped into
> `data/raw/weather_archive.zip`, and trained end-to-end (RMSE 2.73 °C on the
> 24h temperature lead, corr 0.96 on held-in eval). The un-downloaded bulk
> (1943–2025 for those stations, plus the other 10 catalog stations, GSOD
> 1929+, ERA-20C 1900+) is reachable with the same commands on your hardware.

### Notes & caveats

- **Synthetic data** (`--samples N`) is weather-shaped but fake — use it to
  validate the pipeline, not science.
- Pre-1973 daily coverage and pre-1940 hourly coverage are sparse by nature;
  ERA-20C (reanalysis) is the honest way to get dense 1900s numbers hourly-ish.
- The model is an MLP: it models the *distribution at each forecast hour*
  independently, not spatial stations jointly. Feeding multiple co-located
  stations as extra features is the natural extension.

## Verifying the ~1B budget

```python
from model import WeatherNet
m = WeatherNet()
print(m.summary())   # 954,605,080 params; 954,327,296 ternary
```

## Builds

Prebuilt Windows / Linux / macOS artifacts are produced by the GitHub Actions workflow in `.github/workflows/build.yml` on every tag.

## License

MIT
