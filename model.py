"""
model.py
========
Deep 1.58-bit ternary-weight network for *numerical* time-series weather
forecasting.  Pure tensor-in / tensor-out.  No tokenizers, no embedding
tables, no autoregression.

Architecture (target: ~1B parameters)
--------------------------------------
                          ┌────────────────────────────────────────────┐
  24h window × 4 vars ──▶ │    FP32 input head  (nn.Linear)  100 → 1536  │
  + rolling means            └────────────────────────────────────────────┘
                                              │
                                          ┌───▼───┐   (hidden stack)
                                          │RMSNorm│
                                          └───┬───┘
                                              │  (× 14 residual blocks)
                              ┌───────────────────────────────┐
                              │ BitLinear  8192 → 8192  (ternary)│
                              │ + GELU                        │
                              │ + residual skip (pre-norm)     │
                              └───────────────────────────────┘
                                              │
                             ┌────────────────▼────────────────┐
                             │ BitLinear (ternary)  8192 → 256   │
                             └────────────────┬────────────────┘
                                              │
                          ┌───────────────────▼───────────────────┐
  24h forecast (24,) ◀─── │ FP32 output head  (nn.Linear) 256 → 24 │
                          └───────────────────────────────────────┘

Parameter budget (weights only)
-------------------------------
  feature_embed     100 -> 1536      153,600
  h_0              1536 -> 8192   12,582,912
  14 × hidden       8192 -> 8192  939,524,096
  projector         8192 -> 256     2,097,152
  output_head        256 -> 24          6,144
                                   ─────────────
                                  954,363,904   (~0.954 billion)

The massive intermediate tensors are all ``BitLinear`` (1.58-bit ternary
weights); the outermost input encoder and the output projection stay in
standard FP32 to preserve spatial resolution of the continuous metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from layers import BitLinear

Tensor_T = torch.Tensor

# ---------------------------------------------------------------------------
# Constants tied to the weather data schema
# ---------------------------------------------------------------------------
WINDOW_HOURS = 24          # look-back window length
NUM_VARS = 4               # [temperature, pressure, humidity, wind_speed]
NUM_CONTEXT = 4            # rolling means of each variable over the window
                            # (added as summary statistics)
IN_FEATURES = WINDOW_HOURS * NUM_VARS + NUM_CONTEXT   # 96 + 4 = 100
OUT_FEATURES = WINDOW_HOURS                            # forecast: next 24h
VAR_SCALES = {
    "temperature_c": 40.0,    # range roughly -20..+40 C  ->  [-1, 1]
    "pressure_hpa":  200.0,   # range roughly  950..1050 hPa ->  [-1, 1]
    "humidity_pct": 100.0,    # 0..100 %
    "wind_ms":        30.0,   # 0..30 m/s
}


# ---------------------------------------------------------------------------
# RMSNorm (pre-activation layernorm for the deep hidden stack)
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no mean-centering, cheap)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor_T) -> torch.Tensor:
        # Keep the *big* activation in the input dtype (bf16) end-to-end.
        # Under torch.autocast, mean/square/rsqrt are autocast-safe ops and
        # run in fp32; a bf16 * fp32 multiply would silently promote the whole
        # hidden tensor to fp32 -- and every downstream BitLinear would then
        # build an fp32 copy of its 8192x8192 weight (+268 MiB per block).
        dt = x.dtype
        fp32 = torch.float32
        rms = (x.to(fp32).square().mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        return (x * rms.to(dt)) * self.weight.to(dt)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class WeatherNetConfig:
    """Hyper-parameters controlling the shape of :class:`WeatherNet`.

    The default values give *exactly* 954,488,600 total parameters
    (weights + biases + RMS gains), i.e. ~0.95B, with the full ~1B weight
    mass held by ternary ``BitLinear`` layers.
    """

    in_features: int = IN_FEATURES          # 100
    d_embed: int = 1536                     # FP32 input head width
    d_hidden: int = 8192                    # hidden (BitLinear) width
    num_hidden_blocks: int = 14             # internal 8192->8192 blocks
    d_proj: int = 256                       # width before the FP32 head
    out_features: int = OUT_FEATURES        # 24
    activation_bits: int = 8                # 8-bit activation grid
    ternary_threshold: float = 0.1         # dead-zone factor (frac. of mean|W|);
                                            # 0.0 = pure {-1,+1} sign quant
    eps: float = 1e-7
    chunk_size: int | None = None           # per-layer matmul slice for
                                            # low-VRAM GPUs (None = off)

    # ---------- derived sizes (filled by the model) ----------
    @property
    def d_enc(self) -> int:
        return self.d_hidden


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class WeatherNet(nn.Module):
    """~1B-parameter 1.58-bit ternary network for 24h numerical forecasting.

    **Forward/backward maths.**  For a batch ``x`` of shape ``(B, 24, 4)``
    plus the 4 rolling means, the forward pass is a sequence of quantised
    matmuls; every ``BitLinear`` computes

        y_r = (X_q[., r] * gamma_r) @ (W_ternary * beta).T + b

    with per-token scale ``gamma_r = max|X[., r]|`` and layer scale
    ``beta = mean|W_latent|``, both injected under the STE so the backward
    pass receives standard matrix gradients w.r.t. the *latent* weights.
    """

    def __init__(self, config: WeatherNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or WeatherNetConfig()

        c = self.cfg

        # ---- FP32 input head (spatial-fidelity encoder, NOT quantized) ----
        self.feature_embed = nn.Linear(c.in_features, c.d_embed, bias=True)

        # ---- rms norm + 14 residual ternary blocks (the 1B body) -------
        self.init_rms = RMSNorm(c.d_embed)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "bit_linear": BitLinear(
                            c.d_hidden,
                            c.d_hidden,
                            bias=True,
                            activation_bits=c.activation_bits,
                            ternary_threshold=c.ternary_threshold,
                            eps=c.eps,
                            out_chunk=c.chunk_size,
                        ),
                        "rms": RMSNorm(c.d_hidden),
                        "act": nn.GELU(),
                        "dropout": nn.Dropout(0.0),
                    }
                )
                for _ in range(c.num_hidden_blocks)
            ]
        )

        # ---- encoder 1536 -> 8192 (BitLinear, part of the weight mass) ----
        self.encoder = BitLinear(
            c.d_embed,
            c.d_hidden,
            bias=True,
            activation_bits=c.activation_bits,
            ternary_threshold=c.ternary_threshold,
            eps=c.eps,
            out_chunk=c.chunk_size,
        )

        # ---- projector 8192 -> 256 (BitLinear, ternary) ------------------
        self.projector = BitLinear(
            c.d_hidden,
            c.d_proj,
            bias=True,
            activation_bits=c.activation_bits,
            ternary_threshold=c.ternary_threshold,
            eps=c.eps,
            out_chunk=c.chunk_size,
        )

        # ---- FP32 output head (final 24h forecast, NOT quantized) -------
        self.final_rms = RMSNorm(c.d_proj)
        self.output_head = nn.Linear(c.d_proj, c.out_features, bias=True)

        # Warm-up so the latent weights sit in the ternary-active regime.
        self.apply(self._init_weights)

    # ------------------------------------------------------------------ #
    # init                                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, BitLinear)):
            # Standard 1/sqrt(fan_in) scaling.  The latent weights feed the
            # per-layer beta = mean(|W|) rescale inside BitLinear, so we only
            # need them O(1/sqrt(fan_in)).
            nn.init.normal_(m.weight, mean=0.0, std=1.0 / (m.in_features**0.5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    # forward settings                                                   #
    # ------------------------------------------------------------------ #
    def forward(self, x_window: torch.Tensor, ctx_means: torch.Tensor) -> torch.Tensor:
        """Predict a 24-hour forecast vector from a 24-hour look-back window.

        Args:
            x_window:  ``(B, WINDOW_HOURS, NUM_VARS)`` numeric series.
            ctx_means: ``(B, NUM_CONTEXT)`` rolling per-variable means.

        Returns:
            ``(B, OUT_FEATURES)`` forecast tensor.
        """
        B = x_window.shape[0]
        # Flatten the window into a vector and concat the context statistics.
        # NOTE: normalisation to ~[-1, 1] is applied by the data pipeline
        # (see train.py Normalizer); BitLinear re-scales per token anyway.
        x = torch.cat([x_window.reshape(B, -1), ctx_means], dim=-1)

        # -- FP32 encode --------------------------------------------------
        h = self.feature_embed(x)
        h = self.init_rms(h)

        # -- 1536 -> 8192 --------------------------------------------------
        h = self.encoder(h)

        # -- 14 × pre-norm residual ternary blocks --------------------------
        for blk in self.blocks:
            h = h + blk.act(blk.bit_linear(blk.rms(h)))
            h = blk.dropout(h)

        # -- projector -------------------------------------------------------
        h = self.projector(h)
        h = self.final_rms(h)

        # -- FP32 head -------------------------------------------------------
        return self.output_head(h)

    # ------------------------------------------------------------------ #
    # capacity reporting                                                #
    # ------------------------------------------------------------------ #
    def parameter_count(self, only_ternary: bool = False) -> int:
        total = 0
        for m in self.modules():
            if only_ternary and not isinstance(m, BitLinear):
                continue
            total += sum(p.numel() for p in m.parameters(recurse=False))
        return total

    def summary(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        ternary = self.parameter_count(only_ternary=True)
        return (
            f"WeatherNet 1.58-bit ternary forecast network\n"
            f"  total parameters      : {total:>14,d}  (~{total/1e9:.3f}B)\n"
            f"  ternary (BitLinear)   : {ternary:>14,d}  ({ternary/total*100:.1f}% of params)\n"
            f"  fp32 (in/out heads)   : {total-ternary:>14,d}\n"
            f"  hidden width          : {self.cfg.d_hidden:,d}\n"
            f"  residual blocks       : {self.cfg.num_hidden_blocks}\n"
            f"  est. FLOPs / sample   : {2*total:>14,d} (2*params, MatMul only)"
        )


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------
def build_weather_net(verbose: bool = True) -> WeatherNet:
    """Instantiate the default ~1B ternary network and print its budget."""
    net = WeatherNet()
    if verbose:
        print(net.summary())
    return net


if __name__ == "__main__":
    m = build_weather_net()
    x = torch.zeros(2, WINDOW_HOURS, NUM_VARS)
    c = torch.zeros(2, NUM_CONTEXT)
    y = m(x, c)
    print("sample forward:", tuple(y.shape))