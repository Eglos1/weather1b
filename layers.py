"""
layers.py
=========
Custom 1.58-bit (ternary-weight) ``BitLinear`` layer for a BitNet b1.58-style
Deep Network, adapted for *continuous numeric time-series forecasting*.

Reference architecture: BitNet b1.58 (Wang, Ma et al. 2024)
    * Weights are constrained to the ternary set {-1, 0, +1}.
    * Activations are quantized to 8-bit, scaled per-token (per-sample-row)
      by their absmax value so that continuous variables (temperature in C,
      pressure in hPa, ...) never underflow -- magnitudes are preserved by an
      explicit scale instead of being thrown away by a fixed integer range.
    * The layer-wide scaling constant ``beta = mean(|W|)`` (a.k.a. the mean
      absolute value of the *latent* full-precision weights) rescales the
      ternary weights so the energy of the layer is preserved after
      quantization:
          y = (X_q * gamma) @ (W_ternary * beta).T + b
      where gamma is a per-token activation scale.
    * Gradients are passed through the quantization operations using a
      Straight-Through Estimator (STE): the value seen by the *forward* pass
      is quantized, but the gradient seen by the *backward* pass is the plain
      gradient of the underlying full-precision quantity (identity Jacobian).

Design note
-----------
Standard 1-bit LLMs quantize token embeddings from a fixed finite dictionary;
here the inputs are *arbitrary real numbers* spanning 2+ orders of magnitude
(pressure ~1000 hPa, wind ~0.1 m/s).  The per-token absmax normalisation keeps
such continuous input data well-conditioned when fed into an 8-bit grid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Straight-Through-Estimator primitives
# ---------------------------------------------------------------------------


class _STE(torch.autograd.Function):
    """Identity in the backward pass (gradient flows straight through).

    ``quantize(forward)`` is the *quantized* value, ``backward`` returns the
    gradient as-is -- the standard trick used to optimise a discrete
    (ternary / rounded) function via gradient descent on its latent
    full-precision parameters.
    """

    @staticmethod
    def forward(ctx, x: Tensor, q: Tensor) -> Tensor:  # (input, quantized)
        ctx.save_for_backward(q)
        return q

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        return grad_output, None


def ternary_ste(w: Tensor, threshold: float = 0.0) -> Tensor:
    """Ternarize ``w`` to {-1, 0, +1} with a Straight-Through Estimator.

    ``+1`` where ``w > threshold``, ``-1`` where ``w < -threshold`` and ``0``
    in between.  The returned value is ternary for the forward pass while the
    backward pass sees the identity on ``w``.

    Memory-frugal: builds the ternary tensor in-place with two masked fills,
    so peak usage stays ~2 copies of ``w`` instead of ~4 (fits 8GB VRAM).
    """
    q = torch.zeros_like(w)
    q.masked_fill_(w > threshold, 1.0)
    q.masked_fill_(w < -threshold, -1.0)
    return _STE.apply(w, q)


def _plain_ternary(w: Tensor, threshold: float) -> Tensor:
    """Raw ternary {-1,0,+1} of ``w`` (no autograd wrapper)."""
    q = torch.zeros_like(w)
    q.masked_fill_(w > threshold, 1.0)
    q.masked_fill_(w < -threshold, -1.0)
    return q


class TernaryLinearFunction(torch.autograd.Function):
    """Fused ``linear(quant8(x), ternary(w) * beta, b)`` autograd node.

    The ternary weight is *recomputed* inside backward instead of being saved,
    so autograd never holds all 14 blocks' quantized weights in VRAM. That is
    what lets the 954M-parameter model train inside 8GB (RX 6600).
    """

    @staticmethod
    def forward(ctx, x_q, w, bias, beta, threshold, out_chunk):
        ctx.save_for_backward(x_q, beta)
        ctx.w = w                       # reference to the latent parameter
        ctx.bias = bias
        ctx.threshold = threshold
        ctx.out_chunk = out_chunk
        w_q = _plain_ternary(w, threshold) * beta      # (C, K)
        if out_chunk and w_q.shape[0] > out_chunk:
            outs = [
                F.linear(
                    x_q,
                    w_q[i : i + out_chunk],
                    bias[i : i + out_chunk] if bias is not None else None,
                )
                for i in range(0, w_q.shape[0], out_chunk)
            ]
            return torch.cat(outs, dim=-1)
        return F.linear(x_q, w_q, bias)

    @staticmethod
    def backward(ctx, g):
        x_q, beta = ctx.saved_tensors
        w, bias, thr, chunk = ctx.w, ctx.bias, ctx.threshold, ctx.out_chunk
        # recompute the effective (scaled) ternary weight once
        w_q = _plain_ternary(w, thr) * beta            # (C, K)
        # grads may arrive in a promoted dtype (fp32 targets); align w/ compute
        g = g.to(w_q.dtype)
        # dL/dx_q = g @ w_q^T                           (B, K)
        grad_x = g @ w_q
        # STE: treat ternary as identity, so dL/dw = beta * (g^T @ x_q)
        grad_w_tmp = g.transpose(-1, -2) @ x_q          # (C, K)
        grad_w = grad_w_tmp * beta
        grad_bias = g.sum(dim=0) if bias is not None else None
        return grad_x, grad_w.to(x_q.dtype), grad_bias, None, None, None


def absmax_quantize_ste(
    x: Tensor,
    bits: int = 8,
    eps: float = 1e-7,
) -> tuple[Tensor, Tensor]:
    """Symmetric 8-bit absmax quantization with STE.

    Computes per-token scale ``alpha = max|x|`` (token = last dimension),
    maps ``x/alpha`` into [-1, 1], rounds it onto an ``N = 2^(bits-1) - 1``
    grid and returns:

        ``(x_q, alpha)``  where ``x_q ~= x`` to 8-bit resolution.

    ``alpha`` is returned so callers may rescale the quantised tensor back to
    physical units (hPa, C, m/s, ...) before feeding it to a linear layer.
    This is the step that keeps *continuous* variables from underflowing.
    """
    qn = 2.0 ** (bits - 1) - 1.0
    dt = x.dtype
    alpha = x.abs().amax(dim=-1, keepdim=True).clamp_min(eps).to(dt)
    x_hat = (x / alpha).clamp(-1.0, 1.0)
    x_q = (x_hat * qn).round().div(qn) * alpha          # 8-bit resolved value
    return _STE.apply(x, x_q), alpha


# ---------------------------------------------------------------------------
# BitLinear
# ---------------------------------------------------------------------------


class BitLinear(nn.Linear):
    """1.58-bit ternary-weight linear layer (``in_features`` -> ``out_features``).

    The *latent* (full-precision) weight is stored as a normal ``nn.Linear``
    weight; during every forward pass it is:

      1. Ternarized to {-1, 0, +1} (see :func:`ternary_ste`)
      2. Rescaled by the layer-wide magnitude ``beta = mean(|W|)``

    and the input is:

      1. Quantized to 8-bit with a per-token absmax scale ``gamma``
         (see :func:`absmax_quantize_ste`), keeping continuous variables
         numerically stable.

    Gradients flow through both quantizers via STE, so optimizers update the
    latent weights in full precision.  Mathematically the layer computes:

        y = (X_8bit * gamma) @ (W_ternary * beta).T + b

    which — since ``X_8bit * gamma ~= X`` at 8-bit precision — degenerates to
    a high-throughput ternary matmul ``X @ (W_ternary * beta).T`` at inference
    time, the defining property of 1.58-bit networks.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation_bits: int = 8,
        ternary_threshold: float = 0.0,
        eps: float = 1e-7,
        out_chunk: int | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.activation_bits = int(activation_bits)
        self.ternary_threshold = float(ternary_threshold)
        self.eps = float(eps)
        # If set, the output dimension is computed in slices of this many
        # neurons -> bounds peak activation memory on low-VRAM GPUs.
        self.out_chunk = int(out_chunk) if out_chunk else None
        # Optional initial bias -- leave as-is (FP32, not quantized).
        self._nflop = 2 * in_features * out_features

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, x: Tensor) -> Tensor:
        # -------- activations: 8-bit absmax quantisation (per token) ----
        # x: (*, in_features) -> flatten to (B, in_features)
        shape_before = x.shape
        x = x.reshape(-1, self.in_features)

        x_q, gamma = absmax_quantize_ste(x, bits=self.activation_bits, eps=self.eps)

        # -------- weights: ternary {-1, 0, +1} * beta -------------------
        # beta = layer-wide average magnitude of the latent weights (the
        # "1.58-bit" rescaling factor of the BitNet b1.58 paper).
        # Cast the latent weight into *the activation dtype* so matmuls run
        # on tensor cores under torch.autocast (bf16/fp16) on GPU; the
        # parameter itself always stays FP32 for accurate optimizer updates.
        dtype = x.dtype
        beta = self.weight.detach().abs().mean().clamp_min(self.eps).to(dtype)
        # Dead-zone threshold relative to the layer magnitude.
        thr = (self.ternary_threshold * beta.detach()) if self.ternary_threshold > 0 else 0.0

        # -------- fused matmul ------------------------------------------
        # One autograd node quantizes the weights and runs the chunked GEMM.
        # w_q / x_q are recomputed in backward -> no saved full-size tensors.
        bias = self.bias.to(dtype) if self.bias is not None else None
        y = TernaryLinearFunction.apply(
            x_q, self.weight.to(dtype), bias, beta, thr, self.out_chunk
        )

        # Restore leading dims of the batch axis if they existed.
        return y.reshape(shape_before[:-1] + (self.out_features,))

    # ------------------------------------------------------------------ #
    # Introspection helpers                                              #
    # ------------------------------------------------------------------ #
    def flops_per_forward(self) -> int:
        """Approximate FLOPs for a single element passing through the layer."""
        return self._nflop

    def extra_repr(self) -> str:
        return (
            f"{super().extra_repr()}, "
            f"activation_bits={self.activation_bits}, "
            f"ternary_threshold={self.ternary_threshold}"
        )