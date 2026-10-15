"""A compact GPT-style decoder tuned to ~500M parameters (sized to run on
the same 8 GB GPU used for WeatherNet, in bf16 with an 8 GB budget)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

TIE_EMBEDDINGS = True


@dataclass
class GPTConfig:
    vocab_size: int = 16384
    n_layer: int = 24
    d_model: int = 1280
    n_head: int = 16
    ffn: int = 5120          # 4x d_model
    seq_len: int = 512
    drop: float = 0.0
    chunk: int | None = 128  # seq-chunk for matmuls (low-VRAM), None=off


def _chunked_linear(x, w, b, chunk):
    """Split the sequence axis and matmul each chunk to cap activation RAM."""
    if chunk is None or chunk >= x.size(1):
        return F.linear(x, w, b)
    return torch.cat([F.linear(x[:, i:i + chunk], w, b)
                      for i in range(0, x.size(1), chunk)], dim=1)


class Attn(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.c = c
        d, h = c.d_model, c.n_head
        self.h = h
        self.dh = d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        # causal mask (seq_len x seq_len), saved for reuse
        self.register_buffer("mask", torch.tril(
            torch.ones(c.seq_len, c.seq_len, dtype=torch.bool)).view(
            1, 1, c.seq_len, c.seq_len))

    def forward(self, x):
        c = self.c
        B, T, d = x.shape
        q, k, v = _chunked_linear(x, self.qkv.weight, None, c.chunk) \
            .chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=self.mask[:, :, :T, :T],
            is_causal=False)  # explicit tril mask, works on all torch builds
        y = y.transpose(1, 2).reshape(B, T, d)
        return _chunked_linear(y, self.proj.weight, None, c.chunk)


class Block(nn.Module):
    def __init__(self, c: GPTConfig, i: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d_model)
        self.attn = Attn(c)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.w1 = nn.Linear(c.d_model, c.ffn, bias=False)
        self.w2 = nn.Linear(c.ffn, c.d_model, bias=False)
        self.c = c

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + _chunked_linear(
            F.gelu(_chunked_linear(self.ln2(x), self.w1.weight, None,
                                   self.c.chunk)),
            self.w2.weight, None, self.c.chunk)
        return x


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab_size, c.d_model)
        self.pos = nn.Embedding(c.seq_len, c.d_model)
        self.blocks = nn.ModuleList([Block(c, i) for i in range(c.n_layer)])
        self.lnf = nn.LayerNorm(c.d_model)
        if not TIE_EMBEDDINGS:
            self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.register_buffer("positions", torch.arange(c.seq_len))
        n = sum(p.numel() for p in self.parameters())
        print(f"[gpt] GPT-{c.n_layer}x{c.d_model}  "
              f"parameters: {n:,} ({n/1e6:.1f}M)")

    def forward(self, idx, targets=None):
        c = self.c
        B, T = idx.shape
        x = self.tok(idx) + self.pos(self.positions[:T])
        for blk in self.blocks:
            x = blk(x)
        x = self.lnf(x)
        logits = _chunked_linear(x, self.tok.weight, None, c.chunk) \
            if TIE_EMBEDDINGS else self.head(x)
        loss = None
        if targets is not None:
            # causal LM: at position t predict token t+1 (shift-by-one)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, c.vocab_size),
                targets[:, 1:].reshape(-1))
        return logits, loss


def count_params(c: GPTConfig) -> int:
    m = GPT(c)
    return sum(p.numel() for p in m.parameters())