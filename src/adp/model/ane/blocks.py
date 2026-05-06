from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


__all__ = [
    "ANELayerNorm",
    "ANEMultiHeadAttention",
    "ANEFeedForward",
    "to_ane_layout",
    "from_ane_layout",
]


def to_ane_layout(x: Tensor) -> Tensor:
    if x.ndim == 3:
        # (B, S, C) -> (B, C, 1, S)
        return x.permute(0, 2, 1).unsqueeze(2).contiguous()
    if x.ndim == 4:
        b, c, h, w = x.shape
        return x.reshape(b, c, 1, h * w).contiguous()
    raise ValueError(
        f"to_ane_layout: unsupported rank {x.ndim} for shape {tuple(x.shape)}"
    )


def from_ane_layout(x: Tensor, *, spatial: tuple[int, int] | None = None) -> Tensor:
    if x.ndim != 4 or x.shape[2] != 1:
        raise ValueError(f"from_ane_layout expects (B,C,1,S), got {tuple(x.shape)}")
    b, c, _, s = x.shape
    if spatial is not None:
        h, w = spatial
        if h * w != s:
            raise ValueError(f"spatial {spatial} != S={s}")
        return x.reshape(b, c, h, w).contiguous()
    return x.squeeze(2).permute(0, 2, 1).contiguous()


class ANELayerNorm(nn.Module):
    def __init__(
        self, num_channels: int, eps: float = 1e-5, clip_mag: float | None = None
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.clip_mag = clip_mag
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        if self.clip_mag is not None:
            x = torch.clamp(x, -self.clip_mag, self.clip_mag)

        # Normalize over channel axis (dim=1) of (B, C, 1, S).
        mean = x.mean(dim=1, keepdim=True)
        x_zm = x - mean
        var = x_zm.pow(2).mean(dim=1, keepdim=True)
        x_norm = x_zm * torch.rsqrt(var + self.eps)
        return x_norm * self.weight + self.bias


class ANEMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=bias)
        self.k_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=bias)
        self.v_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=bias)
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=bias)
        self.dropout = dropout

    def forward(
        self,
        q: Tensor,
        k: Tensor | None = None,
        v: Tensor | None = None,
        *,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        if k is None:
            k = q
        if v is None:
            v = k

        B, C, _, Sq = q.shape
        Sk = k.shape[-1]
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(q)  # (B, C, 1, Sq)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Split heads: (B, C, 1, S) -> (B, H, D, S). Channel axis splits into heads
        # so each head's tensor stays contiguous along the S axis on ANE.
        q = q.view(B, H, D, Sq)
        k = k.view(B, H, D, Sk)
        v = v.view(B, H, D, Sk)

        # Attention scores: (B, H, Sq, Sk). Implemented as matmul (not einsum) to
        # ensure CoreML/ANE op conversion einsum apparently annoying as the dont
        # convert easy
        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale

        if key_padding_mask is not None:
            mask = key_padding_mask.view(B, 1, 1, Sk)
            attn = attn.masked_fill(mask, -1e4)

        attn = torch.softmax(attn, dim=-1)
        if self.dropout > 0.0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        # agg values v (B, H, D, Sk) @ attn^T (B, H, Sk, Sq) -> (B, H, D, Sq).
        out = torch.matmul(v, attn.transpose(-2, -1))

        # merge heads back to (B, C, 1, Sq).
        out = out.reshape(B, C, 1, Sq)
        return self.out_proj(out)


class ANEFeedForward(nn.Module):
    """Conv1x1 -> GELU -> Conv1x1"""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 4
        self.fc1 = nn.Conv2d(embed_dim, hidden_dim, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden_dim, embed_dim, kernel_size=1)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        if self.dropout > 0.0 and self.training:
            x = F.dropout(x, p=self.dropout)
        x = self.fc2(x)
        if self.dropout > 0.0 and self.training:
            x = F.dropout(x, p=self.dropout)
        return x
