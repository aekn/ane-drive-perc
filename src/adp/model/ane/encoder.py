from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from adp.model.ane.blocks import (
    ANEFeedForward,
    ANELayerNorm,
    ANEMultiHeadAttention,
)


__all__ = ["ANEHybridEncoder"]


def _conv_bn_act(
    in_ch: int,
    out_ch: int,
    kernel: int = 3,
    stride: int = 1,
    groups: int = 1,
    act: bool = True,
) -> nn.Sequential:
    pad = (kernel - 1) // 2
    layers: list[nn.Module] = [
        nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(out_ch),
    ]
    if act:
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


class _CSPRepBlock(nn.Module):
    def __init__(self, channels: int, expansion: float = 0.5) -> None:
        super().__init__()
        hidden = max(int(channels * expansion), 16)
        self.cv1 = _conv_bn_act(channels, hidden, kernel=1)
        self.cv2 = _conv_bn_act(channels, hidden, kernel=1)
        self.bottleneck = nn.Sequential(
            _conv_bn_act(hidden, hidden, kernel=3, groups=hidden),
            _conv_bn_act(hidden, hidden, kernel=1),
        )
        self.cv3 = _conv_bn_act(2 * hidden, channels, kernel=1)

    def forward(self, x: Tensor) -> Tensor:
        a = self.bottleneck(self.cv1(x))
        b = self.cv2(x)
        return self.cv3(torch.cat([a, b], dim=1))


class _AIFIBlock(nn.Module):
    """Single transformer block on the deepest scale, ANE layout."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = ANELayerNorm(embed_dim)
        self.attn = ANEMultiHeadAttention(
            embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.norm2 = ANELayerNorm(embed_dim)
        self.ffn = ANEFeedForward(
            embed_dim,
            hidden_dim=int(embed_dim * ffn_ratio),
            dropout=dropout,
        )

    def forward(self, x_ane: Tensor, pos_ane: Tensor | None = None) -> Tensor:
        residual = x_ane
        h = self.norm1(x_ane)
        q = h if pos_ane is None else h + pos_ane
        k = h if pos_ane is None else h + pos_ane
        h = self.attn(q, k, h)
        x_ane = residual + h

        residual = x_ane
        h = self.norm2(x_ane)
        h = self.ffn(h)
        return residual + h


def build_2d_sincos_pos_embed(
    h: int,
    w: int,
    embed_dim: int,
    temperature: float = 10000.0,
    device: torch.device | None = None,
) -> Tensor:
    """2D sin-cos positional embedding, returned in ANE layout (1, C, 1, H*W)."""
    if embed_dim % 4 != 0:
        raise ValueError(
            f"embed_dim must be divisible by 4 for 2D sincos, got {embed_dim}"
        )
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing="ij",
    )
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim
    omega = 1.0 / (temperature**omega)

    out_x = grid_x.flatten()[:, None] * omega[None, :]
    out_y = grid_y.flatten()[:, None] * omega[None, :]
    pos = torch.cat(
        [out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()], dim=1
    )  # (H*W, embed_dim)
    return pos.t().unsqueeze(0).unsqueeze(2)  # (1, C, 1, H*W)


class ANEHybridEncoder(nn.Module):
    """ANE hybrid encoder: per-scale projection + AIFI + CCFM (FPN + PAN)."""

    def __init__(
        self,
        in_channels: Sequence[int],
        embed_dim: int = 192,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
        eval_spatial_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError(
                f"ANEHybridEncoder expects 3 input scales (strides 8/16/32), "
                f"got {len(in_channels)}"
            )
        self.in_channels = tuple(in_channels)
        self.embed_dim = embed_dim
        self.eval_spatial_size = (
            eval_spatial_size  # (H_input, W_input) for cached pos embed
        )

        # perscale 1x1 projection to embed_dim.
        self.input_proj = nn.ModuleList(
            [_conv_bn_act(c, embed_dim, kernel=1, act=False) for c in in_channels]
        )

        # AIFI on the deepest scale.
        self.aifi = _AIFIBlock(
            embed_dim, num_heads=num_heads, ffn_ratio=ffn_ratio, dropout=dropout
        )

        # CCFM top-down (FPN) and bottom-up (PAN).
        self.fpn_blocks = nn.ModuleList(
            [
                _CSPRepBlock(embed_dim),  # used after fusing 1/32 -> 1/16
                _CSPRepBlock(embed_dim),  # used after fusing 1/16 -> 1/8
            ]
        )
        self.fpn_lateral = nn.ModuleList(
            [
                _conv_bn_act(
                    embed_dim, embed_dim, kernel=1
                ),  # for the upsampled higher map
                _conv_bn_act(embed_dim, embed_dim, kernel=1),
            ]
        )
        self.pan_downsample = nn.ModuleList(
            [
                _conv_bn_act(embed_dim, embed_dim, kernel=3, stride=2),  # 1/8 -> 1/16
                _conv_bn_act(embed_dim, embed_dim, kernel=3, stride=2),  # 1/16 -> 1/32
            ]
        )
        self.pan_blocks = nn.ModuleList(
            [
                _CSPRepBlock(embed_dim),
                _CSPRepBlock(embed_dim),
            ]
        )

        # cached aifi pos emb.
        self._pos_cache_key: tuple[int, int] | None = None
        self.register_buffer("_pos_cache", torch.zeros(0), persistent=False)
        if eval_spatial_size is not None:
            H_in, W_in = int(eval_spatial_size[0]), int(eval_spatial_size[1])
            h_deep, w_deep = H_in // 32, W_in // 32
            pos = build_2d_sincos_pos_embed(
                h_deep, w_deep, embed_dim, device=torch.device("cpu")
            )
            self._pos_cache = pos
            self._pos_cache_key = (h_deep, w_deep)
            self._static_deep_h = h_deep
            self._static_deep_w = w_deep
        else:
            self._static_deep_h = None
            self._static_deep_w = None

    @property
    def out_channels(self) -> tuple[int, int, int]:
        return (self.embed_dim, self.embed_dim, self.embed_dim)

    def _get_pos_embed(self, h: int, w: int, device: torch.device) -> Tensor:
        key = (h, w)
        if self._pos_cache_key != key or self._pos_cache.numel() == 0:
            pos = build_2d_sincos_pos_embed(h, w, self.embed_dim, device=device)
            self._pos_cache = pos.to(device)
            self._pos_cache_key = key
        return self._pos_cache

    def forward(self, feats: Sequence[Tensor]) -> list[Tensor]:
        if len(feats) != 3:
            raise ValueError(f"expected 3 input feature scales, got {len(feats)}")

        proj = [self.input_proj[i](f) for i, f in enumerate(feats)]

        deepest = proj[-1]  # (B, C, H, W) at 1/32
        if self._static_deep_h is not None:
            h_deep, w_deep = self._static_deep_h, self._static_deep_w
            x_ane = deepest.flatten(2).unsqueeze(2)  # (B, C, 1, H*W)
            x_ane = self.aifi(x_ane, pos_ane=self._pos_cache)
            proj[-1] = x_ane.squeeze(2).unflatten(-1, (h_deep, w_deep))
        else:
            b, c, h, w = deepest.shape
            x_ane = deepest.reshape(b, c, 1, h * w).contiguous()
            pos = self._get_pos_embed(h, w, deepest.device)
            x_ane = self.aifi(x_ane, pos_ane=pos)
            proj[-1] = x_ane.reshape(b, c, h, w).contiguous()

        td_outs = [proj[-1]]
        for i in range(len(proj) - 1, 0, -1):
            higher = td_outs[-1]
            lower = proj[i - 1]
            up = F.interpolate(
                self.fpn_lateral[len(proj) - 1 - i](higher),
                size=lower.shape[-2:],
                mode="nearest",
            )
            fused = self.fpn_blocks[len(proj) - 1 - i](up + lower)
            td_outs.append(fused)
        td_outs = list(reversed(td_outs))  # [1/8, 1/16, 1/32]

        bu_outs = [td_outs[0]]
        for i in range(len(td_outs) - 1):
            down = self.pan_downsample[i](bu_outs[-1])
            fused = self.pan_blocks[i](down + td_outs[i + 1])
            bu_outs.append(fused)

        return bu_outs
