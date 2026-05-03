from __future__ import annotations

from functools import lru_cache

import torch


FPN_STRIDES: tuple[int, ...] = (8, 16, 32)


@lru_cache(maxsize=64)
def _grid_cached(h: int, w: int, stride: int, device_str: str) -> torch.Tensor:
    device = torch.device(device_str)
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # (H*W, 2)


def points_for_feature_map(
    h: int, w: int, stride: int, device: torch.device
) -> torch.Tensor:
    """Return (H*W, 2) tensor of (x, y) point centers in input coords."""
    return _grid_cached(h, w, stride, str(device))


def all_level_points(
    feature_shapes: list[tuple[int, int]],
    strides: tuple[int, ...],
    device: torch.device,
) -> list[torch.Tensor]:
    if len(feature_shapes) != len(strides):
        raise ValueError("feature_shapes and strides must align")
    return [
        points_for_feature_map(h, w, s, device)
        for (h, w), s in zip(feature_shapes, strides, strict=True)
    ]
