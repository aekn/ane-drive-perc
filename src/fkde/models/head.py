"""FCOS detection head with dual-assignment training.

A single shared head produces three predictions per pixel per FPN level:
cls_logits, reg, ctr.
Predictions are scored against two assignment rules at training: one-to-many 
and one-to-one, weighted by ProgLoss. At deploy, only one-to-one branch is read.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


class Scale(nn.Module):
    """Per-level learnable scalar applied to regression outputs."""

    def __init__(self, init_value: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class FCOSHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_levels: int = 3,
        num_tower_convs: int = 4,
        prior_prob: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_levels = num_levels

        cls_tower: list[nn.Module] = []
        reg_tower: list[nn.Module] = []
        for _ in range(num_tower_convs):
            cls_tower += [
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.GroupNorm(8, in_channels),
                nn.ReLU(inplace=True),
            ]
            reg_tower += [
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.GroupNorm(8, in_channels),
                nn.ReLU(inplace=True),
            ]
        self.cls_tower = nn.Sequential(*cls_tower)
        self.reg_tower = nn.Sequential(*reg_tower)

        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)

        # One per FPN level so each level can scale regression independently.
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(num_levels)])

        self._init_weights(prior_prob)

    def _init_weights(self, prior_prob: float) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Focal loss prior on the cls bias: log((1-p)/p).
        bias_init = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_init)

    def forward(
        self, feats: Sequence[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        if len(feats) != self.num_levels:
            raise ValueError(
                f"expected {self.num_levels} feature levels, got {len(feats)}"
            )

        logits, bboxes, centernesses = [], [], []
        for i, f in enumerate(feats):
            cls_f = self.cls_tower(f)
            reg_f = self.reg_tower(f)
            logits.append(self.cls_logits(cls_f))
            # Regression strictly positive (l/t/r/b are non-negative distances).
            raw = self.bbox_pred(reg_f)
            bboxes.append(torch.exp(self.scales[i](raw).clamp(max=6.0)))
            centernesses.append(self.centerness(reg_f))
        return logits, bboxes, centernesses
