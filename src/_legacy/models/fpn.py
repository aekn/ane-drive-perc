"""Feature Pyramid Network neck.

3-level FPN over (C3, C4, C5). Standard top-down construction:
    P5 = lateral_5(C5)
    P4 = lateral_4(C4) + upsample(P5)
    P3 = lateral_3(C3) + upsample(P4)
each followed by a 3x3 smoothing conv. All output channels share `out_channels`.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels: Sequence[int], out_channels: int = 256) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("FPN expects exactly 3 input feature maps (C3, C4, C5)")

        self.lateral_3 = nn.Conv2d(in_channels[0], out_channels, 1)
        self.lateral_4 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.lateral_5 = nn.Conv2d(in_channels[2], out_channels, 1)

        self.smooth_3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.out_channels = out_channels
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c3, c4, c5 = feats
        p5 = self.lateral_5(c5)
        p4 = self.lateral_4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral_3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        return self.smooth_3(p3), self.smooth_4(p4), self.smooth_5(p5)
