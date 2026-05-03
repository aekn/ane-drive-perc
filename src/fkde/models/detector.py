"""End-to-end detector: backbone -> FPN -> FCOS head.

Training mode forward returns raw head outputs.
Eval mode forward runs head, caller chooses decode path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .fpn import FPN
from .head import FCOSHead
from .repvit import RepViT


@dataclass
class DetectorConfig:
    num_classes: int = 10
    fpn_channels: int = 256
    backbone_channels: tuple[int, int, int, int] = (48, 96, 192, 384)
    backbone_depths: tuple[int, int, int, int] = (2, 2, 14, 2)


class Detector(nn.Module):
    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = RepViT(channels=cfg.backbone_channels, depths=cfg.backbone_depths)
        self.fpn = FPN(in_channels=self.backbone.out_channels, out_channels=cfg.fpn_channels)
        self.head = FCOSHead(in_channels=cfg.fpn_channels, num_classes=cfg.num_classes)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        feats = self.backbone(x)
        feats = self.fpn(feats)
        return self.head(feats)

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        self.backbone.switch_to_deploy()
