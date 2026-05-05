from typing import TypedDict

import torch
from torch import nn


class TinyDetectorOutput(TypedDict):
    class_logits: torch.Tensor
    box_cxcywh: torch.Tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyAnchorFreeDetector(nn.Module):
    def __init__(self, num_classes: int = 10, width: int = 64) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.stride = 8

        self.backbone = nn.Sequential(
            ConvBlock(3, width // 2, stride=2),
            ConvBlock(width // 2, width, stride=2),
            ConvBlock(width, width, stride=2),
            ConvBlock(width, width, stride=1),
            ConvBlock(width, width, stride=1),
        )

        self.class_head = nn.Sequential(
            ConvBlock(width, width, stride=1),
            nn.Conv2d(width, num_classes, kernel_size=1),
        )

        self.box_head = nn.Sequential(
            ConvBlock(width, width, stride=1),
            nn.Conv2d(width, 4, kernel_size=1),
        )

        self._init_heads()

    def _init_heads(self) -> None:
        final_class_conv = self.class_head[-1]
        if isinstance(final_class_conv, nn.Conv2d):
            nn.init.constant_(final_class_conv.bias, -4.595)

        final_box_conv = self.box_head[-1]
        if isinstance(final_box_conv, nn.Conv2d):
            nn.init.constant_(final_box_conv.bias, 0.0)

    def forward(self, images: torch.Tensor) -> TinyDetectorOutput:
        features = self.backbone(images)

        class_logits = self.class_head(features)
        box_cxcywh = torch.sigmoid(self.box_head(features))

        return {
            "class_logits": class_logits,
            "box_cxcywh": box_cxcywh,
        }
