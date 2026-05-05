from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from ane_drive_perc.data.geometry import (
    ResizeTransform,
    clip_boxes_xyxy,
    resize_boxes_xyxy,
)


@dataclass(frozen=True)
class DetectionResizeConfig:
    height: int
    width: int
    preserve_aspect_ratio: bool = True
    letterbox_value: float = 114.0 / 255.0


class ResizeDetectionSample:
    def __init__(self, config: DetectionResizeConfig) -> None:
        self.config = config

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        image = sample["image"]
        target = dict(sample["target"])

        if not isinstance(image, torch.Tensor):
            raise TypeError("Expected sample['image'] to be a torch.Tensor.")
        if image.ndim != 3:
            raise ValueError(f"Expected image shape CxHxW, got {tuple(image.shape)}.")

        resized_image, transform = (
            self._letterbox(image)
            if self.config.preserve_aspect_ratio
            else self._direct_resize(image)
        )

        resized_boxes = resize_boxes_xyxy(target["boxes"], transform)
        target["boxes"] = clip_boxes_xyxy(
            resized_boxes,
            height=self.config.height,
            width=self.config.width,
        )
        target["input_size"] = (self.config.height, self.config.width)
        target["resize_transform"] = transform
        return {"image": resized_image, "target": target}

    def _direct_resize(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, ResizeTransform]:
        _, orig_height, orig_width = image.shape
        resized = F.interpolate(
            image.unsqueeze(0),
            size=(self.config.height, self.config.width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        transform = ResizeTransform(
            orig_height=orig_height,
            orig_width=orig_width,
            new_height=self.config.height,
            new_width=self.config.width,
            scale_y=self.config.height / orig_height,
            scale_x=self.config.width / orig_width,
        )
        return resized, transform

    def _letterbox(self, image: torch.Tensor) -> tuple[torch.Tensor, ResizeTransform]:
        channels, orig_height, orig_width = image.shape
        scale = min(self.config.width / orig_width, self.config.height / orig_height)
        resized_width = int(round(orig_width * scale))
        resized_height = int(round(orig_height * scale))

        resized = F.interpolate(
            image.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        canvas = image.new_full(
            (channels, self.config.height, self.config.width),
            fill_value=self.config.letterbox_value,
        )
        pad_top = float((self.config.height - resized_height) // 2)
        pad_left = float((self.config.width - resized_width) // 2)
        top = int(pad_top)
        left = int(pad_left)
        canvas[:, top : top + resized_height, left : left + resized_width] = resized

        transform = ResizeTransform(
            orig_height=orig_height,
            orig_width=orig_width,
            new_height=self.config.height,
            new_width=self.config.width,
            scale_y=scale,
            scale_x=scale,
            pad_top=pad_top,
            pad_left=pad_left,
        )
        return canvas, transform
