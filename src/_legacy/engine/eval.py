"""Validation and mAP evaluation."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from .decode import decode_with_nms


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    image_size: int,
    score_threshold: float = 0.05,
    nms_iou: float = 0.6,
    pre_nms_topk: int = 1000,
    post_nms_topk: int = 100,
    amp: bool = True,
) -> dict[str, float]:
    """Returns {'map': ..., 'map_50': ..., 'map_75': ...}."""
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    with torch.inference_mode():
        for images, boxes, labels, _names in val_loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                cls_logits, bbox_preds, ctr = model(images)

            preds = decode_with_nms(
                [c.float() for c in cls_logits],
                [b.float() for b in bbox_preds],
                [ct.float() for ct in ctr],
                image_size=(image_size, image_size),
                pre_nms_topk=pre_nms_topk,
                post_nms_topk=post_nms_topk,
                score_threshold=score_threshold,
                nms_iou=nms_iou,
            )

        preds_cpu = [
            {
                "boxes": p["boxes"].cpu(),
                "scores": p["scores"].cpu(),
                "labels": p["labels"].cpu(),
            }
            for p in preds
        ]
        targets_cpu = [
            {"boxes": b.cpu().float(), "labels": l.cpu().long()}
            for b, l in zip(boxes, labels)
        ]
        metric.update(preds_cpu, targets_cpu)

    out = metric.compute()
    return {
        "map": float(out["map"].item()),
        "map_50": float(out["map_50"].item()),
        "map_75": float(out["map_75"].item()),
    }
