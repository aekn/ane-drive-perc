"""Inference-time decoding from FCOS head outputs.

decode_topk_no_nms: Deploy path. Reads top-k highest-scoring points.
decode_with_nms: Legacy path using class*center scoring and NMS.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torchvision.ops import batched_nms

from .anchors import all_level_points, FPN_STRIDES


def _flatten(tensors: Sequence[torch.Tensor], channels: int) -> torch.Tensor:
    out = []
    for t in tensors:
        out.append(t.permute(0, 2, 3, 1).reshape(t.shape[0], -1, channels))
    return torch.cat(out, dim=1)


def _decode_xyxy(points: torch.Tensor, ltrb: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            points[..., 0] - ltrb[..., 0],
            points[..., 1] - ltrb[..., 1],
            points[..., 0] + ltrb[..., 2],
            points[..., 1] + ltrb[..., 3],
        ],
        dim=-1,
    )


def _gather_points(feats: list[torch.Tensor], strides: tuple[int, ...]) -> torch.Tensor:
    shapes = [(t.shape[-2], t.shape[-1]) for t in feats]
    pts = all_level_points(shapes, strides, feats[0].device)
    return torch.cat(pts, dim=0)


@torch.no_grad()
def decode_topk_no_nms(
    cls_logits: list[torch.Tensor],  # per-level (B, C, H, W)
    bbox_preds: list[torch.Tensor],
    image_size: tuple[int, int],  # (H, W) input image size
    strides: tuple[int, ...] = FPN_STRIDES,
    top_k: int = 100,
    score_threshold: float = 0.05,
) -> list[dict[str, torch.Tensor]]:
    """Returns one dict per image with keys 'boxes', 'scores', 'labels'."""
    B, C = cls_logits[0].shape[0], cls_logits[0].shape[1]
    H_img, W_img = image_size

    flat_cls = _flatten(cls_logits, C)  # (B, P, C)
    flat_reg = _flatten(bbox_preds, 4)
    flat_pts = _gather_points(cls_logits, strides)  # (P, 2)

    scores_all = flat_cls.sigmoid()  # (B, P, C)

    results: list[dict[str, torch.Tensor]] = []
    for b in range(B):
        scores_b = scores_all[b]  # (P, C)
        max_scores, labels = scores_b.max(dim=1)  # (P,)
        keep = max_scores > score_threshold
        if keep.sum() == 0:
            results.append(
                {
                    "boxes": torch.zeros((0, 4), device=flat_cls.device),
                    "scores": torch.zeros((0,), device=flat_cls.device),
                    "labels": torch.zeros(
                        (0,), dtype=torch.long, device=flat_cls.device
                    ),
                }
            )
            continue

        pts_keep = flat_pts[keep]
        reg_keep = flat_reg[b][keep]
        boxes = _decode_xyxy(pts_keep, reg_keep)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W_img)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H_img)
        scores = max_scores[keep]
        labs = labels[keep]

        if scores.numel() > top_k:
            topv, topi = scores.topk(top_k)
            boxes, scores, labs = boxes[topi], topv, labs[topi]

        results.append({"boxes": boxes, "scores": scores, "labels": labs})
    return results


@torch.no_grad()
def decode_with_nms(
    cls_logits: list[torch.Tensor],
    bbox_preds: list[torch.Tensor],
    centernesses: list[torch.Tensor],
    image_size: tuple[int, int],
    strides: tuple[int, ...] = FPN_STRIDES,
    pre_nms_topk: int = 1000,
    post_nms_topk: int = 100,
    score_threshold: float = 0.05,
    nms_iou: float = 0.6,
) -> list[dict[str, torch.Tensor]]:
    """Score = sigmoid(cls) * sigmoid(ctr); class-aware (batched) NMS."""
    B, C = cls_logits[0].shape[0], cls_logits[0].shape[1]
    H_img, W_img = image_size

    flat_cls = _flatten(cls_logits, C)
    flat_reg = _flatten(bbox_preds, 4)
    flat_ctr = _flatten(centernesses, 1).squeeze(-1)
    flat_pts = _gather_points(cls_logits, strides)

    scores_all = flat_cls.sigmoid() * flat_ctr.sigmoid().unsqueeze(-1)

    results: list[dict[str, torch.Tensor]] = []
    for b in range(B):
        scores_b = scores_all[b]  # (P, C)
        max_scores, labels = scores_b.max(dim=1)
        keep = max_scores > score_threshold
        if keep.sum() == 0:
            results.append(
                {
                    "boxes": torch.zeros((0, 4), device=flat_cls.device),
                    "scores": torch.zeros((0,), device=flat_cls.device),
                    "labels": torch.zeros(
                        (0,), dtype=torch.long, device=flat_cls.device
                    ),
                }
            )
            continue

        keep_indices = keep.nonzero(as_tuple=True)[0]
        if keep_indices.numel() > pre_nms_topk:
            # top-k within the threshold-filtered set
            # doesn't include points below score threshold
            topv, topi_local = max_scores[keep_indices].topk(pre_nms_topk)
            keep_idx = keep_indices[topi_local]
        else:
            keep_idx = keep_indices

        pts_keep = flat_pts[keep_idx]
        reg_keep = flat_reg[b][keep_idx]
        boxes = _decode_xyxy(pts_keep, reg_keep)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W_img)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H_img)
        scores = max_scores[keep_idx]
        labs = labels[keep_idx]

        nms_keep = batched_nms(boxes, scores, labs, nms_iou)[:post_nms_topk]
        results.append(
            {
                "boxes": boxes[nms_keep],
                "scores": scores[nms_keep],
                "labels": labs[nms_keep],
            }
        )
    return results
