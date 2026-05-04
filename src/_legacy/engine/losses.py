"""Detection losses with consistent num_gt normalization.

Loss terms normalized by total GT boxes in batch. Puts branches on same scale so ProgLoss schedule controls balance.
Regression loss computed as single global weighted sum over matched positives.
Centerness supervised by BCE so head receives gradient as ProgLoss shifts weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss

from .anchors import FPN_STRIDES
from .matcher import hungarian_match_o2o
from .targets import assign_targets_o2m


def progloss_weights(epoch: int, max_epochs: int) -> tuple[float, float]:
    """Linear ramp: w_o2m goes 1.0 -> 0.5; w_o2o goes 0.5 -> 1.0."""
    if max_epochs <= 0:
        return 0.5, 1.0
    t = min(max(epoch / max_epochs, 0.0), 1.0)
    return 1.0 - 0.5 * t, 0.5 + 0.5 * t


def sigmoid_focal_loss(
    logits: torch.Tensor,  # (N, C)
    targets: torch.Tensor,  # (N,) class index in [0, C]; C == background
    num_classes: int,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Sum-reduced sigmoid focal loss. Background = `num_classes` index."""
    one_hot = torch.zeros_like(logits)
    pos = targets < num_classes
    one_hot[pos, targets[pos]] = 1.0

    p = logits.sigmoid()
    pt = p * one_hot + (1 - p) * (1 - one_hot)
    alpha_t = alpha * one_hot + (1 - alpha) * (1 - one_hot)
    bce = F.binary_cross_entropy_with_logits(logits, one_hot, reduction="none")
    return (alpha_t * (1 - pt).pow(gamma) * bce).sum()


def area_weights(
    boxes_xyxy: torch.Tensor,
    a_ref: float = 1024.0,
    beta: float = 0.5,
    w_max: float = 4.0,
) -> torch.Tensor:
    """Per-box weight emphasizing small objects. Disable by setting beta=0."""
    if beta <= 0.0:
        return torch.ones(boxes_xyxy.shape[0], device=boxes_xyxy.device)
    area = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=0) * (
        boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    ).clamp(min=0)
    w = (a_ref / area.clamp(min=1.0)) ** beta
    return w.clamp(min=1.0, max=w_max)


def _flatten_levels(tensors: Sequence[torch.Tensor], channels: int) -> torch.Tensor:
    """List of (B, C, H_l, W_l) -> (B, sum(H_l*W_l), C)."""
    out = []
    for t in tensors:
        b = t.shape[0]
        out.append(t.permute(0, 2, 3, 1).reshape(b, -1, channels))
    return torch.cat(out, dim=1)


def _decode_pred_boxes(points: torch.Tensor, ltrb: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            points[..., 0] - ltrb[..., 0],
            points[..., 1] - ltrb[..., 1],
            points[..., 0] + ltrb[..., 2],
            points[..., 1] + ltrb[..., 3],
        ],
        dim=-1,
    )


def _centerness_target(
    boxes_xyxy: torch.Tensor, points_xy: torch.Tensor
) -> torch.Tensor:
    l = points_xy[:, 0] - boxes_xyxy[:, 0]
    t = points_xy[:, 1] - boxes_xyxy[:, 1]
    r = boxes_xyxy[:, 2] - points_xy[:, 0]
    b = boxes_xyxy[:, 3] - points_xy[:, 1]
    lr_min = torch.minimum(l, r).clamp(min=0)
    lr_max = torch.maximum(l, r).clamp(min=1e-8)
    tb_min = torch.minimum(t, b).clamp(min=0)
    tb_max = torch.maximum(t, b).clamp(min=1e-8)
    return torch.sqrt((lr_min / lr_max) * (tb_min / tb_max))


@dataclass
class LossCfg:
    num_classes: int
    center_sampling_radius: float = 1.5
    size_bands: tuple[tuple[float, float], ...] = ((-1, 64), (64, 128), (128, 1e8))
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    area_a_ref: float = 1024.0
    area_beta: float = 0.5
    area_w_max: float = 4.0
    lambda_cls_match: float = 1.0
    lambda_box_match: float = 2.5
    strides: tuple[int, ...] = FPN_STRIDES


@dataclass
class LossOutput:
    total: torch.Tensor
    o2m_cls: torch.Tensor
    o2m_reg: torch.Tensor
    o2m_ctr: torch.Tensor
    o2o_cls: torch.Tensor
    o2o_reg: torch.Tensor
    o2o_ctr: torch.Tensor
    num_pos_o2m: int
    num_pos_o2o: int
    num_gt: int


def detection_loss(
    cls_logits: list[torch.Tensor],  # per-level (B, C, H, W)
    bbox_preds: list[torch.Tensor],  # per-level (B, 4, H, W)
    centernesses: list[torch.Tensor],  # per-level (B, 1, H, W)
    points_per_level: list[torch.Tensor],
    gt_boxes_per_image: list[torch.Tensor],
    gt_labels_per_image: list[torch.Tensor],
    cfg: LossCfg,
    progloss: tuple[float, float],
) -> LossOutput:
    device = cls_logits[0].device
    B = cls_logits[0].shape[0]
    C = cfg.num_classes

    num_gt = sum(int(gb.shape[0]) for gb in gt_boxes_per_image)
    denom = max(num_gt, 1)

    flat_cls = _flatten_levels(cls_logits, C)  # (B, P, C)
    flat_reg = _flatten_levels(bbox_preds, 4)  # (B, P, 4)
    flat_ctr = _flatten_levels(centernesses, 1).squeeze(-1)  # (B, P)
    flat_points = torch.cat(points_per_level, dim=0)  # (P, 2)

    # o2m target
    cls_t, reg_t, ctr_t, pos_mask = assign_targets_o2m(
        gt_boxes_per_image,
        gt_labels_per_image,
        points_per_level,
        cfg.strides,
        cfg.num_classes,
        size_bands=cfg.size_bands,
        center_sampling_radius=cfg.center_sampling_radius,
    )
    num_pos_o2m = int(pos_mask.sum().item())

    # o2m loss
    o2m_cls = (
        sigmoid_focal_loss(
            flat_cls.reshape(-1, C),
            cls_t.reshape(-1),
            num_classes=C,
            alpha=cfg.focal_alpha,
            gamma=cfg.focal_gamma,
        )
        / denom
    )

    if num_pos_o2m > 0:
        pred_boxes = _decode_pred_boxes(
            flat_points.unsqueeze(0).expand(B, -1, -1), flat_reg
        )  # (B, P, 4)
        gt_boxes_decoded = _decode_pred_boxes(
            flat_points.unsqueeze(0).expand(B, -1, -1), reg_t
        )
        pred_pos = pred_boxes[pos_mask]
        gt_pos = gt_boxes_decoded[pos_mask]
        ctr_pred_pos = flat_ctr[pos_mask]
        ctr_t_pos = ctr_t[pos_mask]

        a_w = area_weights(gt_pos, cfg.area_a_ref, cfg.area_beta, cfg.area_w_max)
        giou = generalized_box_iou_loss(pred_pos, gt_pos, reduction="none")
        o2m_reg = (giou * a_w).sum() / denom
        o2m_ctr = (
            F.binary_cross_entropy_with_logits(ctr_pred_pos, ctr_t_pos, reduction="sum")
            / denom
        )
    else:
        o2m_reg = torch.zeros((), device=device)
        o2m_ctr = torch.zeros((), device=device)

    # o2o
    o2o_cls_sum = torch.zeros((), device=device)
    o2o_reg_giou_terms: list[torch.Tensor] = []
    o2o_reg_w_terms: list[torch.Tensor] = []
    o2o_ctr_logit_terms: list[torch.Tensor] = []
    o2o_ctr_target_terms: list[torch.Tensor] = []
    num_pos_o2o = 0

    for b in range(B):
        gt_b = gt_boxes_per_image[b].to(device)
        lab_b = gt_labels_per_image[b].to(device)
        if gt_b.numel() == 0:
            cls_logits_b = flat_cls[b]
            bg = torch.full(
                (cls_logits_b.shape[0],), C, dtype=torch.long, device=device
            )
            o2o_cls_sum = o2o_cls_sum + sigmoid_focal_loss(
                cls_logits_b, bg, C, cfg.focal_alpha, cfg.focal_gamma
            )
            continue

        pt_idx, gt_idx = hungarian_match_o2o(
            flat_cls[b],
            flat_reg[b],
            flat_points,
            gt_b,
            lab_b,
            lambda_cls=cfg.lambda_cls_match,
            lambda_box=cfg.lambda_box_match,
            focal_alpha=cfg.focal_alpha,
            focal_gamma=cfg.focal_gamma,
        )
        num_pos_o2o += pt_idx.numel()

        cls_t_b = torch.full((flat_cls.shape[1],), C, dtype=torch.long, device=device)
        cls_t_b[pt_idx] = lab_b[gt_idx]
        o2o_cls_sum = o2o_cls_sum + sigmoid_focal_loss(
            flat_cls[b], cls_t_b, C, cfg.focal_alpha, cfg.focal_gamma
        )

        pred_xyxy_b = _decode_pred_boxes(flat_points, flat_reg[b])  # (P, 4)
        pred_match = pred_xyxy_b[pt_idx]
        gt_match = gt_b[gt_idx]
        a_w = area_weights(gt_match, cfg.area_a_ref, cfg.area_beta, cfg.area_w_max)
        giou = generalized_box_iou_loss(pred_match, gt_match, reduction="none")
        o2o_reg_giou_terms.append(giou)
        o2o_reg_w_terms.append(a_w)

        ctr_pred_match = flat_ctr[b][pt_idx]
        ctr_t_match = _centerness_target(gt_match, flat_points[pt_idx])
        o2o_ctr_logit_terms.append(ctr_pred_match)
        o2o_ctr_target_terms.append(ctr_t_match)

    o2o_cls = o2o_cls_sum / denom

    if o2o_reg_giou_terms:
        all_giou = torch.cat(o2o_reg_giou_terms)
        all_aw = torch.cat(o2o_reg_w_terms)
        o2o_reg = (all_giou * all_aw).sum() / denom
        all_ctr_logits = torch.cat(o2o_ctr_logit_terms)
        all_ctr_targets = torch.cat(o2o_ctr_target_terms)
        o2o_ctr = (
            F.binary_cross_entropy_with_logits(
                all_ctr_logits, all_ctr_targets, reduction="sum"
            )
            / denom
        )
    else:
        o2o_reg = torch.zeros((), device=device)
        o2o_ctr = torch.zeros((), device=device)

    w_o2m, w_o2o = progloss
    total = w_o2m * (o2m_cls + o2m_reg + o2m_ctr) + w_o2o * (
        o2o_cls + o2o_reg + o2o_ctr
    )

    return LossOutput(
        total=total,
        o2m_cls=o2m_cls.detach(),
        o2m_reg=o2m_reg.detach(),
        o2m_ctr=o2m_ctr.detach(),
        o2o_cls=o2o_cls.detach(),
        o2o_reg=o2o_reg.detach(),
        o2o_ctr=o2o_ctr.detach(),
        num_pos_o2m=num_pos_o2m,
        num_pos_o2o=num_pos_o2o,
        num_gt=num_gt,
    )
