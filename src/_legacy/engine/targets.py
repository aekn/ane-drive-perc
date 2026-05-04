from __future__ import annotations

from collections.abc import Sequence

import torch


INF = 1e8


def _per_image_assign(
    gt_boxes: torch.Tensor,  # (M, 4) xyxy
    gt_labels: torch.Tensor,  # (M,) int
    points: torch.Tensor,  # (P, 2) xy
    point_strides: torch.Tensor,  # (P,) per-point stride
    point_size_bands: torch.Tensor,  # (P, 2) [lo, hi]
    num_classes: int,
    radius: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    P = points.shape[0]
    M = gt_boxes.shape[0]

    if M == 0:
        cls_targets = torch.full(
            (P,), num_classes, dtype=torch.long, device=points.device
        )
        reg_targets = torch.zeros((P, 4), dtype=torch.float32, device=points.device)
        ctr_targets = torch.zeros((P,), dtype=torch.float32, device=points.device)
        pos_mask = torch.zeros((P,), dtype=torch.bool, device=points.device)
        return cls_targets, reg_targets, ctr_targets, pos_mask

    px = points[:, 0:1]  # (P, 1)
    py = points[:, 1:2]  # (P, 1)
    x1, y1 = gt_boxes[:, 0], gt_boxes[:, 1]
    x2, y2 = gt_boxes[:, 2], gt_boxes[:, 3]
    l = px - x1  # (P, M)
    t = py - y1
    r = x2 - px
    b = y2 - py
    ltrb = torch.stack([l, t, r, b], dim=-1)  # (P, M, 4)

    inside_box = ltrb.min(dim=-1).values > 0  # (P, M)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    radius_per_point = (point_strides * radius).unsqueeze(1)  # (P, 1)
    near_center_x = (px - cx).abs() < radius_per_point
    near_center_y = (py - cy).abs() < radius_per_point
    center_mask = near_center_x & near_center_y  # (P, M)

    max_reg = ltrb.max(dim=-1).values  # (P, M)
    lo = point_size_bands[:, 0:1]
    hi = point_size_bands[:, 1:2]
    band_mask = (max_reg >= lo) & (max_reg < hi)

    valid = inside_box & center_mask & band_mask  # (P, M)

    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)  # (M,)
    cost = areas.unsqueeze(0).expand(P, M).clone()  # (P, M)
    cost[~valid] = INF
    min_cost, gt_idx = cost.min(dim=1)  # (P,)
    pos_mask = min_cost < INF  # (P,)

    cls_targets = torch.full((P,), num_classes, dtype=torch.long, device=points.device)
    cls_targets[pos_mask] = gt_labels[gt_idx[pos_mask]]

    matched_ltrb = ltrb[torch.arange(P, device=points.device), gt_idx]  # (P, 4)
    reg_targets = torch.zeros_like(matched_ltrb)
    reg_targets[pos_mask] = matched_ltrb[pos_mask]

    l_, t_, r_, b_ = reg_targets.unbind(dim=-1)
    lr_min = torch.minimum(l_, r_).clamp(min=0)
    lr_max = torch.maximum(l_, r_).clamp(min=1e-8)
    tb_min = torch.minimum(t_, b_).clamp(min=0)
    tb_max = torch.maximum(t_, b_).clamp(min=1e-8)
    ctr_targets = torch.sqrt((lr_min / lr_max) * (tb_min / tb_max))
    ctr_targets[~pos_mask] = 0.0

    return cls_targets, reg_targets, ctr_targets, pos_mask


def assign_targets_o2m(
    gt_boxes_per_image: list[torch.Tensor],  # list of (M_b, 4)
    gt_labels_per_image: list[torch.Tensor],  # list of (M_b,)
    points_per_level: Sequence[torch.Tensor],  # each (N_l, 2)
    strides: Sequence[int],
    num_classes: int,
    size_bands: Sequence[tuple[float, float]] = ((-1, 64), (64, 128), (128, INF)),
    center_sampling_radius: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B = len(gt_boxes_per_image)
    device = points_per_level[0].device

    flat_points = torch.cat(list(points_per_level), dim=0)
    stride_chunks = [
        torch.full((p.shape[0],), s, dtype=torch.float32, device=device)
        for p, s in zip(points_per_level, strides, strict=True)
    ]
    flat_strides = torch.cat(stride_chunks, dim=0)
    band_chunks = []
    for p, (lo, hi) in zip(points_per_level, size_bands, strict=True):
        band_chunks.append(
            torch.tensor([lo, hi], dtype=torch.float32, device=device)
            .unsqueeze(0)
            .expand(p.shape[0], 2)
        )
    flat_bands = torch.cat(band_chunks, dim=0)

    cls_all, reg_all, ctr_all, pos_all = [], [], [], []
    for b in range(B):
        c, r, ct, pm = _per_image_assign(
            gt_boxes_per_image[b].to(device),
            gt_labels_per_image[b].to(device),
            flat_points,
            flat_strides,
            flat_bands,
            num_classes,
            center_sampling_radius,
        )
        cls_all.append(c)
        reg_all.append(r)
        ctr_all.append(ct)
        pos_all.append(pm)

    return (
        torch.stack(cls_all, dim=0),
        torch.stack(reg_all, dim=0),
        torch.stack(ctr_all, dim=0),
        torch.stack(pos_all, dim=0),
    )
