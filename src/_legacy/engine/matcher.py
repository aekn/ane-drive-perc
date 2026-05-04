from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou


def _decode_ltrb_to_xyxy(points: torch.Tensor, ltrb: torch.Tensor) -> torch.Tensor:
    x1 = points[:, 0] - ltrb[:, 0]
    y1 = points[:, 1] - ltrb[:, 1]
    x2 = points[:, 0] + ltrb[:, 2]
    y2 = points[:, 1] + ltrb[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


@torch.no_grad()
def hungarian_match_o2o(
    cls_logits: torch.Tensor,  # (P, C)
    reg_pred: torch.Tensor,  # (P, 4)
    points: torch.Tensor,  # (P, 2)
    gt_boxes: torch.Tensor,  # (M, 4) xyxy
    gt_labels: torch.Tensor,  # (M,)
    lambda_cls: float = 1.0,
    lambda_box: float = 2.5,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    M = gt_boxes.shape[0]
    if M == 0:
        empty = torch.empty((0,), dtype=torch.long, device=points.device)
        return empty, empty

    cls_probs = cls_logits.sigmoid()  # (P, C)
    p_gt = cls_probs[:, gt_labels].clamp(min=1e-8, max=1 - 1e-8)  # (P, M)

    cls_cost = focal_alpha * (1 - p_gt).pow(focal_gamma) * (-p_gt.log())

    pred_xyxy = _decode_ltrb_to_xyxy(points, reg_pred)
    giou = generalized_box_iou(pred_xyxy, gt_boxes)  # (P, M)
    box_cost = 1.0 - giou

    cost = lambda_cls * cls_cost + lambda_box * box_cost  # (P, M)

    row, col = linear_sum_assignment(cost.detach().cpu().numpy())
    return (
        torch.as_tensor(row, dtype=torch.long, device=points.device),
        torch.as_tensor(col, dtype=torch.long, device=points.device),
    )
