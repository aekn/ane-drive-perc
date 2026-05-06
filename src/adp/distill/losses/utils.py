from __future__ import annotations

import torch
from torch import Tensor

from adp.distill.losses.matching import MatchedPair
from adp.model.dfine.arch.utils import box_cxcywh_to_xyxy, generalized_box_iou


__all__ = ["compute_pair_quality_weights", "aifi_ramp_schedule"]


def compute_pair_quality_weights(
    teacher_logits: Tensor,
    teacher_boxes: Tensor,
    targets: list[dict[str, Tensor]],
    pairs: list[MatchedPair],
    *,
    eps: float = 1e-6,
) -> list[Tensor]:
    out: list[Tensor] = []
    teacher_probs = (
        teacher_logits.detach().sigmoid()
    )  # focal-loss conv: per-class sigmoid

    for b, pair in enumerate(pairs):
        if len(pair) == 0:
            out.append(torch.zeros(0, device=teacher_logits.device))
            continue

        target_labels = targets[b]["labels"]
        target_boxes = targets[b]["boxes"]
        s_q = pair.s_q.to(teacher_logits.device)
        t_q = pair.t_q.to(teacher_logits.device)

        teacher_boxes_b = teacher_boxes[b, t_q]  # (P, 4)
        if target_boxes.numel() == 0:
            out.append(torch.zeros(s_q.numel(), device=teacher_logits.device))
            continue

        iou_matrix = generalized_box_iou(
            box_cxcywh_to_xyxy(teacher_boxes_b),
            box_cxcywh_to_xyxy(target_boxes),
        )
        iou_per_pair, gt_idx_per_pair = iou_matrix.clamp(min=0.0).max(dim=1)

        gt_classes = target_labels[gt_idx_per_pair]  # (P,)
        prob_per_pair = teacher_probs[b, t_q, gt_classes]  # (P,)

        weight = (prob_per_pair.clamp(min=0.0) + eps).sqrt() * (
            iou_per_pair + eps
        ).sqrt()
        out.append(weight.detach())  # weights don't backprop

    return out


def aifi_ramp_schedule(
    step: int,
    total_steps: int,
    *,
    peak_weight: float,
    warmup_frac: float = 0.10,
    ramp_frac: float = 0.20,
) -> float:
    if total_steps <= 0:
        return float(peak_weight)
    progress = step / float(total_steps)
    warmup_end = float(warmup_frac)
    ramp_end = float(warmup_frac + ramp_frac)
    if progress < warmup_end:
        return 0.0
    if progress >= ramp_end:
        return float(peak_weight)
    # Linear ramp.
    span = ramp_end - warmup_end
    if span <= 0:
        return float(peak_weight)
    frac = (progress - warmup_end) / span
    return float(peak_weight) * frac
