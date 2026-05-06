import torch
from torch import Tensor, nn

from adp.distill.losses.matching import MatchedPair
from adp.model.dfine.arch.utils import box_cxcywh_to_xyxy, generalized_box_iou


__all__ = ["BoxDistillLoss"]


class BoxDistillLoss(nn.Module):
    """L1 + GIoU on (teacher_box, student_box) for GT-matched query pairs.

    Both boxes are in cxcywh normalized to [0, 1] (the same convention
    DFINECriterion uses).
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        giou_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.l1_weight = float(l1_weight)
        self.giou_weight = float(giou_weight)

    def forward(
        self,
        student_boxes: Tensor,
        teacher_boxes: Tensor,
        pairs: list[MatchedPair],
        weights: list[Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """
        Args:
            student_boxes: (B, Q_s, 4) cxcywh in [0, 1].
            teacher_boxes: (B, Q_t, 4) cxcywh in [0, 1] (detached upstream).
            pairs:         per-batch matched query indices (length B).
            weights:       optional per-pair quality weights from
                           adp.distill.losses.utils.compute_pair_quality_weights.
        """
        device = student_boxes.device
        s_collected: list[Tensor] = []
        t_collected: list[Tensor] = []
        w_collected: list[Tensor] = []
        for b, pair in enumerate(pairs):
            if len(pair) == 0:
                continue
            s_collected.append(student_boxes[b, pair.s_q.to(device)])
            t_collected.append(teacher_boxes[b, pair.t_q.to(device)])
            if weights is not None:
                w_collected.append(weights[b].to(device))

        if not s_collected:
            zero = student_boxes.sum() * 0.0
            return {
                "loss_box_l1_distill": zero,
                "loss_box_giou_distill": zero,
                "loss_box_distill": zero,
            }

        s = torch.cat(s_collected, dim=0)  # (N, 4)
        t = torch.cat(t_collected, dim=0)  # (N, 4)

        l1_per = (s - t).abs().sum(dim=-1)  # (N,)
        giou_matrix = generalized_box_iou(box_cxcywh_to_xyxy(s), box_cxcywh_to_xyxy(t))
        giou_per = 1.0 - torch.diag(giou_matrix)  # (N,)

        if w_collected:
            w = torch.cat(w_collected, dim=0)
            denom = w.sum().clamp_min(1e-6)
            l1 = (l1_per * w).sum() / denom
            giou_loss = (giou_per * w).sum() / denom
        else:
            l1 = l1_per.mean()
            giou_loss = giou_per.mean()

        total = self.l1_weight * l1 + self.giou_weight * giou_loss
        return {
            "loss_box_l1_distill": l1,
            "loss_box_giou_distill": giou_loss,
            "loss_box_distill": total,
        }
