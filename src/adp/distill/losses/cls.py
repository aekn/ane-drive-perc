import torch
import torch.nn.functional as F
from torch import Tensor, nn

from adp.distill.losses.matching import MatchedPair


__all__ = ["ClsDistillLoss"]


class ClsDistillLoss(nn.Module):
    """Temperature-scaled KL between teacher and student cls logits, FG only.

    Loss formula (averaged over matched pairs and classes):

        L = T^2 * KL( softmax(teacher / T) || softmax(student / T) )

    Following the standard distillation convention (Hinton 2015) of scaling by
    T^2 so the gradient magnitude doesn't shrink when T > 1.
    """

    def __init__(self, temperature: float = 2.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = float(temperature)

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        pairs: list[MatchedPair],
        weights: list[Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """
        Args:
            student_logits: (B, Q_s, K) raw cls logits.
            teacher_logits: (B, Q_t, K) raw cls logits (detached upstream).
            pairs:          per-batch matched query indices.
            weights:        optional per-pair quality weights.
        """
        device = student_logits.device
        s_collected: list[Tensor] = []
        t_collected: list[Tensor] = []
        w_collected: list[Tensor] = []
        for b, pair in enumerate(pairs):
            if len(pair) == 0:
                continue
            s_collected.append(student_logits[b, pair.s_q.to(device)])
            t_collected.append(teacher_logits[b, pair.t_q.to(device)])
            if weights is not None:
                w_collected.append(weights[b].to(device))

        if not s_collected:
            return {"loss_cls_distill": student_logits.sum() * 0.0}

        s = torch.cat(s_collected, dim=0)  # (N, K)
        t = torch.cat(t_collected, dim=0)  # (N, K)

        T = self.temperature
        log_p_s = F.log_softmax(s / T, dim=-1)
        p_t = F.softmax(t / T, dim=-1)
        kl_per_pair = (p_t * (p_t.clamp_min(1e-12).log() - log_p_s)).sum(dim=-1)  # (N,)

        if w_collected:
            w = torch.cat(w_collected, dim=0)
            denom = w.sum().clamp_min(1e-6)
            kl = (kl_per_pair * w).sum() / denom
        else:
            kl = kl_per_pair.mean()

        return {"loss_cls_distill": kl * (T * T)}
