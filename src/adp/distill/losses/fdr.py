import torch
import torch.nn.functional as F
from torch import Tensor, nn

from adp.distill.losses.matching import MatchedPair


__all__ = ["FDRDistributionDistillLoss"]


class FDRDistributionDistillLoss(nn.Module):
    """KL divergence between teacher and student per-edge bin distributions.

    Args:
        reg_max:     bin count (we use 32 for both teacher and student, so the
                     bin distributions are directly comparable).
        temperature: softens both distributions before KL. T=1 is direct
                     KL; T=2 smooths the bins more.
    """

    def __init__(self, reg_max: int = 32, temperature: float = 1.0) -> None:
        super().__init__()
        if reg_max < 4 or reg_max % 2 != 0:
            raise ValueError(f"reg_max must be even and >= 4, got {reg_max}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.reg_max = reg_max
        self.temperature = float(temperature)

    def forward(
        self,
        student_corners: Tensor,
        teacher_corners: Tensor,
        pairs: list[MatchedPair],
        weights: list[Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """
        Args:
            student_corners: (B, Q_s, 4*(reg_max+1)) raw FGL distribution logits.
            teacher_corners: (B, Q_t, 4*(reg_max+1)) raw FGL logits (detached upstream).
            pairs:           per-batch matched query indices.
            weights:         optional per-pair quality weights (broadcast across 4 edges).
        """
        device = student_corners.device
        bins = self.reg_max + 1
        T = self.temperature

        s_collected: list[Tensor] = []
        t_collected: list[Tensor] = []
        w_collected: list[Tensor] = []
        for b, pair in enumerate(pairs):
            if len(pair) == 0:
                continue
            s_pq = student_corners[b, pair.s_q.to(device)]  # (P, 4*bins)
            t_pq = teacher_corners[b, pair.t_q.to(device)]
            s_collected.append(s_pq.reshape(-1, 4, bins))
            t_collected.append(t_pq.reshape(-1, 4, bins))
            if weights is not None:
                w_collected.append(weights[b].to(device))

        if not s_collected:
            return {"loss_fdr_distill": student_corners.sum() * 0.0}

        s = torch.cat(s_collected, dim=0)  # (N, 4, bins)
        t = torch.cat(t_collected, dim=0).detach()
        log_p_s = F.log_softmax(s / T, dim=-1)
        p_t = F.softmax(t / T, dim=-1)
        # KL per (pair, edge): sum over bins.
        kl_per_pair_edge = (p_t * (p_t.clamp_min(1e-12).log() - log_p_s)).sum(
            dim=-1
        )  # (N, 4)
        # Mean over the 4 edges -> per-pair scalar.
        kl_per_pair = kl_per_pair_edge.mean(dim=-1)  # (N,)

        if w_collected:
            w = torch.cat(w_collected, dim=0)  # (N,)
            denom = w.sum().clamp_min(1e-6)
            loss = (kl_per_pair * w).sum() / denom
        else:
            loss = kl_per_pair.mean()
        # standard temperature scaling so gradient stays comparable
        loss = loss * (T * T)
        return {"loss_fdr_distill": loss}
