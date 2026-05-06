from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


__all__ = ["AIFIFeatureDistillLoss"]


class AIFIFeatureDistillLoss(nn.Module):
    """Cosine-similarity feature distillation on AIFI deepest-scale output.

    Args:
        student_channels: channel count of student's AIFI output.
        teacher_channels: channel count of teacher's AIFI output.
    """

    def __init__(self, *, student_channels: int, teacher_channels: int) -> None:
        super().__init__()
        # Linear projection from student -> teacher channels. Operates on the
        # flattened token dimension, equivalent to Conv1x1 on the spatial map.
        self.projector = nn.Linear(student_channels, teacher_channels, bias=True)
        # Init: small random gain so the early signal is noise-dominated, then
        # adapter learns alignment. Standard Kaiming-ish for linear-only path.
        nn.init.xavier_uniform_(self.projector.weight)
        nn.init.zeros_(self.projector.bias)

    def forward(
        self,
        student_feat: Tensor,
        teacher_feat: Tensor,
    ) -> dict[str, Tensor]:
        """
        Args:
            student_feat: AIFI output, (B, Cs, H, W).
            teacher_feat: AIFI output, (B, Ct, H, W) (detached upstream).

        Spatial dimensions must match between student and teacher (both are
        the deepest scale at the same image resolution).
        """
        if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
            raise ValueError(
                f"AIFI feature spatial mismatch: "
                f"student {tuple(student_feat.shape[-2:])} vs teacher {tuple(teacher_feat.shape[-2:])}"
            )

        B, Cs, H, W = student_feat.shape
        Ct = teacher_feat.shape[1]

        # (B, Cs, H, W) -> (B*H*W, Cs) -> linear -> (B*H*W, Ct).
        s_tokens = student_feat.permute(0, 2, 3, 1).reshape(B * H * W, Cs)
        s_proj = self.projector(s_tokens)  # (B*H*W, Ct)
        t_tokens = teacher_feat.permute(0, 2, 3, 1).reshape(B * H * W, Ct).detach()

        # Mean negative cosine similarity 2 get std loss (minimize -> align)
        cos_sim = F.cosine_similarity(s_proj, t_tokens, dim=-1)
        loss = (1.0 - cos_sim).mean()
        return {"loss_aifi_cosine_distill": loss}
