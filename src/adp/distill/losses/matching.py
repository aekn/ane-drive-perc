from __future__ import annotations

import torch
from torch import Tensor


__all__ = ["gt_mediated_pairs", "MatchedPair"]


class MatchedPair:
    """Per-batch pairing of student and teacher query indices for one image.

    s_q and t_q are the matched query indices into the student and
    teacher prediction tensors respectively; both have the same length, equal
    to the number of GT boxes that both models managed to match (typically
    all of them, since Hungarian assigns every GT to a unique query).
    """

    __slots__ = ("s_q", "t_q")

    def __init__(self, s_q: Tensor, t_q: Tensor) -> None:
        self.s_q = s_q
        self.t_q = t_q

    def __len__(self) -> int:
        return int(self.s_q.numel())


def gt_mediated_pairs(
    student_indices: list[tuple[Tensor, Tensor]],
    teacher_indices: list[tuple[Tensor, Tensor]],
) -> list[MatchedPair]:
    """Build per-batch (student_q, teacher_q) pairs from independent GT matches.

    Args:
        student_indices: per-batch (query_idx, gt_idx) from running the
            Hungarian matcher on student outputs vs targets.
        teacher_indices: same, on teacher outputs vs the same targets.

    Returns:
        A list of MatchedPair, one per batch element. The pair lists
        may be empty if the image has no GTs.
    """
    if len(student_indices) != len(teacher_indices):
        raise ValueError(
            f"batch size mismatch: student {len(student_indices)} vs teacher {len(teacher_indices)}"
        )

    out: list[MatchedPair] = []
    for (s_q, s_gt), (t_q, t_gt) in zip(student_indices, teacher_indices):
        if s_gt.numel() == 0 or t_gt.numel() == 0:
            empty = torch.zeros(0, dtype=torch.long, device=s_q.device)
            out.append(MatchedPair(empty, empty.clone()))
            continue
        # Lookup: gt_idx -> teacher query index (one entry per GT).
        t_lookup = {int(g): int(q) for q, g in zip(t_q.tolist(), t_gt.tolist())}
        s_q_list: list[int] = []
        t_q_list: list[int] = []
        for q, g in zip(s_q.tolist(), s_gt.tolist()):
            t_q_for_gt = t_lookup.get(int(g))
            if t_q_for_gt is None:
                continue
            s_q_list.append(int(q))
            t_q_list.append(t_q_for_gt)
        out.append(
            MatchedPair(
                torch.tensor(s_q_list, dtype=torch.long, device=s_q.device),
                torch.tensor(t_q_list, dtype=torch.long, device=s_q.device),
            )
        )
    return out
