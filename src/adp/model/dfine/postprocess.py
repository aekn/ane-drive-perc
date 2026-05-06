"""D-FINE postprocessing: convert raw decoder outputs to scored detections.

Used in two contexts:
  - Evaluation: rescale boxes to original image size for COCOeval.
  - Export: with input size baked in, this graph fuses into ONNX/CoreML
    so the deployed artifact has no host-side postprocessing.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class DFINEPostProcessor(nn.Module):
    """Sigmoid -> top-K over (queries x classes) -> cxcywh -> xyxy.

    Args:
      num_classes: number of foreground classes.
      num_top_queries: K.
    """

    def __init__(self, num_classes: int, num_top_queries: int = 300) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries

    @torch.no_grad()
    def forward(
        self, outputs: dict[str, Tensor], orig_sizes: Tensor
    ) -> list[dict[str, Tensor]]:
        """Returns per-image dicts with absolute-pixel xyxy boxes.

        Args:
          outputs: {"pred_logits": [B,Q,C], "pred_boxes": [B,Q,4] (cxcywh, normalized)}
          orig_sizes: [B, 2] long, (height, width)
        """
        logits = outputs["pred_logits"]
        boxes_cxcywh = outputs["pred_boxes"]
        batch_size, num_queries, num_classes = logits.shape

        scores_all = torch.sigmoid(logits)
        flat = scores_all.flatten(1)
        k = min(self.num_top_queries, flat.shape[1])

        topk_scores, topk_idx = torch.topk(flat, k, dim=-1)
        topk_labels = topk_idx % num_classes
        topk_qidx = topk_idx // num_classes

        batch_idx = (
            torch.arange(batch_size, device=boxes_cxcywh.device)
            .unsqueeze(1)
            .expand_as(topk_qidx)
        )
        topk_boxes = boxes_cxcywh[batch_idx, topk_qidx]  # [B, K, 4]

        cx, cy, w, h = topk_boxes.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_xyxy_norm = torch.stack([x1, y1, x2, y2], dim=-1)

        # Scale normalized -> absolute using each image's original size
        heights = orig_sizes[:, 0].view(-1, 1, 1).float()
        widths = orig_sizes[:, 1].view(-1, 1, 1).float()
        scale = torch.cat([widths, heights, widths, heights], dim=2)
        boxes_xyxy_abs = boxes_xyxy_norm * scale

        results: list[dict[str, Any]] = []
        for b in range(batch_size):
            results.append(
                {
                    "boxes": boxes_xyxy_abs[b],
                    "scores": topk_scores[b],
                    "labels": topk_labels[b],
                }
            )
        return results
