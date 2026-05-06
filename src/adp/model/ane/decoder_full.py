from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from adp.model.ane.decoder import ANEDecoderLayer
from adp.model.ane.encoder import build_2d_sincos_pos_embed
from adp.model.ane.heads import (
    ANEClsHead,
    ANEFGLBoxHead,
    fgl_decode_boxes,
    fgl_weighting_function,
)


__all__ = ["ANEDecoder"]


def _bias_init_with_prob(prior_prob: float = 0.01) -> float:
    """Inverse sigmoid for cls bias init (focal style)."""
    return float(-math.log((1.0 - prior_prob) / prior_prob))


class ANEDecoder(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        embed_dim: int = 192,
        num_queries: int = 300,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
        reg_max: int = 16,
        reg_scale: float = 4.0,
        num_levels: int = 3,
        eval_spatial_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.num_levels = num_levels
        self.reg_max = reg_max
        self.eval_spatial_size = eval_spatial_size

        self.layers = nn.ModuleList(
            [
                ANEDecoderLayer(
                    embed_dim, num_heads=num_heads, ffn_ratio=ffn_ratio, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.cls_heads = nn.ModuleList(
            [ANEClsHead(embed_dim, num_classes) for _ in range(num_layers)]
        )
        self.box_heads = nn.ModuleList(
            [
                ANEFGLBoxHead(embed_dim, hidden_dim=embed_dim, reg_max=reg_max)
                for _ in range(num_layers)
            ]
        )

        self.query_embed = nn.Parameter(torch.zeros(1, embed_dim, 1, num_queries))
        nn.init.normal_(self.query_embed, std=0.02)

        self.init_ref_points = nn.Parameter(torch.zeros(1, num_queries, 4))
        nn.init.uniform_(self.init_ref_points, -1.0, 1.0)

        self.level_embed = nn.Parameter(torch.zeros(num_levels, embed_dim))
        nn.init.normal_(self.level_embed, std=0.02)

        self.up = nn.Parameter(torch.tensor([0.5]))
        self.register_buffer(
            "reg_scale", torch.tensor([reg_scale], dtype=torch.float32)
        )

        self._mem_pos_cache_key: tuple[tuple[int, int], ...] | None = None
        self.register_buffer("_mem_pos_cache", torch.zeros(0), persistent=False)
        if eval_spatial_size is not None:
            H_in, W_in = int(eval_spatial_size[0]), int(eval_spatial_size[1])
            self._static_levels: list[tuple[int, int]] = [
                (H_in // 8, W_in // 8),
                (H_in // 16, W_in // 16),
                (H_in // 32, W_in // 32),
            ][:num_levels]
            parts = []
            for level, (h, w) in enumerate(self._static_levels):
                sincos = build_2d_sincos_pos_embed(
                    h, w, embed_dim, device=torch.device("cpu")
                )
                level_emb = self.level_embed[level].view(1, -1, 1, 1)
                parts.append(sincos + level_emb)
            self._mem_pos_cache = torch.cat(parts, dim=-1)
            self._mem_pos_cache_key = tuple(self._static_levels)
        else:
            self._static_levels = None

        self.register_buffer("_project_static", torch.zeros(0), persistent=False)
        self._project_is_static = False

        self._reset_head_biases()

    def convert_to_deploy(self) -> None:
        from adp.model.ane.heads import fgl_weighting_function

        with torch.no_grad():
            project = fgl_weighting_function(self.reg_max, self.up, self.reg_scale)
        self._project_static = project.detach().clone()
        self._project_is_static = True

    def _reset_head_biases(self) -> None:
        bias = _bias_init_with_prob(0.01)
        for head in self.cls_heads:
            nn.init.constant_(head.proj.bias, bias)
        # Final box head bias init to zero so initial deltas around ref are zero-mean.
        for head in self.box_heads:
            nn.init.constant_(head.mlp.layers[-1].weight, 0.0)
            nn.init.constant_(head.mlp.layers[-1].bias, 0.0)

    def _build_memory_pos(self, encoder_feats: Sequence[Tensor]) -> Tensor:
        if self._static_levels is not None and self._mem_pos_cache.numel() > 0:
            return self._mem_pos_cache.to(encoder_feats[0].device)

        spatial_key = tuple((int(f.shape[-2]), int(f.shape[-1])) for f in encoder_feats)
        if self._mem_pos_cache_key == spatial_key and self._mem_pos_cache.numel() > 0:
            return self._mem_pos_cache

        device = encoder_feats[0].device
        parts = []
        for level, feat in enumerate(encoder_feats):
            h, w = int(feat.shape[-2]), int(feat.shape[-1])
            sincos = build_2d_sincos_pos_embed(h, w, self.embed_dim, device=device)
            level_emb = self.level_embed[level].view(1, -1, 1, 1)
            parts.append(sincos + level_emb)
        pos = torch.cat(parts, dim=-1)
        self._mem_pos_cache = pos
        self._mem_pos_cache_key = spatial_key
        return pos

    def forward(
        self,
        encoder_feats: Sequence[Tensor],
        targets: list[dict[str, Tensor]] | None = None,  # noqa: ARG002 (DN deferred)
    ) -> dict[str, Tensor | list]:
        if len(encoder_feats) != self.num_levels:
            raise ValueError(
                f"expected {self.num_levels} encoder feature scales, got {len(encoder_feats)}"
            )

        B = encoder_feats[0].shape[0]
        device = encoder_feats[0].device

        memory_parts = [f.flatten(2).unsqueeze(2) for f in encoder_feats]
        memory = torch.cat(memory_parts, dim=-1)  # (B, C, 1, S_total)
        memory_pos = self._build_memory_pos(encoder_feats)

        query_pos = self.query_embed.expand(B, -1, -1, -1)
        queries = torch.zeros_like(query_pos)
        ref_points = self.init_ref_points.sigmoid().expand(
            B, -1, -1
        )  # (B, Q, 4) cxcywh

        if self._project_is_static and self._project_static.numel() > 0:
            project = self._project_static
        else:
            project = fgl_weighting_function(self.reg_max, self.up, self.reg_scale)

        out_logits: list[Tensor] = []
        out_bboxes: list[Tensor] = []
        out_corners: list[Tensor] = []
        out_refs: list[Tensor] = []

        for i, layer in enumerate(self.layers):
            queries = layer(
                queries,
                memory,
                query_pos=query_pos,
                memory_pos=memory_pos,
            )

            logits_i = self.cls_heads[i](queries)
            corners_i = self.box_heads[i](queries)
            boxes_i = fgl_decode_boxes(
                corners_i,
                ref_points,
                reg_max=self.reg_max,
                project=project,
                reg_scale=self.reg_scale,
            )

            out_logits.append(logits_i)
            out_bboxes.append(boxes_i)
            out_corners.append(corners_i)
            out_refs.append(ref_points)

            ref_points = boxes_i.detach()

        if self.training:
            aux_outputs = [
                {
                    "pred_logits": out_logits[i],
                    "pred_boxes": out_bboxes[i],
                    "pred_corners": out_corners[i],
                    "ref_points": out_refs[i],
                    "teacher_corners": out_corners[-1],
                    "teacher_logits": out_logits[-1],
                }
                for i in range(self.num_layers - 1)
            ]
            return {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_corners": out_corners[-1],
                "ref_points": out_refs[-1],
                "up": self.up,
                "reg_scale": self.reg_scale,
                "aux_outputs": aux_outputs,
            }

        return {
            "pred_logits": out_logits[-1],
            "pred_boxes": out_bboxes[-1],
        }
