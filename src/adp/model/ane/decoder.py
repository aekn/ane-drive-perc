from torch import Tensor, nn

from adp.model.ane.blocks import (
    ANEFeedForward,
    ANELayerNorm,
    ANEMultiHeadAttention,
)


__all__ = ["ANEDecoderLayer"]


class ANEDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_q1 = ANELayerNorm(embed_dim)
        self.self_attn = ANEMultiHeadAttention(
            embed_dim, num_heads=num_heads, dropout=dropout
        )

        self.norm_q2 = ANELayerNorm(embed_dim)
        self.cross_attn = ANEMultiHeadAttention(
            embed_dim, num_heads=num_heads, dropout=dropout
        )

        self.norm_q3 = ANELayerNorm(embed_dim)
        self.ffn = ANEFeedForward(
            embed_dim,
            hidden_dim=int(embed_dim * ffn_ratio),
            dropout=dropout,
        )

    def forward(
        self,
        queries: Tensor,
        memory: Tensor,
        *,
        query_pos: Tensor | None = None,
        memory_pos: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        # selfattn
        residual = queries
        h = self.norm_q1(queries)
        q = h if query_pos is None else h + query_pos
        k = h if query_pos is None else h + query_pos
        h = self.self_attn(q, k, h)
        queries = residual + h

        # crosattn
        residual = queries
        h = self.norm_q2(queries)
        q = h if query_pos is None else h + query_pos
        k_mem = memory if memory_pos is None else memory + memory_pos
        h = self.cross_attn(q, k_mem, memory, key_padding_mask=memory_key_padding_mask)
        queries = residual + h

        # ffn
        residual = queries
        h = self.norm_q3(queries)
        h = self.ffn(h)
        return residual + h
