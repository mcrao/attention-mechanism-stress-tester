"""Grouped-query attention."""

from __future__ import annotations

from benchmark.attention.base import AttentionConfig, ProjectedAttention


class GroupedQueryAttention(ProjectedAttention):
    def __init__(
        self,
        *,
        embed_dim: int,
        query_heads: int,
        kv_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__(
            AttentionConfig(
                embed_dim=embed_dim,
                query_heads=query_heads,
                kv_heads=kv_heads,
                dropout=dropout,
                bias=bias,
            )
        )

