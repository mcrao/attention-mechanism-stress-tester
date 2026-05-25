"""Interchangeable attention module implementations."""

from benchmark.attention.base import AttentionConfig, AttentionOutput, ProjectedAttention
from benchmark.attention.gqa import GroupedQueryAttention
from benchmark.attention.mha import MultiHeadAttention
from benchmark.attention.mla import LatentKVAttention
from benchmark.attention.mqa import MultiQueryAttention

__all__ = [
    "AttentionConfig",
    "AttentionOutput",
    "GroupedQueryAttention",
    "LatentKVAttention",
    "MultiHeadAttention",
    "MultiQueryAttention",
    "ProjectedAttention",
]
