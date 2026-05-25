"""Simplified latent-KV attention for MLA-style experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from benchmark.attention.base import AttentionOutput, KVCache, repeat_kv


class LatentKVAttention(nn.Module):
    """Compress keys/values into a latent cache, then expand for attention.

    This is intentionally not a DeepSeek reproduction. It is a controlled
    benchmark variant for studying how a smaller cached representation changes
    memory pressure and throughput.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        query_heads: int,
        kv_heads: int,
        latent_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % query_heads != 0:
            raise ValueError("embed_dim must be divisible by query_heads")
        if query_heads % kv_heads != 0:
            raise ValueError("query_heads must be divisible by kv_heads")

        self.embed_dim = embed_dim
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.head_dim = embed_dim // query_heads
        self.latent_dim = latent_dim
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_down_proj = nn.Linear(embed_dim, latent_dim, bias=bias)
        self.v_down_proj = nn.Linear(embed_dim, latent_dim, bias=bias)
        self.k_up_proj = nn.Linear(latent_dim, kv_heads * self.head_dim, bias=bias)
        self.v_up_proj = nn.Linear(latent_dim, kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape_q(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.query_heads, self.head_dim).transpose(1, 2)

    def _shape_kv(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.kv_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> AttentionOutput:
        query = self._shape_q(self.q_proj(x))
        latent_key = self.k_down_proj(x)
        latent_value = self.v_down_proj(x)

        if past_key_value is not None:
            past_latent_key, past_latent_value = past_key_value
            latent_key = torch.cat([past_latent_key, latent_key], dim=1)
            latent_value = torch.cat([past_latent_value, latent_value], dim=1)

        key = self._shape_kv(self.k_up_proj(latent_key))
        value = self._shape_kv(self.v_up_proj(latent_value))
        cache = (latent_key, latent_value) if use_cache else None

        repeat_factor = self.query_heads // self.kv_heads
        expanded_key = repeat_kv(key, repeat_factor)
        expanded_value = repeat_kv(value, repeat_factor)
        is_causal = attention_mask is None and past_key_value is None

        attended = F.scaled_dot_product_attention(
            query,
            expanded_key,
            expanded_value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attended = attended.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.embed_dim)
        return AttentionOutput(output=self.out_proj(attended), kv_cache=cache)

