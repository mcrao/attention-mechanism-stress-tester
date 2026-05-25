"""Common attention interfaces used by the benchmark harness."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


KVCache = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class AttentionConfig:
    embed_dim: int
    query_heads: int
    kv_heads: int
    dropout: float = 0.0
    bias: bool = False

    @property
    def head_dim(self) -> int:
        if self.embed_dim % self.query_heads != 0:
            raise ValueError("embed_dim must be divisible by query_heads")
        return self.embed_dim // self.query_heads

    def validate(self) -> None:
        _ = self.head_dim
        if self.query_heads % self.kv_heads != 0:
            raise ValueError("query_heads must be divisible by kv_heads")


@dataclass
class AttentionOutput:
    output: torch.Tensor
    kv_cache: KVCache | None = None


def repeat_kv(x: torch.Tensor, repeat_factor: int) -> torch.Tensor:
    """Repeat KV heads to match the query head count."""

    if repeat_factor == 1:
        return x
    batch, kv_heads, tokens, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, kv_heads, repeat_factor, tokens, head_dim)
    return x.reshape(batch, kv_heads * repeat_factor, tokens, head_dim)


class ProjectedAttention(nn.Module):
    """Shared implementation for MHA, GQA, and MQA.

    Tensor shapes:
        input: [batch, tokens, embed_dim]
        q:     [batch, query_heads, tokens, head_dim]
        k/v:   [batch, kv_heads, tokens, head_dim]
    """

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.embed_dim, config.kv_heads * config.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.embed_dim, config.kv_heads * config.head_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)

    def _shape_q(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.config.query_heads, self.config.head_dim).transpose(1, 2)

    def _shape_kv(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.config.kv_heads, self.config.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> AttentionOutput:
        query = self._shape_q(self.q_proj(x))
        key = self._shape_kv(self.k_proj(x))
        value = self._shape_kv(self.v_proj(x))

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)

        cache = (key, value) if use_cache else None
        repeat_factor = self.config.query_heads // self.config.kv_heads
        expanded_key = repeat_kv(key, repeat_factor)
        expanded_value = repeat_kv(value, repeat_factor)

        is_causal = attention_mask is None and past_key_value is None
        attended = F.scaled_dot_product_attention(
            query,
            expanded_key,
            expanded_value,
            attn_mask=attention_mask,
            dropout_p=self.config.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attended = attended.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.config.embed_dim)
        return AttentionOutput(output=self.out_proj(attended), kv_cache=cache)

