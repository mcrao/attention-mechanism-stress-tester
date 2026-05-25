"""Shared metric helpers for inference benchmarks."""

from __future__ import annotations


DTYPE_BYTES = {
    "fp32": 4,
    "float32": 4,
    "bf16": 2,
    "bfloat16": 2,
    "fp16": 2,
    "float16": 2,
    "int8": 1,
    "fp8": 1,
}


def kv_cache_bytes(
    *,
    layers: int,
    sequence_length: int,
    kv_heads: int,
    head_dim: int,
    batch_size: int = 1,
    dtype: str = "fp16",
) -> int:
    """Return KV cache bytes for a decoder-only transformer.

    Formula:
        2 * batch * layers * sequence_length * kv_heads * head_dim * dtype_bytes

    The leading 2 accounts for separate key and value caches.
    """

    dtype_key = dtype.lower()
    if dtype_key not in DTYPE_BYTES:
        supported = ", ".join(sorted(DTYPE_BYTES))
        raise ValueError(f"Unsupported dtype '{dtype}'. Supported: {supported}")

    return (
        2
        * batch_size
        * layers
        * sequence_length
        * kv_heads
        * head_dim
        * DTYPE_BYTES[dtype_key]
    )


def bytes_to_gib(num_bytes: int) -> float:
    """Convert bytes to GiB."""

    return num_bytes / (1024**3)

