"""Benchmark CLI for MHA, MQA, GQA, and simplified MLA attention."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from benchmark.attention import GroupedQueryAttention, LatentKVAttention, MultiHeadAttention, MultiQueryAttention
from benchmark.attention.base import KVCache
from benchmark.metrics import DTYPE_BYTES, bytes_to_gib, kv_cache_bytes


ATTENTION_CHOICES = ["mha", "mqa", "gqa", "mla"]


def _torch_dtype(dtype: str) -> torch.dtype:
    dtype_key = dtype.lower()
    if dtype_key in {"fp32", "float32"}:
        return torch.float32
    if dtype_key in {"fp16", "float16"}:
        return torch.float16
    if dtype_key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def _default_kv_heads(attention: str, query_heads: int) -> int:
    return {
        "mha": query_heads,
        "gqa": max(1, query_heads // 4),
        "mqa": 1,
        "mla": max(1, query_heads // 8),
    }[attention]


def _resolve_kv_heads(args: argparse.Namespace, attention: str) -> int:
    if attention == "mha":
        return args.query_heads
    if attention == "mqa":
        return 1

    if args.group_size is not None:
        if args.query_heads % args.group_size != 0:
            raise ValueError("--query-heads must be divisible by --group-size")
        return args.query_heads // args.group_size

    if args.kv_heads is not None:
        return args.kv_heads

    return _default_kv_heads(attention, args.query_heads)


def _build_attention(
    *,
    attention: str,
    embed_dim: int,
    query_heads: int,
    kv_heads: int,
    latent_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    if attention == "mha":
        module = MultiHeadAttention(embed_dim=embed_dim, query_heads=query_heads)
    elif attention == "mqa":
        module = MultiQueryAttention(embed_dim=embed_dim, query_heads=query_heads)
    elif attention == "gqa":
        module = GroupedQueryAttention(embed_dim=embed_dim, query_heads=query_heads, kv_heads=kv_heads)
    elif attention == "mla":
        module = LatentKVAttention(
            embed_dim=embed_dim,
            query_heads=query_heads,
            kv_heads=kv_heads,
            latent_dim=latent_dim,
        )
    else:
        raise ValueError(f"Unsupported attention type: {attention}")

    return module.to(device=device, dtype=dtype).eval()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_call(device: torch.device, fn: Any) -> float:
    _synchronize(device)
    start = time.perf_counter()
    fn()
    _synchronize(device)
    return (time.perf_counter() - start) * 1000.0


def _synthetic_cache(
    *,
    attention: str,
    batch_size: int,
    context: int,
    kv_heads: int,
    head_dim: int,
    latent_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> KVCache:
    if attention == "mla":
        key = torch.randn(batch_size, context, latent_dim, device=device, dtype=dtype)
        value = torch.randn(batch_size, context, latent_dim, device=device, dtype=dtype)
        return key, value

    key = torch.randn(batch_size, kv_heads, context, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, kv_heads, context, head_dim, device=device, dtype=dtype)
    return key, value


def _estimated_cache_bytes(
    *,
    attention: str,
    layers: int,
    batch_size: int,
    context: int,
    kv_heads: int,
    head_dim: int,
    latent_dim: int,
    dtype: str,
) -> int:
    if attention == "mla":
        return 2 * layers * batch_size * context * latent_dim * DTYPE_BYTES[dtype.lower()]

    return kv_cache_bytes(
        layers=layers,
        sequence_length=context,
        kv_heads=kv_heads,
        head_dim=head_dim,
        batch_size=batch_size,
        dtype=dtype,
    )


def _estimate_attention_flops(
    *,
    attention: str,
    batch_size: int,
    query_tokens: int,
    key_tokens: int,
    embed_dim: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    latent_dim: int,
) -> int:
    q_proj = 2 * batch_size * query_tokens * embed_dim * embed_dim

    if attention == "mla":
        kv_down = 4 * batch_size * query_tokens * embed_dim * latent_dim
        kv_up = 4 * batch_size * key_tokens * latent_dim * kv_heads * head_dim
        kv_proj = kv_down + kv_up
    else:
        kv_proj = 4 * batch_size * query_tokens * embed_dim * kv_heads * head_dim

    out_proj = 2 * batch_size * query_tokens * embed_dim * embed_dim
    score_and_weighted_sum = 4 * batch_size * query_heads * query_tokens * key_tokens * head_dim
    return q_proj + kv_proj + out_proj + score_and_weighted_sum


def _estimate_arithmetic_intensity(
    *,
    flops: int,
    batch_size: int,
    query_tokens: int,
    cache_bytes: int,
    embed_dim: int,
    dtype: str,
) -> float:
    dtype_bytes = DTYPE_BYTES[dtype.lower()]
    activation_bytes = 2 * batch_size * query_tokens * embed_dim * dtype_bytes
    effective_bytes = max(1, activation_bytes + cache_bytes)
    return flops / effective_bytes


def _peak_memory_gib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return bytes_to_gib(torch.cuda.max_memory_allocated(device))


def _run_prefill(
    *,
    module: nn.Module,
    batch_size: int,
    context: int,
    embed_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> float:
    x = torch.randn(batch_size, context, embed_dim, device=device, dtype=dtype)

    with torch.inference_mode():
        for _ in range(warmup):
            module(x, use_cache=True)

        elapsed_ms = 0.0
        for _ in range(iterations):
            elapsed_ms += _time_call(device, lambda: module(x, use_cache=True))

    del x
    return elapsed_ms / iterations


def _run_decode(
    *,
    attention: str,
    module: nn.Module,
    batch_size: int,
    context: int,
    embed_dim: int,
    kv_heads: int,
    head_dim: int,
    latent_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> float:
    x = torch.randn(batch_size, 1, embed_dim, device=device, dtype=dtype)
    past_key_value = _synthetic_cache(
        attention=attention,
        batch_size=batch_size,
        context=context,
        kv_heads=kv_heads,
        head_dim=head_dim,
        latent_dim=latent_dim,
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():
        for _ in range(warmup):
            module(x, past_key_value=past_key_value, use_cache=True)

        elapsed_ms = 0.0
        for _ in range(iterations):
            elapsed_ms += _time_call(device, lambda: module(x, past_key_value=past_key_value, use_cache=True))

    del x, past_key_value
    return elapsed_ms / iterations


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_single(args: argparse.Namespace, attention: str) -> dict[str, Any]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = _torch_dtype(args.dtype)
    query_heads = args.query_heads
    kv_heads = _resolve_kv_heads(args, attention)
    head_dim = args.head_dim
    embed_dim = query_heads * head_dim
    latent_dim = args.latent_dim if args.latent_dim is not None else max(head_dim, embed_dim // 8)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    module = _build_attention(
        attention=attention,
        embed_dim=embed_dim,
        query_heads=query_heads,
        kv_heads=kv_heads,
        latent_dim=latent_dim,
        device=device,
        dtype=dtype,
    )

    prefill_ms = None
    decode_ms = None
    if args.mode in {"prefill", "both"}:
        prefill_ms = _run_prefill(
            module=module,
            batch_size=args.batch_size,
            context=args.context,
            embed_dim=embed_dim,
            device=device,
            dtype=dtype,
            warmup=args.warmup,
            iterations=args.iterations,
        )

    if args.mode in {"decode", "both"}:
        decode_ms = _run_decode(
            attention=attention,
            module=module,
            batch_size=args.batch_size,
            context=args.context,
            embed_dim=embed_dim,
            kv_heads=kv_heads,
            head_dim=head_dim,
            latent_dim=latent_dim,
            device=device,
            dtype=dtype,
            warmup=args.warmup,
            iterations=args.iterations,
        )

    cache_bytes = _estimated_cache_bytes(
        attention=attention,
        layers=args.layers,
        batch_size=args.batch_size,
        context=args.context,
        kv_heads=kv_heads,
        head_dim=head_dim,
        latent_dim=latent_dim,
        dtype=args.dtype,
    )
    one_layer_cache_bytes = _estimated_cache_bytes(
        attention=attention,
        layers=1,
        batch_size=args.batch_size,
        context=args.context,
        kv_heads=kv_heads,
        head_dim=head_dim,
        latent_dim=latent_dim,
        dtype=args.dtype,
    )

    prefill_flops = _estimate_attention_flops(
        attention=attention,
        batch_size=args.batch_size,
        query_tokens=args.context,
        key_tokens=args.context,
        embed_dim=embed_dim,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        latent_dim=latent_dim,
    )
    decode_flops = _estimate_attention_flops(
        attention=attention,
        batch_size=args.batch_size,
        query_tokens=1,
        key_tokens=args.context + 1,
        embed_dim=embed_dim,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        latent_dim=latent_dim,
    )

    prefill_gflops = None if prefill_ms is None else (prefill_flops / 1e9) / (prefill_ms / 1000.0)
    decode_gflops = None if decode_ms is None else (decode_flops / 1e9) / (decode_ms / 1000.0)
    prefill_tps = None if prefill_ms is None else (args.batch_size * args.context) / (prefill_ms / 1000.0)
    decode_tps = None if decode_ms is None else args.batch_size / (decode_ms / 1000.0)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "attention": attention,
        "device": str(device),
        "dtype": args.dtype,
        "mode": args.mode,
        "batch_size": args.batch_size,
        "context": args.context,
        "layers_for_kv_estimate": args.layers,
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "query_heads_per_kv_head": query_heads // kv_heads,
        "head_dim": head_dim,
        "embed_dim": embed_dim,
        "latent_dim": latent_dim if attention == "mla" else None,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "prefill_ms": prefill_ms,
        "decode_ms_per_token": decode_ms,
        "prefill_tokens_per_sec": prefill_tps,
        "decode_tokens_per_sec": decode_tps,
        "prefill_gflops": prefill_gflops,
        "decode_gflops": decode_gflops,
        "prefill_arithmetic_intensity": _estimate_arithmetic_intensity(
            flops=prefill_flops,
            batch_size=args.batch_size,
            query_tokens=args.context,
            cache_bytes=one_layer_cache_bytes,
            embed_dim=embed_dim,
            dtype=args.dtype,
        ),
        "decode_arithmetic_intensity": _estimate_arithmetic_intensity(
            flops=decode_flops,
            batch_size=args.batch_size,
            query_tokens=1,
            cache_bytes=one_layer_cache_bytes,
            embed_dim=embed_dim,
            dtype=args.dtype,
        ),
        "estimated_kv_cache_gib": bytes_to_gib(cache_bytes),
        "estimated_one_layer_kv_cache_gib": bytes_to_gib(one_layer_cache_bytes),
        "peak_torch_memory_gib": _peak_memory_gib(device),
    }

    del module
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark attention mechanisms.")
    parser.add_argument("--attention", choices=[*ATTENTION_CHOICES, "all"], required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mode", choices=["prefill", "decode", "both"], default="both")
    parser.add_argument("--layers", type=int, default=24, help="Layer count used for KV cache estimates.")
    parser.add_argument("--query-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=None, help="Override KV heads for GQA/MLA.")
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help="Query heads per KV head for GQA/MLA. Example: 4 means 16 Q heads -> 4 KV heads.",
    )
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/benchmarks"))
    parser.add_argument("--prefix", default="attention_benchmark")
    args = parser.parse_args()

    if args.kv_heads is not None and args.group_size is not None:
        parser.error("Use either --kv-heads or --group-size, not both.")

    attentions = ATTENTION_CHOICES if args.attention == "all" else [args.attention]
    rows = [run_single(args, attention) for attention in attentions]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.prefix}.json"
    csv_path = args.output_dir / f"{args.prefix}.csv"
    roofline_points_path = args.output_dir / f"{args.prefix}_roofline_points.csv"

    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    _write_csv(csv_path, rows)

    point_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["prefill_gflops"] is not None:
            point_rows.append(
                {
                    "label": f"{row['attention'].upper()} prefill b{row['batch_size']} c{row['context']}",
                    "arithmetic_intensity": row["prefill_arithmetic_intensity"],
                    "gflops": row["prefill_gflops"],
                }
            )
        if row["decode_gflops"] is not None:
            point_rows.append(
                {
                    "label": f"{row['attention'].upper()} decode b{row['batch_size']} c{row['context']}",
                    "arithmetic_intensity": row["decode_arithmetic_intensity"],
                    "gflops": row["decode_gflops"],
                }
            )
    if point_rows:
        _write_csv(roofline_points_path, point_rows)

    print(
        json.dumps(
            {
                "results_json": str(json_path),
                "results_csv": str(csv_path),
                "roofline_points_csv": str(roofline_points_path),
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
