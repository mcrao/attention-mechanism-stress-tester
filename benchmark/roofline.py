"""Roofline plotting for the active GPU.

Example:
    python -m benchmark.roofline --dtype fp16 --output-dir results/roofline

Use --peak-tflops and --memory-bandwidth-gbps to override catalog estimates.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "attention_stress_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from benchmark.gpu_info import detect_gpu, infer_hardware_spec


ATTENTION_COLORS = {
    "MHA": "tab:blue",
    "MQA": "tab:orange",
    "GQA": "tab:green",
    "MLA": "tab:red",
}

MODE_MARKERS = {
    "prefill": "^",
    "decode": "o",
}


def _peak_for_dtype(spec: object, dtype: str) -> float | None:
    if spec is None:
        return None
    dtype_key = dtype.lower()
    if dtype_key in {"fp32", "float32"}:
        return getattr(spec, "fp32_tflops")
    if dtype_key in {"fp16", "float16"}:
        return getattr(spec, "fp16_tensor_tflops")
    if dtype_key in {"bf16", "bfloat16"}:
        return getattr(spec, "bf16_tensor_tflops")
    raise ValueError(f"Unsupported roofline dtype: {dtype}")


def _load_points(path: Path | None) -> list[dict[str, float | str]]:
    if path is None:
        return []

    points: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"label", "arithmetic_intensity", "gflops"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        for row in reader:
            points.append(
                {
                    "label": row["label"],
                    "arithmetic_intensity": float(row["arithmetic_intensity"]),
                    "gflops": float(row["gflops"]),
                }
            )
    return points


def _point_parts(label: str) -> tuple[str, str]:
    parts = label.split()
    attention = parts[0].upper() if parts else "POINT"
    mode = parts[1].lower() if len(parts) > 1 else "point"
    return attention, mode


def _short_label(label: str) -> str:
    attention, mode = _point_parts(label)
    mode_short = {"prefill": "P", "decode": "D"}.get(mode, mode[:1].upper())
    return f"{attention}-{mode_short}"


def _label_offset(index: int, point_count: int, mode: str) -> tuple[int, int, str]:
    if mode == "prefill":
        offsets = [(-56, 24), (-44, 42), (-34, 60), (-24, 78), (-14, 96)]
        dx, dy = offsets[index % len(offsets)]
        return dx, dy, "right"

    offsets = [(14, -22), (18, 10), (20, 34), (22, 58), (24, 82)]
    dx, dy = offsets[index % len(offsets)]
    return dx, dy, "left"


def _format_point_summary(point: dict[str, float | str]) -> str:
    return (
        f"{_short_label(str(point['label'])):<6} "
        f"{float(point['arithmetic_intensity']):>8.2f} "
        f"{float(point['gflops']):>11,.0f}"
    )


def build_roofline(
    *,
    peak_tflops: float,
    memory_bandwidth_gbps: float,
    ai_min: float,
    ai_max: float,
    samples: int,
) -> dict[str, list[float] | float]:
    arithmetic_intensity = np.logspace(np.log10(ai_min), np.log10(ai_max), samples)
    memory_roof_gflops = memory_bandwidth_gbps * arithmetic_intensity
    compute_roof_gflops = peak_tflops * 1000.0
    attainable_gflops = np.minimum(memory_roof_gflops, compute_roof_gflops)
    ridge_point = compute_roof_gflops / memory_bandwidth_gbps
    return {
        "arithmetic_intensity": arithmetic_intensity.tolist(),
        "memory_roof_gflops": memory_roof_gflops.tolist(),
        "attainable_gflops": attainable_gflops.tolist(),
        "compute_roof_gflops": compute_roof_gflops,
        "ridge_point_flop_per_byte": ridge_point,
    }


def plot_roofline(
    *,
    roofline: dict[str, list[float] | float],
    title: str,
    output_png: Path,
    achieved_points: list[dict[str, float | str]],
    label_style: str,
) -> None:
    ai = np.array(roofline["arithmetic_intensity"])
    memory_roof = np.array(roofline["memory_roof_gflops"])
    attainable = np.array(roofline["attainable_gflops"])
    compute_roof = float(roofline["compute_roof_gflops"])
    ridge = float(roofline["ridge_point_flop_per_byte"])

    fig, (ax, ax_info) = plt.subplots(
        ncols=2,
        figsize=(14.5, 7.5),
        gridspec_kw={"width_ratios": [3.7, 1.15], "wspace": 0.08},
    )
    ax.loglog(ai, attainable, label="Attainable roof", linewidth=2.5)
    ax.loglog(ai, memory_roof, "--", label="Memory bandwidth roof", linewidth=1.5)
    ax.axhline(compute_roof, color="tab:red", linestyle="--", label="Compute roof")
    ax.axvline(ridge, color="0.35", linestyle=":", label=f"Ridge point: {ridge:.1f} FLOP/byte")

    prefill_index = 0
    decode_index = 0
    for point in achieved_points:
        x = float(point["arithmetic_intensity"])
        y = float(point["gflops"])
        label = str(point["label"])
        attention, mode = _point_parts(label)
        color = ATTENTION_COLORS.get(attention, "0.25")
        marker = MODE_MARKERS.get(mode, "o")
        ax.scatter([x], [y], s=82, marker=marker, color=color, edgecolor="white", linewidth=0.8, zorder=5)

        if label_style != "none":
            short_label = _short_label(label) if label_style == "short" else label
            offset_index = prefill_index if mode == "prefill" else decode_index
            if mode == "prefill":
                prefill_index += 1
            else:
                decode_index += 1
            dx, dy, ha = _label_offset(offset_index, len(achieved_points), mode)
            ax.annotate(
                short_label,
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va="center",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "0.75", "alpha": 0.9},
                arrowprops={"arrowstyle": "-", "color": "0.45", "lw": 0.8, "shrinkA": 0, "shrinkB": 4},
                zorder=6,
            )

    ax.set_title(title)
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(float(np.min(ai)), max(float(np.max(ai)), max((float(p["arithmetic_intensity"]) for p in achieved_points), default=0) * 1.8))
    ax.set_ylim(
        min(float(np.min(attainable)), min((float(p["gflops"]) for p in achieved_points), default=float(np.min(attainable))) * 0.55),
        max(float(np.max(attainable)), max((float(p["gflops"]) for p in achieved_points), default=0) * 1.8),
    )

    roof_handles, roof_labels = ax.get_legend_handles_labels()
    attention_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markeredgecolor="white", markersize=9, label=name)
        for name, color in ATTENTION_COLORS.items()
    ]
    mode_handles = [
        Line2D([0], [0], marker=marker, color="0.2", linestyle="None", markersize=8, label=mode.title())
        for mode, marker in MODE_MARKERS.items()
    ]
    ax.legend(
        handles=[*roof_handles, *attention_handles, *mode_handles],
        labels=[*roof_labels, *[h.get_label() for h in attention_handles], *[h.get_label() for h in mode_handles]],
        loc="lower right",
        framealpha=0.92,
        fontsize=9,
    )

    ax_info.axis("off")
    if achieved_points:
        header = "Point        AI    GFLOP/s"
        summary = "\n".join([header, "-" * len(header), *(_format_point_summary(point) for point in achieved_points)])
        ax_info.text(
            0.02,
            0.98,
            "Overlay Points\nP = prefill, D = decode",
            ha="left",
            va="top",
            fontsize=10,
            weight="bold",
        )
        ax_info.text(
            0.02,
            0.88,
            summary,
            ha="left",
            va="top",
            fontsize=8.8,
            family="monospace",
            bbox={"boxstyle": "round,pad=0.45", "fc": "white", "ec": "0.78", "alpha": 0.96},
        )
        ax_info.text(
            0.02,
            0.21,
            "Read the plot by comparing each point\n"
            "to the diagonal memory roof and the\n"
            "horizontal compute roof. Points left of\n"
            "the ridge are usually memory-bound.",
            ha="left",
            va="top",
            fontsize=9,
            color="0.25",
        )

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.1, top=0.86, wspace=0.08)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a GPU roofline plot.")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--peak-tflops", type=float, default=None)
    parser.add_argument("--memory-bandwidth-gbps", type=float, default=None)
    parser.add_argument("--ai-min", type=float, default=0.1)
    parser.add_argument("--ai-max", type=float, default=1000.0)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--points-csv", type=Path, default=None, help="Optional achieved points CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/roofline"))
    parser.add_argument("--prefix", default="gpu_roofline")
    parser.add_argument("--title", default=None, help="Optional plot title override.")
    parser.add_argument("--label-style", choices=["short", "full", "none"], default="short")
    args = parser.parse_args()

    gpu = detect_gpu(args.gpu_index)
    spec = infer_hardware_spec(gpu.name)

    peak_tflops = args.peak_tflops
    if peak_tflops is None:
        peak_tflops = _peak_for_dtype(spec, args.dtype)
    if peak_tflops is None:
        raise SystemExit("Could not infer peak TFLOPs. Pass --peak-tflops.")

    memory_bandwidth_gbps = args.memory_bandwidth_gbps
    if memory_bandwidth_gbps is None and spec is not None:
        memory_bandwidth_gbps = spec.memory_bandwidth_gbps
    if memory_bandwidth_gbps is None:
        raise SystemExit("Could not infer memory bandwidth. Pass --memory-bandwidth-gbps.")

    roofline = build_roofline(
        peak_tflops=peak_tflops,
        memory_bandwidth_gbps=memory_bandwidth_gbps,
        ai_min=args.ai_min,
        ai_max=args.ai_max,
        samples=args.samples,
    )
    achieved_points = _load_points(args.points_csv)

    output_png = args.output_dir / f"{args.prefix}.png"
    output_json = args.output_dir / f"{args.prefix}.json"
    title_name = gpu.name or "Detected GPU"
    plot_roofline(
        roofline=roofline,
        title=args.title or f"{title_name} roofline ({args.dtype})",
        output_png=output_png,
        achieved_points=achieved_points,
        label_style=args.label_style,
    )

    report = {
        "gpu": asdict(gpu),
        "catalog_spec": None if spec is None else asdict(spec),
        "dtype": args.dtype,
        "peak_tflops": peak_tflops,
        "memory_bandwidth_gbps": memory_bandwidth_gbps,
        "roofline": roofline,
        "achieved_points": achieved_points,
        "outputs": {"png": str(output_png), "json": str(output_json)},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"png": str(output_png), "json": str(output_json)}, indent=2))


if __name__ == "__main__":
    main()
