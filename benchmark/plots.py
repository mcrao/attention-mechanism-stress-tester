"""Plot benchmark result summaries."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "attention_stress_matplotlib"))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D


ATTENTION_COLORS = {
    "MHA": "tab:blue",
    "MQA": "tab:orange",
    "GQA": "tab:green",
    "MLA": "tab:red",
}


# ── batch-scaling ────────────────────────────────────────────────────────────────

def _load_decode_rows(results_dir: Path, contexts: list[int], batches: list[int]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for context in contexts:
        for batch in batches:
            if batch == 1:
                path = results_dir / f"c{context}_b1_fp16.csv"
            else:
                path = results_dir / f"c{context}_b{batch}_fp16_decode.csv"
            if not path.exists():
                continue

            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    rows.append(
                        {
                            "context": context,
                            "batch": batch,
                            "attention": row["attention"].upper(),
                            "decode_tps": float(row["decode_tokens_per_sec"]),
                            "decode_ms": float(row["decode_ms_per_token"]),
                            "kv_gib": float(row["estimated_kv_cache_gib"]),
                        }
                    )
    return rows


def plot_batch_scaling(*, rows: list[dict[str, float | int | str]], output: Path) -> None:
    contexts = sorted({int(row["context"]) for row in rows})
    fig, axes = plt.subplots(ncols=len(contexts), figsize=(6.4 * len(contexts), 4.8), sharey=False)
    if len(contexts) == 1:
        axes = [axes]

    for ax, context in zip(axes, contexts, strict=True):
        context_rows = [row for row in rows if int(row["context"]) == context]
        for attention, color in ATTENTION_COLORS.items():
            attn_rows = sorted(
                [row for row in context_rows if str(row["attention"]) == attention],
                key=lambda row: int(row["batch"]),
            )
            if not attn_rows:
                continue
            batches = [int(row["batch"]) for row in attn_rows]
            tps = [float(row["decode_tps"]) for row in attn_rows]
            ax.plot(batches, tps, marker="o", linewidth=2.2, color=color, label=attention)

            best_index = max(range(len(tps)), key=tps.__getitem__)
            ax.scatter([batches[best_index]], [tps[best_index]], s=95, color=color, edgecolor="black", zorder=5)

        ax.set_title(f"Context {context}")
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Decode tokens/sec")
        ax.set_xticks(sorted({int(row["batch"]) for row in context_rows}))
        ax.grid(True, alpha=0.28)

    handles, labels = axes[-1].get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markeredgecolor="black",
            markerfacecolor="white",
            markersize=8,
            label="best point",
        )
    )
    labels.append("best point")
    axes[-1].legend(handles, labels, loc="best")
    fig.suptitle("FP16 Decode Batch Scaling", fontsize=14)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


# ── shape sweep ──────────────────────────────────────────────────────────────────

def _load_shape_sweep_data(
    results_dir: Path,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, str] | None,
    dict[str, str] | None,
]:
    """Return (gqa_rows, mla_rows, mqa_ref, mha_ref) for the shape sweep."""

    def _read_attention(path: Path, attention: str) -> dict[str, str] | None:
        if not path.exists():
            return None
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row["attention"] == attention:
                    return row
        return None

    def _read_first(path: Path) -> dict[str, str] | None:
        if not path.exists():
            return None
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        return rows[0] if rows else None

    base = results_dir / "c8192_b4_fp16_decode.csv"

    gqa_rows: list[dict[str, object]] = []
    for group_size in (2, 4, 8):
        raw = (
            _read_attention(base, "gqa")
            if group_size == 4
            else _read_first(results_dir / f"c8192_b4_gqa_g{group_size}_fp16_decode.csv")
        )
        if raw is None:
            continue
        gqa_rows.append(
            {
                "label": f"GQA g{group_size}",
                "sublabel": f"kv_heads={int(raw['kv_heads'])}",
                "group_size": group_size,
                "kv_gib": float(raw["estimated_kv_cache_gib"]),
                "decode_tps": float(raw["decode_tokens_per_sec"]),
            }
        )
    gqa_rows.sort(key=lambda r: float(r["kv_gib"]))  # ascending budget

    mla_rows: list[dict[str, object]] = []
    for latent_dim in (128, 256, 512):
        raw = (
            _read_attention(base, "mla")
            if latent_dim == 256
            else _read_first(results_dir / f"c8192_b4_mla_l{latent_dim}_fp16_decode.csv")
        )
        if raw is None:
            continue
        mla_rows.append(
            {
                "label": f"MLA l{latent_dim}",
                "sublabel": f"latent={latent_dim}",
                "latent_dim": latent_dim,
                "kv_gib": float(raw["estimated_kv_cache_gib"]),
                "decode_tps": float(raw["decode_tokens_per_sec"]),
            }
        )
    mla_rows.sort(key=lambda r: float(r["kv_gib"]))

    return gqa_rows, mla_rows, _read_attention(base, "mqa"), _read_attention(base, "mha")


def plot_shape_sweep(
    *,
    gqa_rows: list[dict[str, object]],
    mla_rows: list[dict[str, object]],
    mqa_ref: dict[str, str] | None,
    mha_ref: dict[str, str] | None,
    output: Path,
) -> None:
    GQA_COLOR = ATTENTION_COLORS["GQA"]   # tab:green
    MLA_COLOR = ATTENTION_COLORS["MLA"]   # tab:red
    MQA_COLOR = ATTENTION_COLORS["MQA"]   # tab:orange
    MHA_COLOR = ATTENTION_COLORS["MHA"]   # tab:blue

    mqa_x = float(mqa_ref["estimated_kv_cache_gib"]) if mqa_ref else None
    mqa_y = float(mqa_ref["decode_tokens_per_sec"]) if mqa_ref else None
    mha_x = float(mha_ref["estimated_kv_cache_gib"]) if mha_ref else None
    mha_y = float(mha_ref["decode_tokens_per_sec"]) if mha_ref else None

    fig, ax = plt.subplots(figsize=(11, 7))

    # ── GQA sweep line ────────────────────────────────────────────────────────────
    if gqa_rows:
        gx = [float(r["kv_gib"]) for r in gqa_rows]
        gy = [float(r["decode_tps"]) for r in gqa_rows]
        ax.plot(
            gx, gy,
            marker="o", linestyle="-", color=GQA_COLOR,
            linewidth=2.4, markersize=10, zorder=4,
            markeredgecolor="white", markeredgewidth=1.0,
            label="GQA  (group size sweep)",
            solid_capstyle="round",
        )

    # ── MLA sweep line ────────────────────────────────────────────────────────────
    if mla_rows:
        mx = [float(r["kv_gib"]) for r in mla_rows]
        my = [float(r["decode_tps"]) for r in mla_rows]
        ax.plot(
            mx, my,
            marker="s", linestyle="--", color=MLA_COLOR,
            linewidth=2.4, markersize=10, zorder=4,
            markeredgecolor="white", markeredgewidth=1.0,
            label="MLA  (latent dim sweep)",
            dash_capstyle="round",
        )

    # ── Reference markers ─────────────────────────────────────────────────────────
    if mqa_x is not None:
        ax.scatter(
            [mqa_x], [mqa_y],
            marker="*", s=380, color=MQA_COLOR, zorder=5,
            edgecolors="#b85c00", linewidths=0.7,
            label="MQA  (reference)",
        )

    if mha_x is not None:
        ax.scatter(
            [mha_x], [mha_y],
            marker="D", s=110, color=MHA_COLOR, zorder=5,
            edgecolors="white", linewidths=0.9,
            label="MHA  (reference)",
        )

    # ── GQA point labels ──────────────────────────────────────────────────────────
    # Offsets: (dx_pt, dy_pt, ha, va)
    _GQA_OFFSETS = {
        2: (10, -16, "left", "top"),    # g2 is lowest; label below
        4: (10,  12, "left", "bottom"),
        8: (10,  12, "left", "bottom"),
    }
    for r in gqa_rows:
        gs = int(r["group_size"])
        dx, dy, ha, va = _GQA_OFFSETS.get(gs, (10, 12, "left", "bottom"))
        ax.annotate(
            f"{r['label']}  ({r['sublabel']})",
            xy=(float(r["kv_gib"]), float(r["decode_tps"])),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9.5, color=GQA_COLOR,
            fontweight="semibold", ha=ha, va=va,
        )

    # ── MLA point labels ──────────────────────────────────────────────────────────
    # l256 and l512 are deliberately placed below-left to pair visually with
    # the GQA labels above-right at the same x positions.
    _MLA_OFFSETS = {
        128: ( 10,  12, "left",  "bottom"),  # near left edge: go right to avoid clipping
        256: (-10, -16, "right", "top"),
        512: (-10, -16, "right", "top"),
    }
    for r in mla_rows:
        ld = int(r["latent_dim"])
        dx, dy, ha, va = _MLA_OFFSETS.get(ld, (-10, 12, "right", "bottom"))
        ax.annotate(
            f"{r['label']}  ({r['sublabel']})",
            xy=(float(r["kv_gib"]), float(r["decode_tps"])),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9.5, color=MLA_COLOR,
            fontweight="semibold", ha=ha, va=va,
        )

    # ── MQA label ─────────────────────────────────────────────────────────────────
    if mqa_x is not None:
        ax.annotate(
            f"MQA\n{mqa_y:,.0f} tok/s",
            xy=(mqa_x, mqa_y),
            xytext=(14, 0),
            textcoords="offset points",
            fontsize=9.5, color=MQA_COLOR,
            fontweight="semibold", ha="left", va="center",
        )

    # ── MHA label ─────────────────────────────────────────────────────────────────
    if mha_x is not None:
        ax.annotate(
            f"MHA\n{mha_y:,.0f} tok/s",
            xy=(mha_x, mha_y),
            xytext=(-14, 10),
            textcoords="offset points",
            fontsize=9.5, color=MHA_COLOR,
            fontweight="semibold", ha="right", va="bottom",
        )

    # ── Same-budget gap brackets (0.75 GiB and 1.5 GiB) ──────────────────────────
    gqa_by_budget = {float(r["kv_gib"]): r for r in gqa_rows}
    mla_by_budget = {float(r["kv_gib"]): r for r in mla_rows}

    for budget in (0.75, 1.5):
        if budget not in gqa_by_budget or budget not in mla_by_budget:
            continue
        g_tps = float(gqa_by_budget[budget]["decode_tps"])
        m_tps = float(mla_by_budget[budget]["decode_tps"])
        pct = (g_tps / m_tps - 1) * 100
        geom_mid = (g_tps * m_tps) ** 0.5       # geometric mean on log scale
        bracket_x = budget * 1.60               # offset right of data points
        text_x = bracket_x * 1.10

        # vertical bracket line with end caps
        ax.plot([bracket_x, bracket_x], [m_tps, g_tps],
                color="#888888", lw=1.6, zorder=3, solid_capstyle="round")
        cap = bracket_x * 0.08
        for end_y in (g_tps, m_tps):
            ax.plot([bracket_x - cap, bracket_x + cap], [end_y, end_y],
                    color="#888888", lw=1.6, zorder=3, solid_capstyle="round")

        # GQA label (green) flush with top endpoint, growing upward
        ax.text(text_x, g_tps, "GQA",
                fontsize=8.5, color=GQA_COLOR, fontweight="semibold",
                va="bottom", ha="left")
        # percentage gap in the middle
        ax.text(text_x, geom_mid, f"+{pct:.0f}%",
                fontsize=8.5, color="#444444", style="italic",
                va="center", ha="left")
        # MLA label (red) flush with bottom endpoint, growing downward
        ax.text(text_x, m_tps, "MLA",
                fontsize=8.5, color=MLA_COLOR, fontweight="semibold",
                va="top", ha="left")

    # ── Axes and scales ───────────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_yscale("log")

    x_ticks = [0.375, 0.75, 1.5, 3.0, 6.0]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(["0.375", "0.75", "1.5", "3.0", "6.0"], fontsize=11)
    ax.xaxis.set_minor_locator(mticker.NullLocator())

    y_ticks = [3000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{t:,}" for t in y_ticks], fontsize=11)
    ax.yaxis.set_minor_locator(mticker.NullLocator())

    ax.set_xlim(0.30, 12.0)
    ax.set_ylim(2900, 26000)

    ax.set_xlabel("Estimated KV cache — 24-layer model (GiB)", fontsize=12, labelpad=8)
    ax.set_ylabel("Decode throughput (tokens / sec)", fontsize=12, labelpad=8)
    ax.set_title(
        "GQA / MLA Shape Sweep — Throughput vs KV Cache Budget\n"
        "context 8192  ·  batch 4  ·  FP16  ·  decode  ·  RTX 4090",
        fontsize=13,
        pad=12,
        linespacing=1.5,
    )

    # ── Grid ──────────────────────────────────────────────────────────────────────
    ax.grid(True, which="major", color="#e0e0e0", linewidth=0.8, linestyle="-", zorder=0)

    # ── Legend ────────────────────────────────────────────────────────────────────
    ax.legend(
        loc="upper right",
        fontsize=10.5,
        framealpha=0.95,
        edgecolor="#cccccc",
        handlelength=2.2,
        handletextpad=0.7,
        borderpad=0.9,
        labelspacing=0.6,
    )

    fig.tight_layout(pad=1.6)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark summaries.")
    parser.add_argument("--plot", choices=["batch-scaling", "shape-sweep"], required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results/benchmarks"))
    # batch-scaling only
    parser.add_argument("--contexts", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default depends on plot type).",
    )
    args = parser.parse_args()

    if args.plot == "batch-scaling":
        out = args.output or Path("results/plots/batch_scaling_fp16_decode.png")
        rows = _load_decode_rows(args.results_dir, args.contexts, args.batches)
        if not rows:
            raise SystemExit("No matching benchmark rows found.")
        plot_batch_scaling(rows=rows, output=out)
        print(str(out))

    elif args.plot == "shape-sweep":
        out = args.output or Path("results/plots/shape_sweep_c8192_b4_fp16_decode.png")
        gqa_rows, mla_rows, mqa_ref, mha_ref = _load_shape_sweep_data(args.results_dir)
        if not gqa_rows and not mla_rows:
            raise SystemExit("No shape-sweep result files found in results-dir.")
        plot_shape_sweep(
            gqa_rows=gqa_rows,
            mla_rows=mla_rows,
            mqa_ref=mqa_ref,
            mha_ref=mha_ref,
            output=out,
        )
        print(str(out))


if __name__ == "__main__":
    main()
