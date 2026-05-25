"""GPU detection and hardware spec helpers.

Drivers expose useful runtime configuration, but not all theoretical roofline
inputs. Peak TFLOPs and memory bandwidth are filled from a small catalog for
common RunPod GPUs, with CLI overrides available in benchmark.roofline.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HardwareSpec:
    name_match: str
    marketing_name: str
    fp32_tflops: float
    fp16_tensor_tflops: float | None
    bf16_tensor_tflops: float | None
    memory_bandwidth_gbps: float
    notes: str


@dataclass
class GPUInfo:
    name: str | None = None
    uuid: str | None = None
    total_memory_gb: float | None = None
    driver_version: str | None = None
    cuda_runtime_version: str | None = None
    torch_version: str | None = None
    compute_capability: str | None = None
    max_sm_clock_mhz: float | None = None
    max_memory_clock_mhz: float | None = None
    power_limit_watts: float | None = None
    detected_by: list[str] | None = None
    platform: str | None = None
    query_errors: dict[str, str] | None = None


GPU_SPEC_CATALOG = [
    HardwareSpec("H100", "NVIDIA H100", 60.0, 989.0, 989.0, 3350.0, "Approximate SXM non-sparsity tensor peak."),
    HardwareSpec("A100-SXM", "NVIDIA A100 SXM", 19.5, 312.0, 312.0, 2039.0, "Approximate SXM non-sparsity tensor peak."),
    HardwareSpec("A100-PCIE", "NVIDIA A100 PCIe", 19.5, 312.0, 312.0, 1555.0, "Approximate PCIe non-sparsity tensor peak."),
    HardwareSpec("A100", "NVIDIA A100", 19.5, 312.0, 312.0, 1555.0, "A100 fallback; override bandwidth if this is an SXM part."),
    HardwareSpec("L40S", "NVIDIA L40S", 91.6, 362.0, 362.0, 864.0, "Approximate non-sparsity tensor peak."),
    HardwareSpec("RTX 6000 Ada", "NVIDIA RTX 6000 Ada", 91.1, 364.0, 364.0, 960.0, "Approximate non-sparsity tensor peak."),
    HardwareSpec("RTX 4090", "NVIDIA GeForce RTX 4090", 82.6, 330.0, 330.0, 1008.0, "Approximate non-sparsity tensor peak."),
    HardwareSpec("A10G", "NVIDIA A10G", 31.2, 125.0, 125.0, 600.0, "Approximate non-sparsity tensor peak."),
    HardwareSpec("A10", "NVIDIA A10", 31.2, 125.0, 125.0, 600.0, "Approximate non-sparsity tensor peak."),
    HardwareSpec("L4", "NVIDIA L4", 30.3, 121.0, 121.0, 300.0, "Approximate non-sparsity tensor peak."),
    HardwareSpec("T4", "NVIDIA T4", 8.1, 65.0, None, 320.0, "Approximate FP16 tensor peak."),
    HardwareSpec("V100", "NVIDIA V100", 15.7, 125.0, None, 900.0, "Approximate SXM-class values; override for PCIe if needed."),
]


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.strip().replace(" W", "").replace(" MHz", "").replace(" MiB", "")
    if cleaned in {"", "[N/A]", "N/A"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _query_torch(index: int) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on environment
        return {"error": f"torch unavailable: {exc}"}

    details: dict[str, Any] = {
        "torch_version": getattr(torch, "__version__", None),
        "cuda_runtime_version": getattr(torch.version, "cuda", None),
    }
    if not torch.cuda.is_available():
        details["error"] = "torch.cuda is not available"
        return details

    props = torch.cuda.get_device_properties(index)
    details.update(
        {
            "name": props.name,
            "total_memory_gb": props.total_memory / (1024**3),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        }
    )
    return details


def _query_nvidia_smi(index: int) -> dict[str, Any]:
    fields = [
        "name",
        "uuid",
        "driver_version",
        "memory.total",
        "power.limit",
        "clocks.max.sm",
        "clocks.max.memory",
    ]
    cmd = [
        "nvidia-smi",
        f"--id={index}",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception as exc:  # pragma: no cover - depends on environment
        return {"error": f"nvidia-smi unavailable: {exc}"}

    values = [part.strip() for part in result.stdout.strip().split(",")]
    if len(values) != len(fields):
        return {"error": f"unexpected nvidia-smi output: {result.stdout.strip()}"}

    raw = dict(zip(fields, values, strict=True))
    memory_mib = _parse_float(raw.get("memory.total"))
    return {
        "name": raw.get("name"),
        "uuid": raw.get("uuid"),
        "driver_version": raw.get("driver_version"),
        "total_memory_gb": None if memory_mib is None else memory_mib / 1024,
        "power_limit_watts": _parse_float(raw.get("power.limit")),
        "max_sm_clock_mhz": _parse_float(raw.get("clocks.max.sm")),
        "max_memory_clock_mhz": _parse_float(raw.get("clocks.max.memory")),
    }


def detect_gpu(index: int = 0) -> GPUInfo:
    """Detect the active GPU using torch and nvidia-smi when available."""

    detected_by: list[str] = []
    query_errors: dict[str, str] = {}
    merged: dict[str, Any] = {"platform": platform.platform()}

    torch_details = _query_torch(index)
    if "error" not in torch_details:
        detected_by.append("torch")
        merged.update({k: v for k, v in torch_details.items() if v is not None})
    else:
        query_errors["torch"] = torch_details["error"]

    smi_details = _query_nvidia_smi(index)
    if "error" not in smi_details:
        detected_by.append("nvidia-smi")
        merged.update({k: v for k, v in smi_details.items() if v is not None})
    else:
        query_errors["nvidia-smi"] = smi_details["error"]

    return GPUInfo(
        name=merged.get("name"),
        uuid=merged.get("uuid"),
        total_memory_gb=merged.get("total_memory_gb"),
        driver_version=merged.get("driver_version"),
        cuda_runtime_version=merged.get("cuda_runtime_version"),
        torch_version=merged.get("torch_version"),
        compute_capability=merged.get("compute_capability"),
        max_sm_clock_mhz=merged.get("max_sm_clock_mhz"),
        max_memory_clock_mhz=merged.get("max_memory_clock_mhz"),
        power_limit_watts=merged.get("power_limit_watts"),
        detected_by=detected_by,
        platform=merged.get("platform"),
        query_errors=query_errors or None,
    )


def infer_hardware_spec(gpu_name: str | None) -> HardwareSpec | None:
    """Return a catalog spec for the detected GPU name when possible."""

    if not gpu_name:
        return None

    normalized_name = gpu_name.upper().replace("NVIDIA", "").strip()
    if "A100" in normalized_name and ("SXM" in normalized_name or "80GB" in normalized_name):
        return GPU_SPEC_CATALOG[1]

    for spec in GPU_SPEC_CATALOG:
        if spec.name_match.upper() in normalized_name:
            return spec
    return None


def gpu_report(index: int = 0) -> dict[str, Any]:
    gpu = detect_gpu(index)
    spec = infer_hardware_spec(gpu.name)
    return {
        "gpu": asdict(gpu),
        "catalog_spec": None if spec is None else asdict(spec),
        "catalog_warning": None
        if spec is not None
        else "No matching spec found. Pass roofline overrides for peak TFLOPs and bandwidth.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect GPU configuration for benchmarking.")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    report = gpu_report(args.gpu_index)
    text = json.dumps(report, indent=2)
    print(text)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
