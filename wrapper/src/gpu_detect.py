"""GPU detection and PyTorch backend/wheel selection (SPEC §4).

Detection order; first match wins: NVIDIA (CUDA) → AMD (ROCm/DirectML) →
Intel (XPU) → CPU. Returns a :class:`Detection` describing the backend, the
torch device string, the pip ``--index-url`` for the torch wheels, any extra
packages (e.g. ``torch-directml``/IPEX), and a VRAM-derived worker hint.

The actual wheel URLs drift over time (SPEC §3 note); they are centralized in
``TORCH_INDEX`` so they are easy to bump. Selection *logic* is what matters here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from . import procutil
from .config import Config

# Backend key → torch wheel index URL. Verify against pytorch.org when bumping.
TORCH_INDEX = {
    "cu128": "https://download.pytorch.org/whl/cu128",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "rocm": "https://download.pytorch.org/whl/rocm6.3",
    "xpu": "https://download.pytorch.org/whl/xpu",
    "cpu": "https://download.pytorch.org/whl/cpu",
    # DirectML uses CPU-build torch + the torch-directml package.
    "dml": "https://download.pytorch.org/whl/cpu",
}


@dataclass
class Detection:
    backend: str  # "cuda" | "rocm" | "dml" | "xpu" | "cpu"
    device: str  # torch device string: "cuda" | "dml" | "xpu" | "cpu"
    torch_index_url: str
    extra_packages: list[str] = field(default_factory=list)
    max_workers_hint: int = 1
    free_vram_gb: float | None = None
    detail: str = ""


# --------------------------------------------------------------------- probes
def _nvidia_query(field_name: str) -> list[str]:
    """Return per-GPU values for an ``nvidia-smi --query-gpu`` field."""
    out = procutil.try_run(
        ["nvidia-smi", f"--query-gpu={field_name}", "--format=csv,noheader,nounits"]
    )
    if not out:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _detect_nvidia() -> Detection | None:
    caps = _nvidia_query("compute_cap")
    if not caps:
        return None
    # Highest compute capability across visible GPUs decides the wheel.
    try:
        max_cap = max(float(c) for c in caps)
    except ValueError:
        max_cap = 0.0
    backend_key = "cu128" if max_cap >= 12.0 else "cu126"

    free_gb = None
    frees = _nvidia_query("memory.free")  # MiB
    if frees:
        try:
            free_gb = max(float(f) for f in frees) / 1024.0
        except ValueError:
            free_gb = None

    return Detection(
        backend="cuda",
        device="cuda",
        torch_index_url=TORCH_INDEX[backend_key],
        free_vram_gb=free_gb,
        detail=f"NVIDIA compute_cap={max_cap} → {backend_key}",
    )


def _has_amd_gpu() -> bool:
    if procutil.IS_WINDOWS:
        return _pci_vendor_present("1002")
    # Linux: rocminfo present and succeeds, or /opt/rocm exists.
    import os

    if os.path.isdir("/opt/rocm"):
        return True
    return procutil.try_run(["rocminfo"]) is not None


def _detect_amd() -> Detection | None:
    if not _has_amd_gpu():
        return None
    if procutil.IS_WINDOWS:
        # DirectML: CPU torch + torch-directml. Op coverage is limited, so the
        # engine runs a self-test and falls back to CPU if it fails (SPEC §4.2).
        return Detection(
            backend="dml",
            device="dml",
            torch_index_url=TORCH_INDEX["dml"],
            extra_packages=["torch-directml"],
            max_workers_hint=1,
            detail="AMD on Windows → DirectML",
        )
    # Linux ROCm: HIP masquerades as a CUDA device.
    return Detection(
        backend="rocm",
        device="cuda",
        torch_index_url=TORCH_INDEX["rocm"],
        detail="AMD on Linux → ROCm",
    )


def _detect_intel() -> Detection | None:
    if not _pci_vendor_present("8086", require_dgpu=True):
        return None
    return Detection(
        backend="xpu",
        device="xpu",
        torch_index_url=TORCH_INDEX["xpu"],
        extra_packages=["intel-extension-for-pytorch"],
        max_workers_hint=1,
        detail="Intel dGPU → XPU (best-effort)",
    )


def _pci_vendor_present(vendor_hex: str, require_dgpu: bool = False) -> bool:
    """Best-effort check for a PCI display controller by vendor id.

    ``vendor_hex`` like ``"1002"`` (AMD) or ``"8086"`` (Intel). On Windows we
    query video controllers via PowerShell; on Linux we read ``lspci``.
    """
    vendor = vendor_hex.lower()
    if procutil.IS_WINDOWS:
        out = procutil.try_run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object -ExpandProperty PNPDeviceID",
            ]
        )
        if not out:
            return False
        return bool(re.search(rf"VEN_{vendor}", out, re.IGNORECASE))
    out = procutil.try_run(["lspci", "-nn"])
    if not out:
        return False
    for line in out.splitlines():
        low = line.lower()
        if "vga" not in low and "3d" not in low and "display" not in low:
            continue
        if f"[{vendor}:" in low:
            if require_dgpu and "integrated" in low:
                continue
            return True
    return False


# ------------------------------------------------------------------ assemble
def _apply_worker_hint(det: Detection, config: Config) -> Detection:
    """Compute ``max_workers_hint`` from free VRAM (SPEC §4.5)."""
    cfg_max = config.max_workers if config.max_workers else 4
    if det.backend in ("dml", "xpu", "cpu"):
        det.max_workers_hint = 1
    elif det.free_vram_gb is not None:
        usable = det.free_vram_gb - config.vram_reserve_gb
        per = max(config.per_job_gb, 0.1)
        hint = math.floor(usable / per)
        det.max_workers_hint = max(1, min(hint, cfg_max))
    else:
        det.max_workers_hint = max(1, min(2, cfg_max))
    return det


def detect_backend(config: Config) -> Detection:
    """Run the full detection chain and return the chosen backend."""
    det = _detect_nvidia() or _detect_amd() or _detect_intel()
    if det is None:
        det = Detection(
            backend="cpu",
            device="cpu",
            torch_index_url=TORCH_INDEX["cpu"],
            max_workers_hint=1,
            detail="No usable GPU detected → CPU",
        )
    if config.torch_index_override:
        det.torch_index_url = config.torch_index_override
        det.detail += f" (torch index overridden: {config.torch_index_override})"
    return _apply_worker_hint(det, config)
