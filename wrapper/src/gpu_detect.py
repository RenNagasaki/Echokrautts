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
    backend: str  # "cuda" | "rocm" | "rocm_win" | "dml" | "xpu" | "cpu"
    device: str  # torch device string: "cuda" | "dml" | "xpu" | "cpu"
    torch_index_url: str
    extra_packages: list[str] = field(default_factory=list)
    max_workers_hint: int = 1
    free_vram_gb: float | None = None
    detail: str = ""
    # Set only by backends that install from individual wheel URLs instead of a
    # pip index — currently just native-Windows ROCm, where AMD publishes no
    # index. ``python_version`` overrides the configured interpreter for the
    # venv, because those wheels are built for exactly one Python.
    wheel_urls: list[str] = field(default_factory=list)
    torch_wheel_urls: list[str] = field(default_factory=list)
    python_version: str | None = None
    torch_version: str | None = None


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


def _gpu_names() -> list[str]:
    """Display-adapter names (Windows only; empty elsewhere or on failure)."""
    if not procutil.IS_WINDOWS:
        return []
    out = procutil.try_run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
        ]
    )
    return [line.strip() for line in (out or "").splitlines() if line.strip()]


def rocm_windows_candidate(config: Config) -> str | None:
    """Return the GPU name if native-Windows ROCm applies to this machine.

    AMD supports ROCm on Windows for a subset of its cards (Radeon 9000 series
    and select 7000), so the adapter name is matched against
    ``config.rocm_windows["gpu_pattern"]``. Matching by name is a heuristic —
    but the alternative, installing a multi-GB ROCm stack on every AMD machine
    and finding out at first inference, is worse. A card that does not match
    keeps the existing DirectML→CPU path.
    """
    settings = config.rocm_windows or {}
    pattern = settings.get("gpu_pattern")
    if not pattern or not settings.get("torch_wheels"):
        return None
    for name in _gpu_names():
        if re.search(pattern, name, re.IGNORECASE):
            return name
    return None


def _rocm_windows_detection(config: Config, gpu_name: str) -> Detection:
    settings = config.rocm_windows
    return Detection(
        backend="rocm_win",
        # HIP presents itself as CUDA, on Windows as much as on Linux.
        device="cuda",
        # Nothing to point an index at — AMD ships individual wheels. The URL is
        # kept only so the field is never empty for callers that log it.
        torch_index_url=TORCH_INDEX["cpu"],
        wheel_urls=list(settings.get("wheels", [])),
        torch_wheel_urls=list(settings.get("torch_wheels", [])),
        python_version=settings.get("python"),
        torch_version=settings.get("torch_version"),
        detail=f"AMD on Windows → ROCm ({gpu_name})",
    )


def _detect_amd(config: Config) -> Detection | None:
    if not _has_amd_gpu():
        return None
    if procutil.IS_WINDOWS:
        gpu_name = rocm_windows_candidate(config)
        if gpu_name:
            return _rocm_windows_detection(config, gpu_name)
        # Not on AMD's Windows-ROCm list → DirectML, as before. DirectML is in
        # maintenance mode upstream and its op coverage does not carry these
        # models, so the engine self-tests it and falls back to CPU (SPEC §4.2).
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


# Backends that can be forced via ``config.gpu_backend``, with the device and
# wheel index each implies. Kept as data so a new backend needs one entry, not a
# branch. ROCm reports itself as a CUDA device (HIP masquerades as CUDA).
FORCED_BACKENDS = {
    "cuda": ("cuda", "cu128", []),
    "rocm": ("cuda", "rocm", []),
    # rocm_win is handled separately in detect_backend: its install comes from
    # config (wheel URLs, Python version), not from this table.
    "rocm_win": ("cuda", "cpu", []),
    "dml": ("dml", "dml", ["torch-directml"]),
    "xpu": ("xpu", "xpu", ["intel-extension-for-pytorch"]),
    "cpu": ("cpu", "cpu", []),
}


def _forced_detection(backend: str) -> Detection:
    """Build a Detection for an explicitly configured backend (no probing).

    Free VRAM stays unknown here — the probe that would have reported it is
    exactly what the caller opted out of — so ``_apply_worker_hint`` falls back
    to its conservative default. Set ``max_workers`` if you want more.
    """
    device, index_key, extras = FORCED_BACKENDS[backend]
    return Detection(
        backend=backend,
        device=device,
        torch_index_url=TORCH_INDEX[index_key],
        extra_packages=list(extras),
        detail=f"{backend} forced by config (gpu_backend={backend})",
    )


def detect_backend(config: Config) -> Detection:
    """Run the full detection chain and return the chosen backend.

    ``config.gpu_backend`` short-circuits the chain. An unknown value is a hard
    error rather than a silent fallback: it is always a typo, and answering a
    misspelled "rocm" with a CPU pool looks like the wrapper simply being slow.
    """
    forced = (config.gpu_backend or "auto").strip().lower()
    if forced != "auto":
        if forced not in FORCED_BACKENDS:
            raise ValueError(
                f"gpu_backend={config.gpu_backend!r} is not one of "
                f"'auto', {', '.join(sorted(FORCED_BACKENDS))}"
            )
        det = (
            _rocm_windows_detection(config, "forced")
            if forced == "rocm_win"
            else _forced_detection(forced)
        )
    else:
        det = _detect_nvidia() or _detect_amd(config) or _detect_intel()
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
