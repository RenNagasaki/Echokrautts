import pytest

from src import gpu_detect
from src.config import Config


@pytest.fixture
def cfg():
    return Config(max_workers=4, vram_reserve_gb=1.5, per_job_gb=3.0)


def _no_amd_no_intel(monkeypatch):
    monkeypatch.setattr(gpu_detect, "_has_amd_gpu", lambda: False)
    monkeypatch.setattr(gpu_detect, "_pci_vendor_present", lambda *a, **k: False)


def test_nvidia_blackwell_picks_cu128(monkeypatch, cfg):
    monkeypatch.setattr(
        gpu_detect,
        "_nvidia_query",
        lambda field: ["12.0"] if field == "compute_cap" else ["24000"],
    )
    det = gpu_detect.detect_backend(cfg)
    assert det.backend == "cuda"
    assert det.device == "cuda"
    assert det.torch_index_url.endswith("cu128")


def test_nvidia_ampere_picks_cu126(monkeypatch, cfg):
    monkeypatch.setattr(
        gpu_detect,
        "_nvidia_query",
        lambda field: ["8.6"] if field == "compute_cap" else ["8000"],
    )
    det = gpu_detect.detect_backend(cfg)
    assert det.torch_index_url.endswith("cu126")


def test_vram_clamp(monkeypatch, cfg):
    # 24 GB free, reserve 1.5, per-job 3 → floor(22.5/3)=7, clamped to cfg_max=4.
    monkeypatch.setattr(
        gpu_detect,
        "_nvidia_query",
        lambda field: ["8.9"] if field == "compute_cap" else ["24576"],
    )
    det = gpu_detect.detect_backend(cfg)
    assert det.max_workers_hint == 4


def test_no_gpu_falls_back_to_cpu(monkeypatch, cfg):
    monkeypatch.setattr(gpu_detect, "_nvidia_query", lambda field: [])
    _no_amd_no_intel(monkeypatch)
    det = gpu_detect.detect_backend(cfg)
    assert det.backend == "cpu"
    assert det.device == "cpu"
    assert det.max_workers_hint == 1


def test_amd_linux_rocm(monkeypatch, cfg):
    monkeypatch.setattr(gpu_detect, "_nvidia_query", lambda field: [])
    monkeypatch.setattr(gpu_detect.procutil, "IS_WINDOWS", False)
    monkeypatch.setattr(gpu_detect, "_has_amd_gpu", lambda: True)
    det = gpu_detect.detect_backend(cfg)
    assert det.backend == "rocm"
    assert det.device == "cuda"  # HIP masquerades as CUDA


def test_amd_windows_directml(monkeypatch, cfg):
    monkeypatch.setattr(gpu_detect, "_nvidia_query", lambda field: [])
    monkeypatch.setattr(gpu_detect.procutil, "IS_WINDOWS", True)
    monkeypatch.setattr(gpu_detect, "_has_amd_gpu", lambda: True)
    det = gpu_detect.detect_backend(cfg)
    assert det.backend == "dml"
    assert det.device == "dml"
    assert "torch-directml" in det.extra_packages
    assert det.max_workers_hint == 1


def test_intel_xpu(monkeypatch, cfg):
    monkeypatch.setattr(gpu_detect, "_nvidia_query", lambda field: [])
    monkeypatch.setattr(gpu_detect, "_has_amd_gpu", lambda: False)
    monkeypatch.setattr(
        gpu_detect, "_pci_vendor_present", lambda vendor, require_dgpu=False: vendor == "8086"
    )
    det = gpu_detect.detect_backend(cfg)
    assert det.backend == "xpu"
    assert det.device == "xpu"


def test_torch_index_override(monkeypatch):
    cfg = Config(torch_index_override="https://example/whl/custom")
    monkeypatch.setattr(gpu_detect, "_nvidia_query", lambda field: [])
    _no_amd_no_intel(monkeypatch)
    det = gpu_detect.detect_backend(cfg)
    assert det.torch_index_url == "https://example/whl/custom"


# --------------------------------------------------------------- gpu_backend
def _probes_explode(monkeypatch):
    """Make every probe fail the test if it runs — forcing must not probe."""

    def boom(*_a, **_k):
        raise AssertionError("detection probe ran despite gpu_backend override")

    monkeypatch.setattr(gpu_detect, "_nvidia_query", boom)
    monkeypatch.setattr(gpu_detect, "_has_amd_gpu", boom)
    monkeypatch.setattr(gpu_detect, "_pci_vendor_present", boom)


def test_forced_rocm_skips_probing(monkeypatch, cfg):
    # The ROCm container case: no rocminfo, no /opt/rocm, no nvidia-smi — the
    # probes would answer "CPU" and the GPU would sit idle.
    _probes_explode(monkeypatch)
    cfg.gpu_backend = "rocm"
    det = gpu_detect.detect_backend(cfg)
    assert det.backend == "rocm"
    assert det.device == "cuda"  # HIP masquerades as CUDA
    assert det.torch_index_url == gpu_detect.TORCH_INDEX["rocm"]


def test_forced_cpu_beats_a_visible_nvidia_gpu(monkeypatch, cfg):
    # The inverse trap: a CPU-only build handed --gpus all still sees nvidia-smi.
    monkeypatch.setattr(
        gpu_detect,
        "_nvidia_query",
        lambda field: ["12.0"] if field == "compute_cap" else ["24000"],
    )
    cfg.gpu_backend = "cpu"
    det = gpu_detect.detect_backend(cfg)
    assert det.backend == "cpu"
    assert det.device == "cpu"


def test_forced_backend_is_case_and_space_insensitive(monkeypatch, cfg):
    _probes_explode(monkeypatch)
    cfg.gpu_backend = "  ROCm "
    assert gpu_detect.detect_backend(cfg).backend == "rocm"


def test_forced_dml_carries_its_extra_package(monkeypatch, cfg):
    _probes_explode(monkeypatch)
    cfg.gpu_backend = "dml"
    det = gpu_detect.detect_backend(cfg)
    assert det.extra_packages == ["torch-directml"]
    # The shared default list must not be mutated by a caller appending to it.
    det.extra_packages.append("mutated")
    assert gpu_detect.detect_backend(cfg).extra_packages == ["torch-directml"]


def test_unknown_forced_backend_raises(monkeypatch, cfg):
    _probes_explode(monkeypatch)
    cfg.gpu_backend = "rocmm"
    with pytest.raises(ValueError, match="rocmm"):
        gpu_detect.detect_backend(cfg)


def test_forced_backend_still_honours_index_override(monkeypatch):
    _probes_explode(monkeypatch)
    cfg = Config(gpu_backend="rocm", torch_index_override="https://example/whl/custom")
    assert gpu_detect.detect_backend(cfg).torch_index_url == "https://example/whl/custom"


def test_auto_is_the_default_and_still_probes(monkeypatch, cfg):
    assert Config().gpu_backend == "auto"
    monkeypatch.setattr(gpu_detect, "_nvidia_query", lambda field: [])
    _no_amd_no_intel(monkeypatch)
    assert gpu_detect.detect_backend(cfg).backend == "cpu"


# ----------------------------------------------------- native Windows ROCm
def _windows_amd(monkeypatch, gpu_names):
    monkeypatch.setattr(gpu_detect, "_nvidia_query", lambda field: [])
    monkeypatch.setattr(gpu_detect.procutil, "IS_WINDOWS", True)
    monkeypatch.setattr(gpu_detect, "_has_amd_gpu", lambda: True)
    monkeypatch.setattr(gpu_detect, "_gpu_names", lambda: gpu_names)


def test_windows_amd_supported_card_picks_rocm(monkeypatch, cfg):
    _windows_amd(monkeypatch, ["AMD Radeon RX 7900 XTX"])
    det = gpu_detect.detect_backend(cfg)
    assert det.backend == "rocm_win"
    assert det.device == "cuda"  # HIP presents as CUDA on Windows too
    assert det.python_version == "3.12"  # AMD ships cp312 wheels only
    assert det.torch_version == "2.9.1"
    assert any("torch-2.9.1" in u for u in det.torch_wheel_urls)
    assert any("rocm_sdk_core" in u for u in det.wheel_urls)


def test_windows_amd_rdna4_also_matches(monkeypatch, cfg):
    _windows_amd(monkeypatch, ["AMD Radeon RX 9070 XT"])
    assert gpu_detect.detect_backend(cfg).backend == "rocm_win"


def test_windows_amd_unsupported_card_keeps_directml(monkeypatch, cfg):
    # An RX 6800 is not on AMD's Windows-ROCm list — installing a multi-GB ROCm
    # stack for it would fail at inference instead of at install.
    _windows_amd(monkeypatch, ["AMD Radeon RX 6800 XT"])
    det = gpu_detect.detect_backend(cfg)
    assert det.backend == "dml"
    assert det.extra_packages == ["torch-directml"]


def test_windows_amd_without_wheels_configured_keeps_directml(monkeypatch, cfg):
    _windows_amd(monkeypatch, ["AMD Radeon RX 7900 XT"])
    cfg.rocm_windows = dict(cfg.rocm_windows, torch_wheels=[])
    assert gpu_detect.detect_backend(cfg).backend == "dml"


def test_linux_amd_is_unaffected_by_the_windows_path(monkeypatch, cfg):
    monkeypatch.setattr(gpu_detect, "_nvidia_query", lambda field: [])
    monkeypatch.setattr(gpu_detect.procutil, "IS_WINDOWS", False)
    monkeypatch.setattr(gpu_detect, "_has_amd_gpu", lambda: True)
    det = gpu_detect.detect_backend(cfg)
    assert det.backend == "rocm"
    assert det.wheel_urls == []  # index install, not wheel URLs


def test_forced_rocm_win_carries_the_wheel_config(monkeypatch, cfg):
    _probes_explode(monkeypatch)
    cfg.gpu_backend = "rocm_win"
    det = gpu_detect.detect_backend(cfg)
    assert det.backend == "rocm_win"
    assert det.torch_wheel_urls and det.python_version == "3.12"


# ------------------------------------------------------- worker pool default
def test_default_is_a_single_worker_even_on_a_big_gpu(monkeypatch):
    # A request is served by exactly one worker, so extra workers only buy
    # concurrency — the default must not silently spend VRAM on it.
    cfg = Config()  # untouched defaults
    assert cfg.max_workers == 1
    monkeypatch.setattr(
        gpu_detect,
        "_nvidia_query",
        lambda field: ["12.0"] if field == "compute_cap" else ["32768"],  # 32 GB free
    )
    assert gpu_detect.detect_backend(cfg).max_workers_hint == 1


def test_raising_max_workers_lets_vram_decide(monkeypatch):
    cfg = Config(max_workers=8, vram_reserve_gb=1.5, per_job_gb=3.0)
    monkeypatch.setattr(
        gpu_detect,
        "_nvidia_query",
        lambda field: ["12.0"] if field == "compute_cap" else ["16384"],  # 16 GB free
    )
    # floor((16 - 1.5) / 3) = 4, below the ceiling of 8.
    assert gpu_detect.detect_backend(cfg).max_workers_hint == 4


def test_null_max_workers_restores_auto(monkeypatch):
    cfg = Config(max_workers=None, vram_reserve_gb=1.5, per_job_gb=3.0)
    monkeypatch.setattr(
        gpu_detect,
        "_nvidia_query",
        lambda field: ["12.0"] if field == "compute_cap" else ["32768"],
    )
    # Auto is still capped at 4, not at the VRAM-derived 10.
    assert gpu_detect.detect_backend(cfg).max_workers_hint == 4
