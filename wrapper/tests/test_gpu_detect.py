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
