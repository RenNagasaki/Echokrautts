"""DirectML: one venv, one torch — the version torch-directml actually allows.

``torch-directml`` hard-declares ``torch==2.4.1`` and has not shipped since
2024-09-14; Microsoft has DirectML in maintenance mode and there is no successor
that carries PyTorch (torchruntime, still maintained in 2026, also selects
DirectML for every AMD card on Windows). Installed as a bare extra into the
configured 2.7 venv it silently downgraded torch, and ``_verify_torch`` then
failed the install — deterministically, in every released version, on every
AMD/Windows machine. Reported live twice.

The fix is to let the backend own the torch version for its whole venv. These
tests pin the two consequences of that: the pin travels with the extra, and
torch AND torchaudio follow the backend rather than the configured pin.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from src import gpu_detect
from src.config import Config

WRAPPER_ROOT = Path(__file__).resolve().parent.parent


def _load_bootstrap():
    spec = importlib.util.spec_from_file_location(
        "bootstrap_dml_under_test", WRAPPER_ROOT / "bootstrap" / "bootstrap.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def bootstrap(monkeypatch):
    mod = _load_bootstrap()
    calls: list[list[str]] = []
    monkeypatch.setattr(mod, "_run_uv", lambda args, *_a, **_k: calls.append([str(a) for a in args]))
    monkeypatch.setattr(mod, "_is_done", lambda _step: False)
    monkeypatch.setattr(mod, "_mark_done", lambda _step: None)
    monkeypatch.setattr(mod, "_venv_python", lambda: Path("py"))
    monkeypatch.setattr(mod, "_uv_path", lambda: Path("uv"))
    monkeypatch.setattr(mod.procutil, "run", lambda *_a, **_k: _Proc())
    monkeypatch.setattr(mod, "_verify_venv", lambda *_a, **_k: None)
    monkeypatch.setattr(mod.ndjson, "progress", lambda *_a, **_k: None)
    monkeypatch.setattr(mod.ndjson, "log", lambda *_a, **_k: None)
    mod._recorded_calls = calls
    return mod


def _dml_detection() -> gpu_detect.Detection:
    """The real detection object, not a hand-built stand-in.

    Building one here would let the branch under test drift away from what
    ``_detect_amd`` actually returns without a single test noticing.
    """
    return gpu_detect.Detection(
        backend="dml",
        device="dml",
        torch_index_url=gpu_detect.TORCH_INDEX["dml"],
        extra_packages=["torch-directml"],
        torch_version=gpu_detect.DML_TORCH_VERSION,
        torchaudio_version=gpu_detect.DML_TORCHAUDIO_VERSION,
        detail="test",
    )


# --------------------------------------------------------------------- detection

def test_dml_detection_carries_the_torch_directml_versions(monkeypatch):
    """What AMD-on-Windows detects must be what torch-directml can live with."""
    monkeypatch.setattr(gpu_detect, "_has_amd_gpu", lambda: True)
    monkeypatch.setattr(gpu_detect.procutil, "IS_WINDOWS", True)
    monkeypatch.setattr(gpu_detect, "rocm_windows_candidate", lambda _c: None)

    det = gpu_detect._detect_amd(Config())

    assert det.backend == "dml"
    assert det.torch_version == "2.4.1"
    assert det.torchaudio_version == det.torch_version


def test_other_backends_keep_the_configured_torch(monkeypatch):
    """The override must stay DirectML-only, not leak into ROCm/CPU."""
    monkeypatch.setattr(gpu_detect, "_has_amd_gpu", lambda: True)
    monkeypatch.setattr(gpu_detect.procutil, "IS_WINDOWS", False)

    det = gpu_detect._detect_amd(Config())

    assert det.backend == "rocm"
    assert det.torch_version is None
    assert det.torchaudio_version is None


# ------------------------------------------------------------------- install shape

def test_torch_pin_travels_with_the_extra(bootstrap):
    """The exact bug: torch-directml installed alone was free to move torch."""
    bootstrap.step_deps(Config(), _dml_detection())

    extra_installs = [c for c in bootstrap._recorded_calls if "torch-directml" in c]
    assert len(extra_installs) == 1, "the extra must be installed exactly once"
    args = extra_installs[0]
    assert "torch==2.4.1" in args
    assert "torchaudio==2.4.1" in args


def test_extra_uses_an_extra_index_not_a_replacement_index(bootstrap):
    """`--index-url` would replace PyPI, where torch-directml itself lives."""
    bootstrap.step_deps(Config(), _dml_detection())

    args = [c for c in bootstrap._recorded_calls if "torch-directml" in c][0]
    assert "--extra-index-url" in args
    assert "--index-url" not in args


def test_torch_and_torchaudio_follow_the_backend_everywhere(bootstrap):
    """A 2.4.1 torch beside a 2.7.0 torchaudio is its own broken install."""
    bootstrap.step_deps(Config(), _dml_detection())

    for call in bootstrap._recorded_calls:
        for arg in call:
            if arg.startswith("torch=="):
                assert arg == "torch==2.4.1"
            if arg.startswith("torchaudio=="):
                assert arg == "torchaudio==2.4.1"


def test_both_engines_are_still_installed_on_directml(bootstrap):
    """The torch override must not cost the engines it was meant to serve."""
    bootstrap.step_deps(Config(), _dml_detection())

    flat = [arg for call in bootstrap._recorded_calls for arg in call]
    assert "coqui-tts" in flat, "XTTS must still be installed"
    assert str(WRAPPER_ROOT) in flat, "F5 must still be installed"
