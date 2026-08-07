"""The native-Windows ROCm install path in ``step_deps``.

AMD publishes no pip index for Windows — only individual wheel URLs, built for
exactly one Python. These tests pin the resulting uv command sequence, because
the path cannot be exercised on this machine (it needs a supported AMD GPU on
Windows) and a silent regression would only surface at a user's install.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from src.config import Config
from src.gpu_detect import Detection, TORCH_INDEX

WRAPPER_ROOT = Path(__file__).resolve().parent.parent


def _load_bootstrap():
    spec = importlib.util.spec_from_file_location(
        "bootstrap_rocm_under_test", WRAPPER_ROOT / "bootstrap" / "bootstrap.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bootstrap(monkeypatch):
    mod = _load_bootstrap()
    # Never touch the disk or the network: record the uv calls instead.
    calls: list[list[str]] = []
    monkeypatch.setattr(mod, "_run_uv", lambda args, *_a, **_k: calls.append(list(args)))
    monkeypatch.setattr(mod, "_is_done", lambda _step: False)
    monkeypatch.setattr(mod, "_mark_done", lambda _step: None)
    monkeypatch.setattr(mod, "_venv_python", lambda: Path("py"))
    monkeypatch.setattr(mod.procutil, "run", lambda *_a, **_k: None)
    monkeypatch.setattr(mod, "_verify_torch", lambda *_a, **_k: None)
    monkeypatch.setattr(mod, "_verify_transformers", lambda *_a: None)
    monkeypatch.setattr(mod.ndjson, "progress", lambda *_a, **_k: None)
    mod._recorded_calls = calls
    return mod


def _rocm_win_detection(config: Config) -> Detection:
    s = config.rocm_windows
    return Detection(
        backend="rocm_win",
        device="cuda",
        torch_index_url=TORCH_INDEX["cpu"],
        wheel_urls=list(s["wheels"]),
        torch_wheel_urls=list(s["torch_wheels"]),
        python_version=s["python"],
        torch_version=s["torch_version"],
    )


def test_venv_uses_the_backends_python_not_the_configured_one(bootstrap):
    config = Config()  # python_version stays 3.11
    bootstrap.step_deps(config, _rocm_win_detection(config))
    venv_cmd = next(c for c in bootstrap._recorded_calls if c[0] == "venv")
    # AMD's wheels are cp312-only; installing them into a 3.11 venv cannot work.
    assert venv_cmd[venv_cmd.index("--python") + 1] == "3.12"
    assert config.python_version == "3.11"


def test_rocm_runtime_is_installed_before_torch(bootstrap):
    config = Config()
    bootstrap.step_deps(config, _rocm_win_detection(config))
    installs = [c for c in bootstrap._recorded_calls if c[:2] == ["pip", "install"]]
    sdk_at = next(i for i, c in enumerate(installs) if any("rocm_sdk_core" in a for a in c))
    torch_at = next(i for i, c in enumerate(installs) if any("/torch-2.9.1" in a for a in c))
    assert sdk_at < torch_at


def test_torch_comes_from_wheel_urls_and_never_an_index(bootstrap):
    config = Config()
    bootstrap.step_deps(config, _rocm_win_detection(config))
    installs = [c for c in bootstrap._recorded_calls if c[:2] == ["pip", "install"]]
    torch_cmds = [c for c in installs if any("/torch-" in a for a in c)]
    assert torch_cmds, "torch was never installed"
    for cmd in torch_cmds:
        assert "--index-url" not in cmd
        assert not any(a.startswith("torch==") for a in cmd)
    # Re-pinned after the engine install, exactly like the index-based path:
    # f5-tts/coqui can otherwise pull a stock torch over AMD's build.
    assert len(torch_cmds) == 2


def test_engines_are_still_one_resolution(bootstrap):
    config = Config()
    bootstrap.step_deps(config, _rocm_win_detection(config))
    engine_cmds = [
        c
        for c in bootstrap._recorded_calls
        if any(a == "coqui-tts" for a in c)
    ]
    assert len(engine_cmds) == 1
    assert config.transformers_constraint in engine_cmds[0]


def test_index_based_backends_are_unchanged(bootstrap):
    config = Config()
    det = Detection(
        backend="cuda", device="cuda", torch_index_url=TORCH_INDEX["cu128"]
    )
    bootstrap.step_deps(config, det)
    installs = [c for c in bootstrap._recorded_calls if c[:2] == ["pip", "install"]]
    torch_cmds = [c for c in installs if any(a.startswith("torch==") for a in c)]
    assert len(torch_cmds) == 2  # install + re-pin
    for cmd in torch_cmds:
        assert "--index-url" in cmd
        assert f"torch=={config.torch_version}" in cmd
    venv_cmd = next(c for c in bootstrap._recorded_calls if c[0] == "venv")
    assert venv_cmd[venv_cmd.index("--python") + 1] == config.python_version
