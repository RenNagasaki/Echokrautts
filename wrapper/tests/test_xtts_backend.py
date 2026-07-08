"""Tests for the XTTS-v2 backend that don't need torch / coqui-tts installed.

The heavy imports (torch, TTS) live inside XTTSWorker.__init__ and the resolver,
so the module imports and its plumbing is testable with fakes.
"""

import os
import sys
import types

from src import xtts_backend
from src.config import Config


def test_module_imports_without_heavy_deps():
    assert xtts_backend.XTTS_MODEL.endswith("xtts_v2")
    assert xtts_backend.DEFAULT_SAMPLE_RATE == 24000


def test_should_use_fp16_only_on_cuda_when_enabled():
    on = Config(xtts_fp16=True)
    off = Config(xtts_fp16=False)
    # Enabled + CUDA → yes.
    assert xtts_backend._should_use_fp16(on, "cuda") is True
    # Enabled but non-CUDA (cpu/dml/xpu resolve to cpu) → no.
    assert xtts_backend._should_use_fp16(on, "cpu") is False
    assert xtts_backend._should_use_fp16(on, "xpu") is False
    # Disabled → never, even on CUDA.
    assert xtts_backend._should_use_fp16(off, "cuda") is False


def test_download_model_uses_resolver(tmp_path, monkeypatch):
    cfg = Config(models_dir=str(tmp_path / "models"))
    seen = []
    monkeypatch.setattr(
        xtts_backend, "_resolve_model_dir", lambda c: seen.append(c) or "dir"
    )
    xtts_backend.download_model(cfg)
    assert seen == [cfg]
    assert cfg.models_path.exists()  # download_model ensures the dir exists


def test_custom_xtts_model_dir_wins(tmp_path):
    # A full XTTS model dir (config.json + a weight file) short-circuits the
    # base-model download entirely.
    cfg = Config(models_dir=str(tmp_path / "models"))
    custom = cfg.custom_model_path
    custom.mkdir(parents=True)
    (custom / "config.json").write_text("{}", encoding="utf-8")
    (custom / "model.pth").write_bytes(b"weights")

    assert xtts_backend._resolve_model_dir(cfg) == str(custom)


def test_custom_xtts_ignored_without_config(tmp_path):
    # A bare checkpoint (F5-style, no config.json) is NOT an XTTS model dir.
    cfg = Config(models_dir=str(tmp_path / "models"))
    custom = cfg.custom_model_path
    custom.mkdir(parents=True)
    (custom / "model.safetensors").write_bytes(b"weights")

    assert xtts_backend._resolve_custom_model_dir(cfg) is None


def test_resolve_model_dir_downloads_and_accepts_license(tmp_path, monkeypatch):
    cfg = Config(models_dir=str(tmp_path / "models"))

    captured = {}

    class FakeManager:
        def __init__(self, output_prefix=None):
            captured["output_prefix"] = output_prefix

        def download_model(self, model_id):
            captured["model_id"] = model_id
            return ("/models/xtts", "/models/xtts/config.json", {})

    fake_manage = types.ModuleType("TTS.utils.manage")
    fake_manage.ModelManager = FakeManager
    monkeypatch.setitem(sys.modules, "TTS", types.ModuleType("TTS"))
    monkeypatch.setitem(sys.modules, "TTS.utils", types.ModuleType("TTS.utils"))
    monkeypatch.setitem(sys.modules, "TTS.utils.manage", fake_manage)
    monkeypatch.delenv("COQUI_TOS_AGREED", raising=False)

    path = xtts_backend._resolve_model_dir(cfg)

    assert path == "/models/xtts"
    assert captured["model_id"] == xtts_backend.XTTS_MODEL
    assert captured["output_prefix"] == str(cfg.models_path)
    # The CPML must be accepted non-interactively so the download never blocks.
    assert os.environ["COQUI_TOS_AGREED"] == "1"
