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


class _FakeModule:
    """Stands in for an nn.Module: records half()/float() calls."""

    def __init__(self):
        self.calls: list[str] = []

    def half(self):
        self.calls.append("half")
        return self

    def float(self):
        self.calls.append("float")
        return self


class _FakeXtts(_FakeModule):
    def __init__(self, perceiver=True):
        super().__init__()
        self.hifigan_decoder = types.SimpleNamespace(speaker_encoder=_FakeModule())
        gpt = types.SimpleNamespace(conditioning_encoder=_FakeModule())
        if perceiver:
            gpt.conditioning_perceiver = _FakeModule()
        self.gpt = gpt


def test_apply_fp16_keeps_reference_audio_frontend_in_float32():
    # Regression: a blanket model.half() crashed get_conditioning_latents with
    # "Input type (torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor)"
    # because the library feeds those modules float32 audio/mel it never casts.
    model = _FakeXtts()

    kept = xtts_backend._apply_fp16(model)

    assert model.calls == ["half"]  # the model as a whole goes half
    assert kept == [
        "hifigan_decoder.speaker_encoder",
        "gpt.conditioning_encoder",
        "gpt.conditioning_perceiver",
    ]
    assert model.hifigan_decoder.speaker_encoder.calls == ["float"]
    assert model.gpt.conditioning_encoder.calls == ["float"]
    assert model.gpt.conditioning_perceiver.calls == ["float"]


def test_apply_fp16_skips_missing_submodules():
    # A coqui-tts version without the perceiver resampler must not blow up at load.
    model = _FakeXtts(perceiver=False)

    kept = xtts_backend._apply_fp16(model)

    assert "gpt.conditioning_perceiver" not in kept
    assert model.calls == ["half"]


class _FakeLatent:
    def __init__(self, dtype="float32"):
        self.dtype = dtype

    def half(self):
        return _FakeLatent("float16")


def _worker_stub(model, fp16):
    worker = object.__new__(xtts_backend.XTTSWorker)
    worker._model = model
    worker._fp16 = fp16
    worker._cond_cache = {}
    return worker


def test_conditioning_casts_latents_to_half_under_fp16():
    # The float32 front-end produces float32 latents; they must reach the half
    # GPT / vocoder as half, or inference crashes with the same dtype error.
    model = types.SimpleNamespace(
        get_conditioning_latents=lambda audio_path: (_FakeLatent(), _FakeLatent())
    )
    worker = _worker_stub(model, fp16=True)

    gpt_latent, speaker_emb = worker._conditioning("voice.wav")

    assert gpt_latent.dtype == "float16"
    assert speaker_emb.dtype == "float16"
    # Cached in the cast form (the cast must not run twice / be lost).
    assert worker._cond_cache["voice.wav"] == (gpt_latent, speaker_emb)
    assert worker._conditioning("voice.wav") == (gpt_latent, speaker_emb)


def test_conditioning_leaves_latents_untouched_without_fp16():
    model = types.SimpleNamespace(
        get_conditioning_latents=lambda audio_path: (_FakeLatent(), _FakeLatent())
    )
    worker = _worker_stub(model, fp16=False)

    gpt_latent, speaker_emb = worker._conditioning("voice.wav")

    assert gpt_latent.dtype == "float32"
    assert speaker_emb.dtype == "float32"


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
