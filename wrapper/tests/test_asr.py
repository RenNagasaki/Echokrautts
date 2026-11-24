"""`src/asr.py`: Whisper for /asr-check, with or without f5-tts installed."""

import sys
import types

import numpy as np
import pytest

from src import asr


@pytest.fixture(autouse=True)
def _fresh_pipeline(monkeypatch):
    monkeypatch.setattr(asr, "_own_pipeline", None)


def test_uses_f5s_pipeline_when_f5_is_installed(monkeypatch):
    """One Whisper in VRAM: F5 also uses its pipeline for the ref-text fallback."""
    calls = []
    monkeypatch.setattr(asr, "_f5_transcribe", lambda: lambda p, language: calls.append((p, language)) or " hallo ")
    monkeypatch.setattr(asr, "_build_pipeline", lambda: pytest.fail("must not build a second Whisper"))
    assert asr.whisper_transcribe(np.zeros(16, dtype=np.float32), 24000, "de") == "hallo"
    payload, language = calls[0]
    assert payload["sampling_rate"] == 24000 and language == "de"


def test_builds_its_own_pipeline_without_f5_and_reuses_it(monkeypatch):
    """Lazy install: under XTTS/MOSS/Qwen there is no f5-tts in the venv."""
    built, seen = [], []

    def fake_pipe(payload, **kwargs):
        seen.append((payload, kwargs))
        return {"text": " guten Tag "}

    monkeypatch.setattr(asr, "_f5_transcribe", lambda: None)
    monkeypatch.setattr(asr, "_build_pipeline", lambda: built.append(1) or fake_pipe)

    assert asr.whisper_transcribe(np.zeros(8), 16000, "de") == "guten Tag"
    assert asr.whisper_transcribe(np.zeros(8), 16000, None) == "guten Tag"
    assert built == [1], "the pipeline is built once per process"
    (p1, k1), (_, k2) = seen
    assert p1["raw"].dtype == np.float32 and p1["sampling_rate"] == 16000
    assert k1["generate_kwargs"] == {"task": "transcribe", "language": "de"}
    assert k2["generate_kwargs"] == {"task": "transcribe"}, "no language → Whisper detects it"
    assert k1["return_timestamps"] is False


@pytest.mark.parametrize("cuda, device", [(True, "cuda:0"), (False, "cpu")])
def test_pipeline_device_and_dtype(cuda, device):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda), float16="f16", float32="f32"
    )
    captured = {}

    def factory(task, **kwargs):
        captured.update(kwargs, task=task)
        return "pipe"

    assert asr._build_pipeline(factory=factory, torch_module=fake_torch) == "pipe"
    assert captured["task"] == "automatic-speech-recognition"
    assert captured["model"] == asr.WHISPER_MODEL
    assert captured["device"] == device
    assert captured["torch_dtype"] == ("f16" if cuda else "f32")


def test_missing_f5_is_detected_as_none(monkeypatch):
    # sys.modules[name] = None makes `import` raise ImportError, like an absent package.
    monkeypatch.setitem(sys.modules, "f5_tts", None)
    assert asr._f5_transcribe() is None


def _fake_logging_module():
    state = {"v": 30}  # WARNING
    return types.SimpleNamespace(
        get_verbosity=lambda: state["v"],
        set_verbosity=lambda v: state.__setitem__("v", v),
        set_verbosity_error=lambda: state.__setitem__("v", 40),
        state=state,
    )


def test_library_warnings_are_errors_only_while_whisper_runs(monkeypatch):
    """transformers/huggingface_hub chatter on every call ("chunk_length_s is very
    experimental", HF_TOKEN …) — noise in a user's log. Their level comes back after."""
    fake = _fake_logging_module()
    monkeypatch.setitem(sys.modules, "fake_noisy_logging", fake)
    monkeypatch.setattr(asr, "_NOISY_LOGGING", ("fake_noisy_logging", "not_installed_logging"))
    during = []
    monkeypatch.setattr(asr, "_f5_transcribe", lambda: None)
    monkeypatch.setattr(asr, "_build_pipeline", lambda: lambda p, **k: during.append(fake.state["v"]) or {"text": "x"})

    assert asr.whisper_transcribe(np.zeros(8), 16000) == "x"
    assert during == [40], "errors only while Whisper runs"
    assert fake.state["v"] == 30, "previous verbosity restored"


def test_library_verbosity_restored_when_whisper_fails(monkeypatch):
    fake = _fake_logging_module()
    monkeypatch.setitem(sys.modules, "fake_noisy_logging", fake)
    monkeypatch.setattr(asr, "_NOISY_LOGGING", ("fake_noisy_logging",))
    monkeypatch.setattr(asr, "_f5_transcribe", lambda: lambda p, language: 1 / 0)

    with pytest.raises(ZeroDivisionError):
        asr.whisper_transcribe(np.zeros(8), 16000)
    assert fake.state["v"] == 30
