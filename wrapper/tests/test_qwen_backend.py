"""Qwen3-TTS backend (`src/qwen_backend.py`) and its wiring — without torch.

A fake FasterQwen3TTS stands in for the model; what matters here is the
contract around it (chunks, prompt cache, xvec fallback, languages, no CPU
path) and the install shape (own venv group, own transformers pin, GPU gate).
"""

from __future__ import annotations

import importlib.util
import io
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from src import engines, ndjson
from src.config import Config, load_config
from src.engine import _default_factory
from src.inline_ref import InlineReference
from src.qwen_backend import (
    ICL_TRAILING_SILENCE_S,
    QWEN_LANGUAGES,
    QwenWorker,
    _model_dir,
    language_name,
)

WRAPPER_ROOT = Path(__file__).resolve().parent.parent


class _FakeInner:
    def __init__(self):
        self.encodes = []

    def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode=False):
        audio, sr = ref_audio
        self.encodes.append({"len": len(audio), "sr": sr, "text": ref_text, "xvec": x_vector_only_mode})
        return [f"prompt-{len(self.encodes)}"]


class _FakeModel:
    sample_rate = 24000

    def __init__(self, source, chunks=None, sr=24000):
        self.source = source
        self.model = _FakeInner()
        self.warmed = False
        self.calls = []
        self._chunks = chunks or [np.ones((2, 3), dtype=np.float64), np.zeros(4)]
        self._sr = sr

    def warmup(self, prefill_len=100):
        self.warmed = True

    def generate_voice_clone_streaming(self, **kw):
        self.calls.append(kw)
        for c in self._chunks:
            yield c, self._sr, {}


def _worker(config=None, keep_priming=False, **model_kw):
    config = config or Config(tts_backend="qwen-0.6b")
    holder = {}

    def factory(source):
        holder["model"] = _FakeModel(source, **model_kw)
        return holder["model"]

    worker = QwenWorker(config, "cuda", model_factory=factory)
    if not keep_priming:
        holder["model"].calls.clear()
        holder["model"].model.encodes.clear()
        worker._prompts = type(worker._prompts)(64)
    return worker, holder["model"]


@pytest.fixture(autouse=True)
def _fresh_log_once():
    ndjson.reset_log_once()


@pytest.fixture
def sample(tmp_path):
    import wave

    path = tmp_path / "voice.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x00\x00" * 2400)
    return path


@pytest.fixture(autouse=True)
def _soundfile_stub(monkeypatch):
    """The test venv has no soundfile; read our tiny wav with the stdlib."""
    import types
    import wave

    def read(path, dtype="float32", always_2d=False):
        with wave.open(str(path), "rb") as w:
            frames = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            return frames.astype(np.float32) / 32768.0, w.getframerate()

    monkeypatch.setitem(sys.modules, "soundfile", types.SimpleNamespace(read=read))


# ------------------------------------------------------------------ worker

def test_no_cuda_no_worker():
    """No silent CPU path: Qwen there would be unusably slow."""
    with pytest.raises(RuntimeError):
        QwenWorker(Config(tts_backend="qwen-0.6b"), "cpu", model_factory=_FakeModel)


def test_load_warms_up_the_cuda_graphs_and_speaks_once():
    """Measured: graph capture alone left the first real line at 1.6-8 s to
    first audio; one throwaway generation at start brings it to 0.3 s."""
    worker, model = _worker(keep_priming=True)
    assert model.warmed, "graph capture belongs to the start, not the first game line"
    assert len(model.calls) == 1 and model.model.encodes[0]["xvec"] is True


def test_a_failed_priming_does_not_stop_the_worker(monkeypatch):
    def boom(self, **kw):
        raise RuntimeError("no")

    monkeypatch.setattr(_FakeModel, "generate_voice_clone_streaming", boom)
    worker, _ = _worker()
    assert worker.sample_rate == 24000


def test_chunks_follow_the_contract(sample):
    worker, _ = _worker()
    parts = list(worker.infer_stream(str(sample), "Hallo Welt.", "Guten Tag.", 1.0, "de"))
    assert [p.dtype for p in parts] == [np.float32, np.float32]
    assert [p.ndim for p in parts] == [1, 1]
    assert worker.sample_rate == 24000
    assert worker.infer(str(sample), "Hallo Welt.", "Guten Tag.", 32, 1.0, "de").shape == (10,)


def test_the_prompt_is_encoded_once_per_voice(sample):
    """The first request per voice paid up to 5 s for encoding (measured)."""
    worker, model = _worker()
    for _ in range(3):
        list(worker.infer_stream(str(sample), "Hallo Welt.", "Satz.", 1.0, "de"))
    assert len(model.model.encodes) == 1
    assert all(c["voice_clone_prompt"] == ["prompt-1"] for c in model.calls)


def test_a_corrected_transcript_is_a_new_prompt(sample):
    worker, model = _worker()
    list(worker.infer_stream(str(sample), "Hallo Welt.", "Satz.", 1.0, "de"))
    list(worker.infer_stream(str(sample), "Hallo, Welt!", "Satz.", 1.0, "de"))
    assert len(model.model.encodes) == 2


def test_icl_appends_trailing_silence_xvec_does_not(sample):
    """What the library does on its own path route (verified bit-identical)."""
    worker, model = _worker()
    list(worker.infer_stream(str(sample), "Hallo Welt.", "Satz.", 1.0, "de"))
    icl = model.model.encodes[0]
    assert icl["len"] == 2400 + int(ICL_TRAILING_SILENCE_S * 24000)
    assert icl["text"] == "Hallo Welt." and icl["xvec"] is False

    xworker, xmodel = _worker()
    list(xworker.infer_stream(str(sample), "", "Satz.", 1.0, "de"))  # no transcript → xvec
    assert xmodel.model.encodes[0] == {"len": 2400, "sr": 24000, "text": "", "xvec": True}


def test_without_transcript_it_falls_back_to_xvec_and_says_so(sample, monkeypatch):
    logged = []
    monkeypatch.setattr(ndjson, "log", lambda msg, level="info": logged.append((level, msg)))
    monkeypatch.setattr(ndjson, "_seen_once", set())  # log_once: other tests hit this path too
    worker, model = _worker()
    parts = list(worker.infer_stream(str(sample), "  ", "Satz.", 1.0, "de"))
    assert parts, "audible, not a 500"
    assert model.model.encodes[0]["xvec"] is True
    assert model.calls[0]["ref_text"] == ""
    assert any(level == "warning" and getattr(msg, "key", None) == "qwen.no_ref_text" for level, msg in logged)


def test_config_reaches_the_generator(sample):
    config = Config(
        tts_backend="qwen-0.6b", qwen_temperature=0.7, qwen_non_streaming_mode=False, qwen_chunk_size=8
    )
    worker, model = _worker(config)
    list(worker.infer_stream(str(sample), "Hallo.", "Satz.", 1.0, "fr"))
    call = model.calls[0]
    assert call["temperature"] == 0.7
    assert call["non_streaming_mode"] is False
    assert call["chunk_size"] == 8
    assert call["language"] == "French"
    assert call["text"] == "Satz."


def test_user_defaults():
    """Decided 2026-11-12: whole text up front, ICL, library temperature.
    The xvec_only option was removed 2026-11-15 (it only sounded worse)."""
    config = Config()
    assert not hasattr(config, "qwen_xvec_only")
    assert config.qwen_non_streaming_mode is True
    assert config.qwen_temperature == 0.9


def test_inline_reference_uses_its_own_text_and_voice_id():
    worker, model = _worker()
    ref = InlineReference(voice_id="abc", audio=np.zeros(1200, dtype=np.float32), text="Spieltext.")
    list(worker.infer_stream(ref, "", "Satz.", 1.0, "de"))
    list(worker.infer_stream(ref, "", "Noch ein Satz.", 1.0, "de"))
    assert len(model.model.encodes) == 1
    assert model.model.encodes[0]["text"] == "Spieltext."
    assert model.calls[0]["ref_text"] == "Spieltext."


def test_a_foreign_sample_rate_is_resampled_not_passed_through(sample, monkeypatch):
    """Passed through it would play at the wrong speed, not fail."""
    import types

    seen = {}

    def resample(t, src, dst):
        seen["rates"] = (src, dst)
        return t

    fake_torch = types.SimpleNamespace(from_numpy=lambda a: types.SimpleNamespace(numpy=lambda: a))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules, "torchaudio", types.SimpleNamespace(functional=types.SimpleNamespace(resample=resample))
    )
    worker, _ = _worker(sr=48000)
    list(worker.infer_stream(str(sample), "Hallo.", "Satz.", 1.0, "de"))
    assert seen["rates"] == (48000, 24000)


def test_a_downloaded_snapshot_is_loaded_from_disk(tmp_path):
    config = Config(tts_backend="qwen-1.7b", models_dir=str(tmp_path))
    _, model = _worker(config)
    assert model.source == "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "nothing on disk → repo id"

    target = _model_dir(config)
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}", encoding="utf-8")
    _, model = _worker(config)
    assert model.source == str(target)
    assert target.parent.name == "qwen"


def test_the_worker_speaks_the_whole_protocol():
    for name in ("infer", "infer_stream", "transcribe", "self_test"):
        assert callable(getattr(QwenWorker, name))
    assert QwenWorker.supports_streaming is True
    assert QwenWorker.supports_inline_reference is True


@pytest.mark.parametrize("code, name", [("de", "German"), ("EN", "English"), (None, "German"), ("xx", "Auto")])
def test_language_names(code, name):
    assert language_name(code, "de") == name


# ------------------------------------------------------------------ wiring

@pytest.mark.parametrize("backend", ["qwen-0.6b", "qwen-1.7b"])
def test_both_qwen_engines_share_one_backend_and_venv_group(backend):
    spec = engines.spec_for(backend)
    assert spec.group == engines.QWEN_GROUP
    assert spec.model_module == "src.qwen_backend"
    assert backend in Config().qwen_models
    assert _default_factory(Config(tts_backend=backend)).__name__ == "make_qwen"


def test_qwen_is_multilingual_per_request():
    from src.server import _request_languages

    assert _request_languages(Config(tts_backend="qwen-0.6b")) == QWEN_LANGUAGES
    assert {"en", "de", "fr", "ja"} <= QWEN_LANGUAGES


@pytest.mark.parametrize("backend, ok", [
    ("cuda", True), ("rocm", True), ("rocm_win", True), ("cpu", False), ("dml", False), ("xpu", False),
])
def test_device_requirement(backend, ok):
    reason = engines.unsupported_device_reason(engines.spec_for("qwen-0.6b"), backend)
    assert (reason is None) is ok
    assert engines.unsupported_device_reason(engines.spec_for("xtts"), backend) is None


def test_qwen_options_coerce_from_env():
    config = load_config(argv=[], env={
        "F5W_QWEN_NON_STREAMING_MODE": "false",
        "F5W_QWEN_TEMPERATURE": "0.7", "F5W_QWEN_CHUNK_SIZE": "8",
    })
    assert config.qwen_non_streaming_mode is False
    assert config.qwen_temperature == 0.7
    assert config.qwen_chunk_size == 8


# --------------------------------------------------------------- bootstrap

def _load_bootstrap():
    spec = importlib.util.spec_from_file_location("bootstrap_qwen_under_test", WRAPPER_ROOT / "bootstrap" / "bootstrap.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


@pytest.fixture
def bootstrap(monkeypatch, tmp_path):
    mod = _load_bootstrap()
    calls = []
    monkeypatch.setattr(mod, "STATE_DIR", tmp_path / ".state")
    monkeypatch.setattr(mod, "_run_uv", lambda args, *_: calls.append([str(a) for a in args]))
    monkeypatch.setattr(mod, "_verify_group", lambda *a, **k: None)
    monkeypatch.setattr(mod.procutil, "run", lambda *a, **k: _Proc())
    monkeypatch.setattr(mod, "_uv_path", lambda: Path("uv"))
    # The bootstrap refuses darwin; these tests are about paths, not the OS.
    monkeypatch.setattr(mod, "_platform_key", lambda: "linux")
    monkeypatch.setattr(mod, "_stop_stale_processes", lambda *a, **k: None)
    monkeypatch.setattr(mod.ndjson, "progress", lambda *a, **k: None)
    monkeypatch.setattr(mod.ndjson, "log", lambda *a, **k: None)
    mod._calls = calls
    return mod


def _detection(mod, backend="cuda"):
    return mod.gpu_detect.Detection(backend=backend, device=backend, torch_index_url="https://x.invalid", detail="t")


def test_qwen_gets_its_own_venv_and_transformers_5(bootstrap):
    config = Config(tts_backend="qwen-0.6b")
    bootstrap.step_deps(config, _detection(bootstrap))

    venv_cmd = next(c for c in bootstrap._calls if c[0] == "venv")
    assert venv_cmd[1].endswith(".venv-qwen")
    flat = [a for c in bootstrap._calls for a in c]
    assert "faster-qwen3-tts==0.5.4" in flat
    assert "transformers==5.15.1" in flat
    assert config.transformers_constraint not in flat, "the classic <5 pin must not leak in"
    assert "coqui-tts" not in flat and "f5-tts" not in flat


def test_qwen_is_refused_before_any_download_without_a_gpu(bootstrap):
    with pytest.raises(bootstrap.FatalError) as err:
        bootstrap.step_deps(Config(tts_backend="qwen-1.7b"), _detection(bootstrap, "cpu"))
    assert bootstrap._calls == []
    assert "NVIDIA" in str(err.value)


def test_qwen_foundation_checks_the_exact_transformers(bootstrap, monkeypatch):
    monkeypatch.setattr(bootstrap, "_verify_torch", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap, "_verify_transformers", lambda *a: pytest.fail("classic canary in qwen venv"))
    monkeypatch.setattr(bootstrap.procutil, "run", lambda *a, **k: _Proc(stdout="5.22.0\n"))
    with pytest.raises(bootstrap.FatalError) as err:
        bootstrap._verify_foundation("py", Config(), "2.7.0", "qwen")
    assert "5.15.1" in str(err.value)

    monkeypatch.setattr(bootstrap.procutil, "run", lambda *a, **k: _Proc(stdout="5.15.1\n"))
    bootstrap._verify_foundation("py", Config(), "2.7.0", "qwen")


def test_the_qwen_server_runs_from_the_qwen_venv(bootstrap, monkeypatch):
    seen = []
    monkeypatch.setattr(bootstrap, "_popen_forward", lambda cmd, env: seen.append(cmd) or 0)
    monkeypatch.setattr(bootstrap, "_server_env", lambda config: {})
    bootstrap.step_serve(Config(tts_backend="qwen-0.6b"))
    assert ".venv-qwen" in seen[0][0]


# --- SoX import noise ---------------------------------------------------------
# sox/__init__.py (pysox 1.5) does, on import: log.py sets the "sox" logger to
# WARNING, then os.popen('sox -h') and, with no output, logger.warning(...).

def _fake_sox_import():
    log = logging.getLogger("sox")
    log.setLevel(logging.WARNING)  # what sox/log.py does — overrides any level
    handle = os.popen("sox -h")
    lines = handle.readlines()
    handle.close()
    if not lines:
        log.warning("SoX could not be found!")
    return lines


def test_quiet_sox_drops_the_warning_although_sox_resets_the_level(caplog):
    from src import qwen_backend

    with caplog.at_level(logging.DEBUG):
        with qwen_backend._quiet_sox():
            _fake_sox_import()
    assert "SoX could not be found" not in caplog.text


def test_quiet_sox_keeps_sox_errors(caplog):
    from src import qwen_backend

    with caplog.at_level(logging.DEBUG):
        with qwen_backend._quiet_sox():
            logging.getLogger("sox").error("real sox failure")
    assert "real sox failure" in caplog.text


def test_quiet_sox_sends_the_shell_error_to_the_null_device(monkeypatch):
    from src import qwen_backend

    seen = []
    monkeypatch.setattr(os, "popen", lambda cmd, *a, **k: seen.append(cmd) or io.StringIO(""))
    with qwen_backend._quiet_sox():
        os.popen("sox -h")
        os.popen("dir")
    assert seen == [f"sox -h 2>{os.devnull}", "dir"]


def test_quiet_sox_restores_popen_even_on_error():
    from src import qwen_backend

    real = os.popen
    with pytest.raises(ImportError):
        with qwen_backend._quiet_sox():
            raise ImportError("boom")
    assert os.popen is real


def test_model_loading_runs_inside_quiet_sox():
    """qwen_tts imports sox lazily inside from_pretrained, not at import time
    (live 2026-11-15) — so the model load itself must run under _quiet_sox."""
    real, seen = os.popen, []

    def factory(source):
        seen.append(os.popen is not real)  # os.popen is patched only inside _quiet_sox
        return _FakeModel(source)

    QwenWorker(Config(tts_backend="qwen-0.6b"), "cuda", model_factory=factory)
    assert seen == [True]
    assert os.popen is real
