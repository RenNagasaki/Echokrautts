"""Tests for the ONNX MOSS runtime.

The runtime itself is not importable here (onnxruntime and 763 MB of exported
graphs are not part of the test venv), so these cover what this repo actually
decides: where the models live, when the ONNX path is considered usable, how
many threads it takes, and — the part that is genuinely ours — the callback to
generator bridge that turns the vendor's internal incremental decode into the
stream the wrapper promises.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src import moss_onnx_backend  # noqa: E402
from src.config import Config  # noqa: E402


# ------------------------------------------------------------------- location


def test_models_live_under_the_wrapper_models_dir(tmp_path):
    cfg = Config(models_dir=str(tmp_path))
    assert moss_onnx_backend.onnx_model_dir(cfg) == tmp_path / "moss_onnx"


def test_directory_name_has_no_hyphen():
    """Same convention as the PyTorch directories.

    A hyphen becomes a literal ``_hyphen_`` in a generated module name, which
    already broke the PyTorch path once. ONNX does not generate modules, but one
    convention is cheaper to keep than a rule about which directory may differ.
    """
    assert "-" not in moss_onnx_backend.ONNX_DIRNAME


# -------------------------------------------------------------- availability


def test_missing_onnxruntime_is_reported_as_such(tmp_path, monkeypatch):
    """The reason has to name the missing thing, or the user cannot act on it."""
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("no onnxruntime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    usable, reason = moss_onnx_backend.is_available(Config(models_dir=str(tmp_path)))
    assert usable is False
    assert "onnxruntime" in reason


def test_missing_weights_are_reported_separately(tmp_path, monkeypatch):
    """Package present but graphs absent is a DIFFERENT fix, so a different message."""
    monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    stub = types.ModuleType("onnx_tts_runtime")

    def missing(_dir):
        raise FileNotFoundError("browser_onnx model assets not found")

    stub.ensure_browser_onnx_model_dir = missing
    monkeypatch.setitem(sys.modules, "onnx_tts_runtime", stub)

    usable, reason = moss_onnx_backend.is_available(Config(models_dir=str(tmp_path)))
    assert usable is False
    assert "Modelldateien" in reason
    assert "onnxruntime ist nicht installiert" not in reason


def test_available_when_both_are_there(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    stub = types.ModuleType("onnx_tts_runtime")
    stub.ensure_browser_onnx_model_dir = lambda d: d
    monkeypatch.setitem(sys.modules, "onnx_tts_runtime", stub)

    assert moss_onnx_backend.is_available(Config(models_dir=str(tmp_path))) == (True, "")


# ------------------------------------------------------------------- threads


def test_configured_thread_count_wins(tmp_path):
    cfg = Config(models_dir=str(tmp_path), moss_onnx_threads=3)
    assert moss_onnx_backend.resolve_threads(cfg) == 3


def test_default_thread_count_stays_modest(tmp_path, monkeypatch):
    """Measured: 16 intra-op threads were as slow as ONE and slower than four.

    So the default must not follow the core count, however many there are — and
    on a machine that is also running a game, taking four is the point.
    """
    monkeypatch.setattr(moss_onnx_backend.os, "cpu_count", lambda: 32)
    assert moss_onnx_backend.resolve_threads(Config(models_dir=str(tmp_path))) == 4


def test_default_thread_count_never_exceeds_the_machine(tmp_path, monkeypatch):
    monkeypatch.setattr(moss_onnx_backend.os, "cpu_count", lambda: 2)
    assert moss_onnx_backend.resolve_threads(Config(models_dir=str(tmp_path))) == 2


def test_default_thread_count_is_at_least_one(tmp_path, monkeypatch):
    monkeypatch.setattr(moss_onnx_backend.os, "cpu_count", lambda: None)
    assert moss_onnx_backend.resolve_threads(Config(models_dir=str(tmp_path))) == 1


# ------------------------------------------------- callback -> generator bridge


class _FakeSession:
    """Stands in for the vendor's codec streaming session."""

    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def run_frames(self, frames):
        # one 80 ms frame -> 3840 samples of 2-channel audio, like the real one
        n = len(frames) * 3840
        return np.ones((1, 2, n), dtype=np.float32), n


def _worker_with(runtime_stub, session, monkeypatch, cfg):
    monkeypatch.setitem(sys.modules, "onnx_tts_runtime", runtime_stub)
    worker = moss_onnx_backend.MossOnnxWorker.__new__(moss_onnx_backend.MossOnnxWorker)
    worker.sample_rate = 24000
    worker._max_new_frames = 375
    worker._resampler = moss_onnx_backend.ContractResampler()
    worker.supports_streaming = True
    worker._runtime = types.SimpleNamespace(codec_streaming_session=session)
    return worker


def _runtime_stub(frames=6, budget=2):
    stub = types.ModuleType("onnx_tts_runtime")
    stub._resolve_stream_decode_frame_budget = lambda *a, **k: budget
    return stub


def test_streaming_yields_pieces_not_one_blob(monkeypatch, tmp_path):
    """The whole reason this module exists: the vendor returns one clip.

    Its streaming path decodes incrementally and then concatenates, so the
    pieces exist and are simply never offered. This bridge has to surface them.
    """
    session = _FakeSession()
    stub = _runtime_stub(budget=2)
    worker = _worker_with(stub, session, monkeypatch, Config(models_dir=str(tmp_path)))

    def generate(rows, on_frame=None):
        for step in range(6):
            on_frame([], step, [0] * 16)
        return []

    worker._runtime.generate_audio_frames = generate
    pieces = list(worker._decode_streaming({"inputIds": []}, 48000))
    assert len(pieces) > 1, "a single piece would mean the bridge did nothing"
    assert all(rate == 48000 for _, rate in pieces)


def test_every_generated_frame_reaches_the_consumer(monkeypatch, tmp_path):
    """A budget that never divides the frame count must not swallow the remainder."""
    session = _FakeSession()
    worker = _worker_with(_runtime_stub(budget=4), session, monkeypatch,
                          Config(models_dir=str(tmp_path)))

    def generate(rows, on_frame=None):
        for step in range(7):          # 7 frames, budget 4 -> 4 + a forced 3
            on_frame([], step, [0] * 16)
        return []

    worker._runtime.generate_audio_frames = generate
    total = sum(chunk.shape[-1] for chunk, _ in worker._decode_streaming({}, 48000))
    assert total == 7 * 3840


def test_a_failure_in_the_worker_thread_reaches_the_caller(monkeypatch, tmp_path):
    """Otherwise the generator just ends early and the clip is silently short."""
    session = _FakeSession()
    worker = _worker_with(_runtime_stub(), session, monkeypatch,
                          Config(models_dir=str(tmp_path)))

    def generate(rows, on_frame=None):
        raise RuntimeError("onnx blew up")

    worker._runtime.generate_audio_frames = generate
    with pytest.raises(RuntimeError, match="onnx blew up"):
        list(worker._decode_streaming({}, 48000))


def test_the_codec_session_is_reset_around_every_request(monkeypatch, tmp_path):
    """Shared, single-slot state: leftovers would splice one clip onto the next."""
    session = _FakeSession()
    worker = _worker_with(_runtime_stub(), session, monkeypatch,
                          Config(models_dir=str(tmp_path)))
    worker._runtime.generate_audio_frames = lambda rows, on_frame=None: []
    list(worker._decode_streaming({}, 48000))
    assert session.resets >= 2, "reset before AND after, or state leaks between requests"
