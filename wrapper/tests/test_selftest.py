"""The shared worker self-test.

It used to exist twice, once per backend, with one difference: F5 passed
``ref_text="test"`` and XTTS passed ``""``. That difference was not cosmetic —
an empty reference text tells F5 to transcribe the clip, which drags its
Whisper preprocessor into a probe whose whole point is to be tiny. So the
shared version keeps the non-empty value, and this file pins that.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import selftest


class _Worker:
    """Records what the probe asked for; answers like a working backend."""

    sample_rate = 24000

    def __init__(self, raises: Exception | None = None):
        self.calls = []
        self._raises = raises

    def infer(self, ref_file, ref_text, gen_text, nfe_step, speed, language=None):
        self.calls.append(
            {"ref_file": ref_file, "ref_text": ref_text, "gen_text": gen_text,
             "nfe_step": nfe_step, "speed": speed}
        )
        if self._raises:
            raise self._raises
        return np.zeros(8, dtype=np.float32)


def test_passing_worker_reports_success():
    worker = _Worker()
    assert selftest.run(worker) is True
    assert len(worker.calls) == 1


def test_reference_text_is_never_empty():
    """Empty ref_text makes F5 load Whisper to transcribe silence."""
    worker = _Worker()
    selftest.run(worker)
    assert worker.calls[0]["ref_text"], "ref_text must stay non-empty"


def test_probe_stays_small():
    """A self-test that costs a full inference defeats its own purpose."""
    worker = _Worker()
    selftest.run(worker)
    assert worker.calls[0]["nfe_step"] <= 8


def test_reference_file_exists_while_infer_runs():
    """The temp dir must outlive the call, not be cleaned up before it."""
    seen = {}

    class _Checking(_Worker):
        def infer(self, ref_file, *a, **k):
            from pathlib import Path

            seen["exists"] = Path(ref_file).is_file()
            return super().infer(ref_file, *a, **k)

    assert selftest.run(_Checking()) is True
    assert seen["exists"] is True


@pytest.mark.parametrize(
    "boom",
    [RuntimeError("no kernel for this op"), MemoryError(), KeyboardInterrupt()],
)
def test_any_failure_answers_false_instead_of_raising(boom):
    """The caller uses this to decide 'fall back to CPU' — it must not explode.

    KeyboardInterrupt is in the list on purpose: it does NOT inherit from
    Exception, so it stays a real interrupt rather than being swallowed.
    """
    worker = _Worker(raises=boom)
    if isinstance(boom, Exception):
        assert selftest.run(worker) is False
    else:
        with pytest.raises(KeyboardInterrupt):
            selftest.run(worker)


def test_both_backends_use_the_shared_probe():
    """Guards against the duplication growing back."""
    import inspect

    from src.engine import F5TTSWorker
    from src.xtts_backend import XTTSWorker

    for cls in (F5TTSWorker, XTTSWorker):
        source = inspect.getsource(cls.self_test)
        assert "selftest.run" in source
        assert "tempfile" not in source


def test_probe_needs_no_audio_library():
    """The reference clip is written with the wrapper's own WAV header.

    soundfile would be swallowed by the except: a machine missing it would be
    reported as a device that cannot run the model. This test passes in the
    unit venv, which has no soundfile at all — that IS the assertion.
    """
    import inspect

    source = inspect.getsource(selftest)
    assert "soundfile" not in source.split('"""')[-1]
    assert selftest.run(_Worker()) is True


def test_reference_clip_is_a_parsable_wav(tmp_path):
    """A header the engines cannot read would fail every probe on every device."""
    import wave

    ref = tmp_path / "probe.wav"
    selftest._write_silence(ref, 24000)
    with wave.open(str(ref)) as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 24000
        assert w.readframes(w.getnframes()) == b"\x00" * 48000
