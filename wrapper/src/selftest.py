"""The worker self-test, shared by every backend.

The engine runs this once per worker on the fragile devices (dml/xpu) whose op
coverage does not carry these models, and falls back to CPU when it fails
(SPEC §4.2). It is deliberately a free function taking a worker rather than a
base class: the backends implement ``WorkerProtocol`` structurally and share no
inheritance, and a probe that needs only ``sample_rate`` and ``infer`` has no
business dictating a class hierarchy.
"""

from __future__ import annotations

from pathlib import Path

from .wav import wrap_pcm

# One second of silence is enough for a reference clip nobody listens to, and
# short enough that the probe stays a probe.
SELF_TEST_SECONDS = 1
# Non-empty on purpose — see run().
SELF_TEST_REF_TEXT = "test"


def _write_silence(path: Path, sample_rate: int) -> None:
    """Write a one-second silent 16-bit mono WAV using the wrapper's own header.

    Deliberately NOT soundfile: this module is imported by both backends and
    must stay importable (and testable) without the audio stack, and the probe
    swallows exceptions — a missing soundfile would therefore have been reported
    as "this device cannot run the model", which is the wrong answer to the
    wrong question. ``wav.wrap_pcm`` is hand-written and unit-tested against the
    stdlib ``wave`` parser, so the file is as trustworthy as the library one.
    """
    path.write_bytes(wrap_pcm(b"\x00" * (2 * sample_rate * SELF_TEST_SECONDS), sample_rate))


def run(worker) -> bool:
    """Synthesize one tiny clip; True when the device carried it.

    ``ref_text`` must stay NON-EMPTY. It is unused by the audio-cloning
    backends, but F5 reads an empty reference text as "transcribe the reference
    clip for me" and pulls in its Whisper preprocessor — which would make this
    probe load a second model stack just to test the first, on a file that is
    pure silence. That difference is the one thing the two copies of this
    function disagreed on before they were merged.

    Any exception means "this device cannot serve the model", which is a
    recoverable answer (the caller drops to CPU), never an error to propagate.
    """
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "selftest.wav"
            _write_silence(ref, worker.sample_rate)
            worker.infer(str(ref), SELF_TEST_REF_TEXT, "test", nfe_step=8, speed=1.0)
        return True
    except Exception:  # noqa: BLE001 — any failure means "fall back to CPU"
        return False
