"""What both MOSS runtimes share.

MOSS runs on two runtimes — the PyTorch one it ships with and the ONNX export
(:mod:`moss_backend`, :mod:`moss_onnx_backend`). They differ only in how audio
is produced; everything the wrapper's worker protocol asks for around that is
identical, and writing it twice already cost something real: the ONNX worker
shipped without ``self_test``, so on a dml/xpu/rocm_win machine the engine's
fragile-device path reported "worker init failed" about a worker that had built
perfectly well, and rebuilt it on the CPU for no reason.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from . import ndjson, selftest

# The wrapper's HTTP contract: s16 mono at this rate. MOSS emits 48 kHz stereo,
# so every chunk is downmixed and resampled on the way out.
DEFAULT_SAMPLE_RATE = 24000


class MossWorkerBase:
    """The protocol parts that do not depend on which runtime produces audio.

    A subclass provides :meth:`infer_stream` and sets ``sample_rate``; this
    supplies the rest. It is deliberately a base class rather than a helper
    module: these are protocol METHODS, and the engine looks them up on the
    worker object.
    """

    #: Overridden per instance from ``config.moss_stream`` by both runtimes.
    supports_streaming = True
    sample_rate = DEFAULT_SAMPLE_RATE

    def infer_stream(
        self,
        ref_file: str,
        ref_text: str,
        gen_text: str,
        speed: float,
        language: str | None = None,
    ) -> Iterator[np.ndarray]:  # pragma: no cover - implemented by the runtimes
        raise NotImplementedError

    def transcribe(self, audio_path: Path) -> str:
        """No-op: MOSS clones from the reference audio, never from a transcript.

        Returning "" rather than raising is the contract — the sample service
        asks every backend and must not need to know which ones care.
        """
        return ""

    def infer(
        self,
        ref_file: str,
        ref_text: str,
        gen_text: str,
        nfe_step: int,
        speed: float,
        language: str | None = None,
    ) -> np.ndarray:
        """One clip, by draining :meth:`infer_stream`.

        Deliberately NOT a separate one-shot call into the runtime, even where
        one exists: one production path means the streamed audio and the
        buffered audio cannot drift apart, and the conversion to the wrapper's
        format lives in exactly one place.
        """
        chunks = list(self.infer_stream(ref_file, ref_text, gen_text, speed, language))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)

    def self_test(self) -> bool:
        """Tiny inference proving the worker can actually run.

        The engine calls this only for fragile devices (dml/xpu/rocm_win). Both
        MOSS runtimes resolve to the CPU there anyway, so it should always pass
        — but it has to EXIST, or the engine's probe raises AttributeError
        inside a `try` that then blames the worker for failing to initialise.
        """
        return selftest.run(self)

    def warn_unsupported_speed(self, speed: float) -> None:
        """Say once that ``speed`` does nothing here.

        MOSS has no equivalent parameter. Silently ignoring a value the caller
        set would look like a wrapper bug from the outside, and logging it per
        request would drown the log — the worker pool builds several workers and
        the line states a fact about the process, not about the request.
        """
        if speed != 1.0:
            ndjson.log_once(
                "MOSS-TTS-Nano kennt keinen speed-Parameter — der Wert wird ignoriert",
                level="warning",
            )
