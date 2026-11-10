"""Speech recognition from memory, for ``POST /asr-check``.

The plugin ranks candidate game clips by how well an ASR transcript matches the
game text (plan ``game-voice-reference.md`` §4, Teil C). The audio is game audio,
so it is transcribed straight from the request buffer — nothing is written.

Reuses the Whisper pipeline F5-TTS already ships (``f5_tts.infer.utils_infer``,
``openai/whisper-large-v3-turbo``) instead of pulling a second ASR stack. The
bootstrap installs every engine whatever backend is active, so the import works
under XTTS and MOSS as well. The ``transformers`` pipeline takes a raw array
(``{"raw": …, "sampling_rate": …}``) and resamples it itself — no file, unlike
``F5TTSWorker.transcribe``, which goes through F5's temp-WAV preprocessing and
must therefore never see game audio.

The pipeline is a process-wide global inside F5 and not thread-safe; the server
serialises calls (``asyncio.Lock``), so this module does not lock.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

# (float32 mono audio, sample rate, language code or None) -> text
PcmTranscriber = Callable[[np.ndarray, int, Optional[str]], str]


def whisper_transcribe(audio: np.ndarray, sample_rate: int, language: Optional[str] = None) -> str:
    """Transcribe in-memory audio with F5's Whisper pipeline (lazy import)."""
    from f5_tts.infer.utils_infer import transcribe  # lazy: torch/transformers

    payload = {"raw": np.asarray(audio, dtype=np.float32), "sampling_rate": int(sample_rate)}
    return (transcribe(payload, language=language) or "").strip()
