"""Speech recognition from memory, for ``POST /asr-check``.

The plugin ranks candidate game clips by how well an ASR transcript matches the
game text (plan ``game-voice-reference.md`` §4, Teil C). The audio is game audio,
so it is transcribed straight from the request buffer — nothing is written.

Two sources for the Whisper pipeline (``openai/whisper-large-v3-turbo``):

* **F5's own** (``f5_tts.infer.utils_infer``) when f5-tts is installed. F5 keeps
  ONE process-wide pipeline and also uses it for its ref-text fallback, so
  sharing it means one Whisper in VRAM, not two.
* **Our own** ``transformers`` pipeline otherwise. Since the bootstrap installs
  only the ACTIVE engine (plan ``qwen3-und-lazy-install.md`` §3.4), f5-tts is
  absent under XTTS, MOSS and Qwen — and /asr-check must keep working there.
  Built with the same model and arguments as F5's, so both paths answer alike.
  Works on transformers 4.57 and 5.x.

The ``transformers`` pipeline takes a raw array (``{"raw": …, "sampling_rate": …}``)
and resamples it itself — no file, unlike ``F5TTSWorker.transcribe``, which goes
through F5's temp-WAV preprocessing and must therefore never see game audio.

Neither pipeline is thread-safe; the server serialises calls (``asyncio.Lock``),
so this module does not lock.
"""

from __future__ import annotations

import contextlib
import importlib
from typing import Any, Callable, Iterator, Optional

import numpy as np

# (float32 mono audio, sample rate, language code or None) -> text
PcmTranscriber = Callable[[np.ndarray, int, Optional[str]], str]

WHISPER_MODEL = "openai/whisper-large-v3-turbo"  # same as F5-TTS

_own_pipeline: Any = None

# Libraries whose warnings Whisper triggers on every build/call and that mean
# nothing to a user: transformers (`torch_dtype` deprecated, "chunk_length_s is
# very experimental", logits processors, clean_up_tokenization_spaces) and
# huggingface_hub (unauthenticated HF Hub requests). Errors still get through.
_NOISY_LOGGING = ("transformers.utils.logging", "huggingface_hub.utils.logging")


@contextlib.contextmanager
def _quiet_libraries() -> Iterator[None]:
    """Errors only from transformers/huggingface_hub while Whisper runs; the
    previous verbosity comes back afterwards, so the TTS engine keeps its own."""
    saved = []
    for name in _NOISY_LOGGING:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        saved.append((module, module.get_verbosity()))
        module.set_verbosity_error()
    try:
        yield
    finally:
        for module, verbosity in saved:
            module.set_verbosity(verbosity)


def _f5_transcribe() -> Optional[Callable]:
    """F5's transcribe function, or None when f5-tts is not installed."""
    try:
        from f5_tts.infer.utils_infer import transcribe  # lazy: torch/transformers
    except ImportError:
        return None
    return transcribe


def _build_pipeline(factory: Optional[Callable] = None, torch_module: Any = None) -> Any:
    """Whisper pipeline on the best local device (fp16 on a GPU, as F5 does)."""
    if torch_module is None:
        import torch as torch_module  # lazy
    if factory is None:
        from transformers import pipeline as factory  # lazy

    if torch_module.cuda.is_available():
        device, dtype = "cuda:0", torch_module.float16
    else:
        device, dtype = "cpu", torch_module.float32
    # ``torch_dtype`` is the spelling both transformers 4.57 and 5.x accept.
    return factory(
        "automatic-speech-recognition", model=WHISPER_MODEL, torch_dtype=dtype, device=device
    )


def _own_transcribe(payload: dict, language: Optional[str]) -> str:
    global _own_pipeline
    if _own_pipeline is None:
        _own_pipeline = _build_pipeline()
    generate_kwargs = {"task": "transcribe"}
    if language:
        generate_kwargs["language"] = language
    result = _own_pipeline(
        payload, chunk_length_s=30, batch_size=128,
        generate_kwargs=generate_kwargs, return_timestamps=False,
    )
    return str(result.get("text", "") if isinstance(result, dict) else result)


def whisper_transcribe(audio: np.ndarray, sample_rate: int, language: Optional[str] = None) -> str:
    """Transcribe in-memory audio with Whisper (F5's pipeline when present)."""
    payload = {"raw": np.asarray(audio, dtype=np.float32), "sampling_rate": int(sample_rate)}
    f5 = _f5_transcribe()
    with _quiet_libraries():
        text = f5(payload, language=language) if f5 is not None else _own_transcribe(payload, language)
    return (text or "").strip()
