"""Inline voice references: reference audio that arrives IN the request.

Why this exists (plan ``echokraut/docs/plans/game-voice-reference.md`` §2.4):
the plugin can clone a voice from clips of the user's own game installation.
Those clips must never touch the disk — not in ``samples/``, not as a temp file
— so ``/tts`` takes them inline (``ref_audio``: base64 PCM s16le, 24 kHz mono,
plus ``ref_text``: the game text of the clip) and this module turns that into an
:class:`InlineReference` that lives in RAM only.

Each worker gets the reference object itself instead of a path. A worker that
can condition from memory says so with ``supports_inline_reference = True`` and
caches its conditioning (XTTS latents, F5's prepared reference tensor, MOSS
prompt codes) in a :class:`ConditioningCache` keyed by :attr:`InlineReference.voice_id`.

Deliberately NOT here: any ASR for a missing ref text (the game text is always
there — ``asr_for_missing_ref_text`` never applies to inline references), and
the sidecar transcript cache of :mod:`samples` (it writes ``.txt`` files).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Generic, Optional, Sequence, TypeVar

import numpy as np

# The one wire format for inline audio. Fixed on purpose: the plugin already
# resamples to this (it is the rate every engine here outputs), and a fixed
# format means the server never has to guess or decode a container.
INLINE_SAMPLE_RATE = 24000
# Below this a reference carries no usable voice; it is a client bug, not a voice.
MIN_REFERENCE_SECONDS = 1.0


class InvalidInlineReference(ValueError):
    """The inline audio/text is malformed or out of range (HTTP 400)."""


@dataclass(frozen=True)
class InlineReference:
    """A reference voice held in memory for exactly one request.

    ``voice_id`` identifies the voice for the conditioning caches; ``audio`` is
    float32 mono in [-1, 1] at :data:`INLINE_SAMPLE_RATE`.
    """

    voice_id: str
    audio: np.ndarray = field(repr=False, compare=False)
    # Out of repr as well: a repr ends up in tracebacks and logs, and this is
    # game text the plugin sent along with game audio.
    text: str = field(repr=False)
    sample_rate: int = INLINE_SAMPLE_RATE

    @property
    def seconds(self) -> float:
        return self.audio.shape[-1] / float(self.sample_rate)

    @property
    def label(self) -> str:
        """Short, non-reversible tag for log lines (never the audio, never the text)."""
        return f"inline:{self.voice_id[:12]}"


def decode_pcm16(b64: str) -> np.ndarray:
    """base64 PCM s16le mono → float32 in [-1, 1]. Raises on malformed input."""
    if not b64:
        raise InvalidInlineReference("ref_audio is empty")
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidInlineReference(f"ref_audio is not valid base64: {exc}") from exc
    if len(raw) == 0:
        raise InvalidInlineReference("ref_audio is empty")
    if len(raw) % 2:
        raise InvalidInlineReference("ref_audio must be 16-bit PCM (even byte count)")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def reference_key(ref_ids: Optional[Sequence[str]], audio: np.ndarray, text: str) -> str:
    """Cache key for an inline reference.

    With clip ids (what the plugin sends: the game clips the reference was cut
    from) the key is the hash of those ids, as the plan specifies — the same
    clips are the same voice. Without ids the key is the hash of audio + text,
    so an anonymous caller still gets a correct (if slower to compute) key and
    never someone else's conditioning.
    """
    h = hashlib.sha256()
    ids = [i for i in (ref_ids or []) if i]
    if ids:
        h.update(b"ids\0")
        for i in ids:
            h.update(i.encode("utf-8"))
            h.update(b"\0")
    else:
        h.update(b"pcm\0")
        h.update(np.ascontiguousarray(audio).tobytes())
        h.update(b"\0")
        h.update(text.encode("utf-8"))
    return h.hexdigest()


def build_reference(
    ref_audio_b64: str,
    ref_text: Optional[str],
    ref_ids: Optional[Sequence[str]] = None,
    max_seconds: float = 15.0,
) -> InlineReference:
    """Validate and build an :class:`InlineReference` (all in memory).

    ``ref_text`` is required: it is the game text of the clip, and F5 needs it
    to align the reference. ASR is never used to fill it in.
    """
    text = (ref_text or "").strip()
    if not text:
        raise InvalidInlineReference("ref_text is required with ref_audio")
    audio = decode_pcm16(ref_audio_b64)
    seconds = audio.shape[-1] / float(INLINE_SAMPLE_RATE)
    if seconds < MIN_REFERENCE_SECONDS:
        raise InvalidInlineReference(
            f"ref_audio is {seconds:.2f}s, at least {MIN_REFERENCE_SECONDS:.0f}s required"
        )
    if seconds > max_seconds:
        raise InvalidInlineReference(
            f"ref_audio is {seconds:.2f}s, at most {max_seconds:g}s allowed "
            f"(inline_ref_max_seconds)"
        )
    return InlineReference(voice_id=reference_key(ref_ids, audio, text), audio=audio, text=text)


def ensure_sentence_end(text: str) -> str:
    """F5 aligns ``ref_text + gen_text`` as one string; without a sentence end
    between them the reference's last word bleeds into the first generated one.
    Mirrors what ``f5_tts.infer.utils_infer.preprocess_ref_audio_text`` does for
    file references, which the inline path has to bypass (it writes a temp WAV).
    """
    if text.endswith(". ") or text.endswith("。"):
        return text
    if text.endswith("."):
        return text + " "
    return text + ". "


V = TypeVar("V")


class ConditioningCache(Generic[V]):
    """Small LRU for per-voice conditioning, RAM only.

    Bounded because the keys come from requests: an unbounded dict keyed by
    caller input is a memory leak with extra steps. ``capacity <= 0`` disables
    caching (every request recomputes).
    """

    def __init__(self, capacity: int):
        self._capacity = int(capacity)
        self._items: "OrderedDict[str, V]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def get_or_compute(self, key: str, compute: Callable[[], V]) -> V:
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        value = compute()
        if self._capacity > 0:
            self._items[key] = value
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)
        return value
