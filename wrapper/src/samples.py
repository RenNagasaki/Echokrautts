"""Voice-sample resolution, validation and reference-transcript handling (SPEC §7).

Requests carry only a *basename* (e.g. ``anna_de.wav``). We resolve it against
the configured ``samples/`` directory and verify the resolved path really lives
inside that directory — no ``..``, no absolute paths, no symlinks escaping out.

Reference transcript resolution order for ``ref_text``:
1. a sidecar ``<stem>.txt`` next to the sample,
2. the ``ref_text`` supplied in the request,
3. ASR transcription (if enabled and a transcriber is injected), cached to a
   ``.txt`` sidecar when the directory is writable, otherwise in memory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from .config import Config

# A transcriber maps an audio file path to its spoken text. Injected by the
# engine (Whisper/F5-TTS ASR) so this module stays model-free and testable.
Transcriber = Callable[[Path], str]


class InvalidSample(Exception):
    """Sample name is malformed or escapes the samples directory (HTTP 400)."""


class SampleNotFound(Exception):
    """Sample name is valid but no such file exists (HTTP 404)."""


class SampleService:
    def __init__(self, config: Config, transcriber: Optional[Transcriber] = None):
        self._config = config
        self._transcriber = transcriber
        self._dir = config.samples_path
        # In-memory ASR cache used when the samples dir is not writable.
        self._mem_ref_cache: dict[str, str] = {}

    # ------------------------------------------------------------------ paths
    def resolve_path(self, name: str) -> Path:
        """Resolve a request sample name to an existing file inside samples/.

        Raises :class:`InvalidSample` (400) for traversal/separator/absolute
        inputs or paths that resolve outside the directory, and
        :class:`SampleNotFound` (404) when the file does not exist.
        """
        if not name or name in (".", ".."):
            raise InvalidSample(f"invalid sample name: {name!r}")
        # Reject anything that isn't a bare basename before touching the FS.
        if name != Path(name).name or os.sep in name or (os.altsep and os.altsep in name):
            raise InvalidSample(f"sample must be a bare filename: {name!r}")

        base = self._dir.resolve()
        candidate = (base / Path(name).name).resolve()
        # resolve() follows symlinks, so a symlink pointing outside is caught here.
        if candidate.parent != base:
            raise InvalidSample(f"sample escapes samples directory: {name!r}")
        if candidate.suffix.lower() not in self._config.normalized_exts:
            raise InvalidSample(f"sample extension not allowed: {name!r}")
        if not candidate.is_file():
            raise SampleNotFound(f"sample not found: {name!r}")
        return candidate

    # ------------------------------------------------------------------ listing
    def list_samples(self, details: bool = False):
        """Return the list of usable sample names (SPEC §5.2).

        ``details=False`` → ``{"samples": [name, ...]}``.
        ``details=True``  → ``{"samples": [{name, has_ref_text, bytes}, ...]}``.
        """
        if not self._dir.is_dir():
            return {"samples": []}
        exts = self._config.normalized_exts
        names = sorted(
            p.name
            for p in self._dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )
        if not details:
            return {"samples": names}
        out = []
        for name in names:
            p = self._dir / name
            out.append(
                {
                    "name": name,
                    "has_ref_text": self._sidecar(p).is_file(),
                    "bytes": p.stat().st_size,
                }
            )
        return {"samples": out}

    # ------------------------------------------------------------------ ref text
    @staticmethod
    def _sidecar(audio_path: Path) -> Path:
        return audio_path.with_suffix(".txt")

    def resolve_ref_text(
        self, audio_path: Path, request_ref_text: Optional[str] = None
    ) -> Optional[str]:
        """Resolve the reference transcript for a sample (see module docstring)."""
        sidecar = self._sidecar(audio_path)
        if sidecar.is_file():
            text = sidecar.read_text(encoding="utf-8").strip()
            if text:
                return text

        if request_ref_text:
            return request_ref_text.strip()

        # Cached ASR result from a previous request in this process.
        cached = self._mem_ref_cache.get(audio_path.name)
        if cached is not None:
            return cached

        if self._config.asr_for_missing_ref_text and self._transcriber is not None:
            text = self._transcriber(audio_path).strip()
            self._mem_ref_cache[audio_path.name] = text
            self._try_write_sidecar(sidecar, text)
            return text

        # No transcript available; caller decides how F5-TTS handles this.
        return None

    @staticmethod
    def _try_write_sidecar(sidecar: Path, text: str) -> None:
        try:
            sidecar.write_text(text, encoding="utf-8")
        except OSError:
            # Read-only samples dir → keep the in-memory cache only.
            pass
