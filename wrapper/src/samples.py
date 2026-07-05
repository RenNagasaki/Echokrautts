"""Voice-sample resolution, validation and reference-transcript handling (SPEC §7).

Requests carry only a *basename* (e.g. ``anna_de.wav``). We resolve it against
the configured ``samples/`` directory and verify the resolved path really lives
inside that directory — no ``..``, no absolute paths, no symlinks escaping out.

A voice can be either:

* a **single audio file** (``samples/anna_de.wav``), as before, or
* a **voice folder** (``samples/anna_de/``) holding several audio files of the
  *same* voice. One file is picked at random **per request** so the output has
  some natural variance. Folders are one level only — sub-directories are never
  descended into.

The request name's extension is *ignored*: only its stem matters. ``anna_de`` and
``anna_de.wav`` resolve identically. A folder always wins over a same-stem single
file (``samples/anna_de/`` shadows ``samples/anna_de.wav``).

Reference transcript resolution order for ``ref_text`` (applied to the *chosen*
audio file):
1. a sidecar ``<stem>.txt`` next to the sample,
2. the ``ref_text`` supplied in the request,
3. ASR transcription (if enabled and a transcriber is injected), cached to a
   ``.txt`` sidecar when the directory is writable, otherwise in memory.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, Optional, Sequence

from .config import Config

# A transcriber maps an audio file path to its spoken text. Injected by the
# engine (Whisper/F5-TTS ASR) so this module stays model-free and testable.
Transcriber = Callable[[Path], str]
# Picks one file from a non-empty list of candidates. Injected so tests can make
# the random selection deterministic; defaults to :func:`random.choice`.
Chooser = Callable[[Sequence[Path]], Path]


class InvalidSample(Exception):
    """Sample name is malformed or escapes the samples directory (HTTP 400)."""


class SampleNotFound(Exception):
    """Sample name is valid but no such file exists (HTTP 404)."""


class SampleService:
    def __init__(
        self,
        config: Config,
        transcriber: Optional[Transcriber] = None,
        chooser: Optional[Chooser] = None,
    ):
        self._config = config
        self._transcriber = transcriber
        self._choose: Chooser = chooser or random.choice
        self._dir = config.samples_path
        # In-memory ASR cache used when the samples dir is not writable.
        self._mem_ref_cache: dict[str, str] = {}

    # ------------------------------------------------------------------ paths
    def resolve_path(self, name: str) -> Path:
        """Resolve a request sample name to an existing audio file inside samples/.

        A voice folder (``samples/<stem>/``) wins over a same-stem single file and
        contributes one randomly chosen file per call. The request extension is
        ignored — only the stem is matched.

        Raises :class:`InvalidSample` (400) for traversal/separator/absolute
        inputs or paths that resolve outside the directory, and
        :class:`SampleNotFound` (404) when nothing matches the stem.
        """
        if not name or name in (".", ".."):
            raise InvalidSample(f"invalid sample name: {name!r}")
        # Reject anything that isn't a bare basename before touching the FS.
        if name != Path(name).name or os.sep in name or (os.altsep and os.altsep in name):
            raise InvalidSample(f"sample must be a bare filename: {name!r}")

        stem = self._stem(name)
        if not stem or stem in (".", ".."):
            raise InvalidSample(f"invalid sample name: {name!r}")

        base = self._dir.resolve()

        # 1) Voice folder wins over a same-stem single file.
        folder = base / stem
        if folder.is_dir() or folder.is_symlink():
            resolved = folder.resolve()
            if resolved.parent != base:
                raise InvalidSample(f"sample escapes samples directory: {name!r}")
            if resolved.is_dir():
                candidates = self._audio_files_in(resolved)
                if not candidates:
                    raise SampleNotFound(f"voice folder has no usable samples: {name!r}")
                return self._choose(candidates)

        # 2) Single file by stem; extension from the request is ignored, so we
        #    try the configured extensions in priority order and take the first.
        for ext in self._ext_priority():
            candidate = base / f"{stem}{ext}"
            if candidate.is_symlink() or candidate.exists():
                resolved = candidate.resolve()
                # resolve() follows symlinks, so one pointing outside is caught here.
                if resolved.parent != base:
                    raise InvalidSample(f"sample escapes samples directory: {name!r}")
                if resolved.is_file():
                    return resolved
        raise SampleNotFound(f"sample not found: {name!r}")

    def _stem(self, name: str) -> str:
        """Strip a trailing audio extension; leave any other name untouched."""
        if Path(name).suffix.lower() in self._config.normalized_exts:
            return Path(name).stem
        return name

    def _ext_priority(self) -> list[str]:
        """Allowed extensions, normalized and de-duplicated in config order."""
        out: list[str] = []
        for e in self._config.allowed_sample_ext:
            e = e.lower()
            e = e if e.startswith(".") else f".{e}"
            if e not in out:
                out.append(e)
        return out

    def _audio_files_in(self, folder: Path) -> list[Path]:
        """Sorted, guarded audio files directly inside a resolved voice folder.

        One level only; symlinks escaping the folder are dropped. Sorted so an
        injected (seeded) chooser is reproducible.
        """
        exts = self._config.normalized_exts
        out: list[Path] = []
        for p in folder.iterdir():
            if p.suffix.lower() not in exts:
                continue
            rp = p.resolve()
            if rp.is_file() and rp.parent == folder:
                out.append(rp)
        return sorted(out)

    # ------------------------------------------------------------------ listing
    def list_samples(self, details: bool = False):
        """Return the list of usable voices (SPEC §5.2).

        A voice is a single audio file (listed with its extension, as before) or
        a voice folder (listed by folder name). A folder shadows a same-stem file.

        ``details=False`` → ``{"samples": [name, ...]}``.
        ``details=True``  → ``{"samples": [{name, has_ref_text, bytes, count}, ...]}``
        where ``count`` is the number of underlying audio files (1 for a single
        file, N for a folder), ``bytes`` their total size, and ``has_ref_text`` is
        true if at least one has a ``.txt`` sidecar.
        """
        if not self._dir.is_dir():
            return {"samples": []}
        exts = self._config.normalized_exts
        base = self._dir
        rbase = base.resolve()

        voices: list[tuple[str, list[Path]]] = []
        folder_stems: set[str] = set()
        # Folders first so a same-stem single file can be shadowed below.
        for p in sorted(base.iterdir()):
            if not p.is_dir():
                continue
            resolved = p.resolve()
            if resolved.parent != rbase:  # symlinked folder escaping the dir
                continue
            files = self._audio_files_in(resolved)
            if files:
                voices.append((p.name, files))
                folder_stems.add(p.name)
        for p in sorted(base.iterdir()):
            if p.is_file() and p.suffix.lower() in exts and p.stem not in folder_stems:
                voices.append((p.name, [p]))
        voices.sort(key=lambda v: v[0])

        if not details:
            return {"samples": [name for name, _ in voices]}
        out = []
        for name, files in voices:
            out.append(
                {
                    "name": name,
                    "has_ref_text": any(self._sidecar(f).is_file() for f in files),
                    "bytes": sum(f.stat().st_size for f in files),
                    "count": len(files),
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

        # Cached ASR result from a previous request in this process. Keyed by the
        # full path so identically-named files in different folders don't collide.
        key = str(audio_path)
        cached = self._mem_ref_cache.get(key)
        if cached is not None:
            return cached

        if self._config.asr_for_missing_ref_text and self._transcriber is not None:
            text = self._transcriber(audio_path).strip()
            self._mem_ref_cache[key] = text
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
