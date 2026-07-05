"""Per-language model resolution and checkpoint download (SPEC §14.3).

A wrapper process serves exactly one language, chosen at startup
(``config.language``). The model for each language is defined in
``config.languages``:

- ``en`` → the official multilingual base ``F5TTS_v1_Base`` (auto-downloaded by
  F5-TTS itself; no explicit checkpoint/vocab).
- ``de``/``fr``/``ja`` → community finetunes (CC-BY-NC) whose checkpoint + vocab
  are fetched from HuggingFace into ``models/`` and passed to F5-TTS as local
  files.

The same :func:`resolve_model` is used by the bootstrap (to pre-download every
language at install time) and by the engine (to load the active language at
startup) — so there is one source of truth for repo/file mapping. HuggingFace
and F5-TTS imports are lazy so this module imports without those heavy deps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import ndjson, progress
from .config import Config, load_config


@dataclass
class ResolvedModel:
    arch: str  # F5-TTS architecture name (F5TTS_v1_Base | F5TTS_Base)
    ckpt_file: str = ""  # local path to checkpoint, "" → let F5-TTS auto-download
    vocab_file: str = ""  # local path to vocab.txt, "" → architecture default


def language_entry(config: Config, language: Optional[str] = None) -> dict:
    lang = language or config.language
    entry = config.languages.get(lang)
    if entry is None:
        raise ValueError(
            f"unknown language {lang!r}; configured: {sorted(config.languages)}"
        )
    return entry


def resolve_model(config: Config, language: Optional[str] = None) -> ResolvedModel:
    """Resolve a language to a concrete (arch, checkpoint, vocab).

    For finetunes this downloads (or reuses the cache for) the checkpoint and
    vocab via ``hf_hub_download`` and returns local file paths. For the base
    model it returns empty paths so F5-TTS handles the download itself.
    """
    entry = language_entry(config, language)
    arch = entry.get("arch", "F5TTS_v1_Base")
    repo = entry.get("hf_repo")
    if not repo:
        return ResolvedModel(arch=arch)

    from huggingface_hub import hf_hub_download  # lazy

    cache_dir = str(config.models_path)
    ckpt = hf_hub_download(repo_id=repo, filename=entry["ckpt_file"], cache_dir=cache_dir)
    vocab = ""
    if entry.get("vocab_file"):
        vocab = hf_hub_download(
            repo_id=repo, filename=entry["vocab_file"], cache_dir=cache_dir
        )
    return ResolvedModel(arch=arch, ckpt_file=ckpt, vocab_file=vocab)


def download_all(config: Config) -> None:
    """Pre-download the checkpoints for every configured language (install time).

    Idempotent: HuggingFace caching means re-runs are fast. Each language is
    reported as an NDJSON ``log`` line (start/done) plus per-file ``progress``
    events with a live percentage, via the tqdm hook in :mod:`src.progress`.
    """
    config.models_path.mkdir(parents=True, exist_ok=True)
    mp = progress.ModelProgress()
    # One patch around the whole loop: huggingface_hub binds its tqdm reference
    # on first import, so re-patching per language would leave later languages
    # reporting under the first one's label. Instead the subclass reads the
    # current label from ``mp`` (set per language via ``stage``).
    with mp.patch():
        for lang, entry in config.languages.items():
            mp.stage(f"{lang}: ")
            ndjson.log(f"Lade Modell '{lang}' …")
            if entry.get("hf_repo"):
                resolve_model(config, lang)
            else:
                # Base model: instantiate F5-TTS to trigger its own weight download.
                from f5_tts.api import F5TTS  # lazy

                F5TTS(model=entry.get("arch", "F5TTS_v1_Base"), hf_cache_dir=str(config.models_path))
            ndjson.log(f"Modell '{lang}' bereit")


if __name__ == "__main__":
    # Entry point used by the bootstrap (run inside the venv): downloads every
    # configured language's checkpoint into models/.
    import sys
    import traceback

    try:
        download_all(load_config())
    except Exception as exc:  # noqa: BLE001 — surface as a fatal NDJSON error
        traceback.print_exc(file=sys.stderr)
        ndjson.error(f"model download failed: {exc}", fatal=True)
        raise SystemExit(1)
