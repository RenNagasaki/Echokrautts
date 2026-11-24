"""Pocket TTS (Kyutai, CPU) backend — engine ``pocket``.

Same worker contract as the other engines (float32 @ 24 kHz per streamed part).
Clones from the reference AUDIO alone (no transcript), one model per language
like F5 — so a language switch is a restart that downloads that language.

Why it is here (user decision 2026-11-12, after listening tests with German
FFXIV references): runs on the CPU, leaving the graphics card to the game, and
sounded better than MOSS. Measured on a Ryzen (german_24l): WER 0.4 % on medium
sentences, speaker similarity 0.60, rtf 0.63 / 0.73 / 0.85 at
``sampler_decode_steps`` 1 / 4 / 8 — 1 occasionally adds reverb, 4 and 8 do
not, so 4 is the default.

Weights: Kyutai's ``kyutai/pocket-tts`` (the voice-CLONING weights) is gated on
HuggingFace. They are CC-BY-4.0, which permits redistribution, so they come from
a mirror pinned to one commit (``config.pocket_repo`` / ``pocket_revision``).
⚠ The mirror (a byte-identical copy of ``lunahr/pocket-tts-ungated``) is an OLDER revision and only works
with its own sentencepiece ``tokenizer.model`` — the package's configs point at
the newer ``tokenizer.json``. We therefore write our own config per language:
the package's template with ``weights_path`` and the tokenizer swapped for the
downloaded files, so nothing reaches for the gated repo at load time.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from . import hfcache, i18n, ndjson, progress, selftest
from .config import Config, load_config
from .inline_ref import ConditioningCache, InlineReference
from .streaming import drain_stream

DEFAULT_SAMPLE_RATE = 24000
MODEL_FILES = ("model.safetensors", "tokenizer.model")


def cpu_threads(configured: int, cores: Optional[int] = None) -> int:
    """Threads for torch: the configured value, or automatic = at most 4."""
    if int(configured) > 0:
        return int(configured)
    return max(1, min(4, cores if cores is not None else (os.cpu_count() or 1)))


LAYER_SIZES = (24, 6)
"""Model sizes the mirror carries. 24 sounds better, 6 is ~2.4x faster."""


def layers(config: Config) -> int:
    """The configured size, 24 for anything unknown (the better-sounding default)."""
    value = int(config.pocket_layers or 24)
    return value if value in LAYER_SIZES else 24


def decode_steps(config: Config) -> int:
    """``pocket_decode_steps``, but never 1 for the 6-layer model: measured
    (2026-11-13) its WER more than doubles there (1.8 % → 4.6 %), and the user
    decided the small model is offered with 4 and 8 only."""
    steps = int(config.pocket_decode_steps)
    if layers(config) == 6 and steps < 4:
        ndjson.log_once(i18n.t("pocket.decode_steps_raised"), level="warning")
        return 4
    return steps


def language_folder(config: Config, language: Optional[str] = None) -> str:
    """The mirror's folder for a wrapper language code and the configured size
    (``de`` + 24 → ``german_24l``). A size the mirror lacks for this language
    (24 layers: no English; 6 layers: no French) falls back to the other one,
    with a warning — a working voice beats a refusal over the model size."""
    code = (language or config.language or "").lower()
    wanted = layers(config)
    for size in (wanted, *(s for s in LAYER_SIZES if s != wanted)):
        folder = (config.pocket_languages.get(str(size)) or {}).get(code)
        if folder:
            if size != wanted:
                ndjson.log_once(
                    i18n.t("pocket.layers_fallback", code=repr(code), wanted=wanted, size=size),
                    level="warning",
                )
            return str(folder)
    available = sorted({c for table in config.pocket_languages.values() for c in (table or {})})
    raise ValueError(
        f"Pocket TTS hat kein Modell für Sprache {code!r} "
        f"(verfügbar: {', '.join(available)}). "
        "Wähle eine andere Engine oder Sprache."
    )


def _model_dir(config: Config) -> Path:
    """``models/pocket/<revision>`` — the revision in the path keeps a mirror
    switch from mixing files of two revisions."""
    return config.models_path / "pocket" / str(config.pocket_revision)[:12]


def _language_files(config: Config, folder: str) -> dict[str, Path]:
    base = _model_dir(config) / "languages" / folder
    return {name: base / name for name in MODEL_FILES}


def write_config(config: Config, folder: str, template_dir: Path) -> Path:
    """Our config for ``folder``: the package template, pointed at the local files."""
    files = _language_files(config, folder)
    text = (template_dir / f"{folder}.yaml").read_text(encoding="utf-8")
    text = re.sub(r"^weights_path: .*$", "weights_path: " + files["model.safetensors"].as_posix(),
                  text, count=1, flags=re.M)
    text = re.sub(r"^(\s*)tokenizer: .*$", r"\1tokenizer: sentencepiece", text, count=1, flags=re.M)
    text = re.sub(r"^(\s*)tokenizer_path: .*$",
                  lambda m: m.group(1) + "tokenizer_path: " + files["tokenizer.model"].as_posix(),
                  text, count=1, flags=re.M)
    target = _model_dir(config) / f"{folder}.yaml"
    target.write_text(text, encoding="utf-8")
    return target


def _package_config_dir() -> Path:
    import pocket_tts  # lazy

    return Path(pocket_tts.__file__).parent / "config"


class PocketWorker:
    """One Pocket TTS model on the CPU, for the startup language."""

    supports_streaming = True
    # get_state_for_audio_prompt takes a tensor too, so inline references need no file.
    supports_inline_reference = True

    def __init__(self, config: Config, device: str, model_loader=None, template_dir: Optional[Path] = None):
        # Pocket is a CPU model by design (its whole point here); the device the
        # engine detected is the game's.
        self.device = "cpu"
        self.sample_rate = DEFAULT_SAMPLE_RATE
        folder = language_folder(config)
        self._states: ConditioningCache = ConditioningCache(max(int(config.inline_ref_cache_size), 64))
        if model_loader is None:
            hfcache.use_models_dir(config)
            import torch  # lazy

            torch.set_num_threads(cpu_threads(config.pocket_threads))
            from pocket_tts import TTSModel  # lazy

            model_loader = TTSModel.load_model
        cfg = write_config(config, folder, template_dir or _package_config_dir())
        steps = decode_steps(config)
        self._model = model_loader(config=str(cfg), sampler_decode_steps=steps)
        sr = getattr(self._model, "sample_rate", None)
        if isinstance(sr, int) and sr > 0:
            self.sample_rate = sr
        self._prime()
        ndjson.log_once(
            f"Pocket TTS: {folder}, sampler_decode_steps={steps}, "
            f"Quelle {config.pocket_repo}@{str(config.pocket_revision)[:7]}"
        )

    def _prime(self) -> None:
        """Speak one throwaway line at startup — measured: the first real request
        otherwise waited 1.9 s for its first audio, every later one 0.3 s.
        Never fatal: a failed priming only means the first line is slow."""
        try:
            import torch  # lazy

            noise = torch.randn(1, self.sample_rate) * 0.01
            state = self._model.get_state_for_audio_prompt(noise)
            for _ in self._model.generate_audio_stream(state, "Hallo."):
                pass
        except Exception as exc:  # noqa: BLE001 — see docstring
            ndjson.log_once(i18n.t("engine.warmup_failed", name="Pocket", error=exc), level="warning")

    def _state(self, ref: "str | InlineReference"):
        if isinstance(ref, InlineReference):
            def encode():
                import torch  # lazy

                return self._model.get_state_for_audio_prompt(
                    torch.from_numpy(np.asarray(ref.audio, dtype=np.float32)).reshape(1, -1)
                )

            return self._states.get_or_compute(f"inline:{ref.voice_id}", encode)
        path = Path(ref)
        mtime = path.stat().st_mtime_ns if path.exists() else 0
        return self._states.get_or_compute(
            f"file:{path}:{mtime}", lambda: self._model.get_state_for_audio_prompt(path)
        )

    def infer_stream(
        self,
        ref_file: "str | InlineReference",
        ref_text: str,
        gen_text: str,
        speed: float,
        language: Optional[str] = None,
    ) -> Iterator[np.ndarray]:
        """``ref_text`` is unused (Pocket clones from the audio) and ``language``
        is fixed per process (one model per language, like F5)."""
        ndjson.warn_ignored_speed("Pocket TTS", speed)
        state = self._state(ref_file)
        for chunk in self._model.generate_audio_stream(state, gen_text):
            arr = chunk.detach().float().cpu().numpy() if hasattr(chunk, "detach") else np.asarray(chunk)
            yield np.asarray(arr, dtype=np.float32).reshape(-1)

    def infer(
        self,
        ref_file: "str | InlineReference",
        ref_text: str,
        gen_text: str,
        nfe_step: int,
        speed: float,
        language: Optional[str] = None,
    ) -> np.ndarray:
        return drain_stream(self.infer_stream(ref_file, ref_text, gen_text, speed, language))

    def transcribe(self, audio_path: Path) -> str:
        return ""  # clones from the audio alone

    def self_test(self) -> bool:
        return selftest.run(self)


def download_model(config: Config) -> None:
    """Fetch the ACTIVE language's model + tokenizer from the pinned mirror. Idempotent."""
    folder = language_folder(config)  # fails here, before a download, for e.g. Japanese
    config.models_path.mkdir(parents=True, exist_ok=True)
    hfcache.use_models_dir(config)
    target = _model_dir(config)
    mp = progress.ModelProgress()
    mp.stage(f"Pocket {folder}: ")
    ndjson.log(i18n.t("model.loading", name=f"Pocket TTS ({folder}, {config.pocket_repo})"))
    with mp.patch():
        from huggingface_hub import snapshot_download  # lazy: imported inside the patch

        snapshot_download(
            repo_id=str(config.pocket_repo),
            revision=str(config.pocket_revision),
            allow_patterns=[f"languages/{folder}/{name}" for name in MODEL_FILES],
            local_dir=str(target),
        )
    missing = [p for p in _language_files(config, folder).values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Pocket-Dateien fehlen nach dem Download: {', '.join(map(str, missing))}")
    ndjson.log(i18n.t("model.ready", name=f"Pocket TTS ({folder})"))


if __name__ == "__main__":
    # Entry point used by the bootstrap (run inside the venv) for the Pocket engine.
    import sys
    import traceback

    try:
        download_model(load_config())
    except Exception as exc:  # noqa: BLE001 — surface as a fatal NDJSON error
        traceback.print_exc(file=sys.stderr)
        ndjson.error(i18n.t("model.download_failed", name="Pocket TTS", error=exc), fatal=True)
        raise SystemExit(1)
