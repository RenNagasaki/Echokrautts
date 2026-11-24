"""Qwen3-TTS backend via faster-qwen3-tts (CUDA graphs) — engines ``qwen-0.6b`` / ``qwen-1.7b``.

Plan ``docs/plans/qwen3-und-lazy-install.md`` §3.5. Same worker contract as the
other engines: ``infer_stream`` yields flat float32 at 24 kHz per part, ``infer``
drains it (one generation path).

Why faster-qwen3-tts and not Qwen's own ``qwen-tts``: rtf 1.8 on an RTX 5090
with the official package against ~0.3 with CUDA graphs. Its 0.5.x line needs
transformers 5, which coqui/XTTS cannot live with — hence the separate venv
group (``.venv-qwen``); this module must therefore never import anything from the
classic engines.

Voice cloning is ICL (in-context): reference audio AND its transcript. The
prompt built from them is cached per voice (bounded LRU, RAM only) — the first
request per voice otherwise paid up to 5 s for encoding. We build the prompt
ourselves from the audio array and hand it over as ``voice_clone_prompt``
instead of letting the library cache by path: verified bit-identical to the
path route (spike 2026-11-13), and it serves inline references the same way.
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from . import hfcache, i18n, ndjson, progress, selftest
from .config import Config, load_config
from .inline_ref import ConditioningCache, InlineReference
from .streaming import drain_stream

DEFAULT_SAMPLE_RATE = 24000

# Qwen3-TTS takes language NAMES. Codes follow the wrapper's other engines.
QWEN_LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
}
QWEN_LANGUAGES = frozenset(QWEN_LANGUAGE_NAMES)

# The ICL prompt ends with the reference's last codec tokens, so the first
# generated token is conditioned on its last phoneme. Trailing silence makes
# that silence instead (what the library does on its path route).
ICL_TRAILING_SILENCE_S = 0.5


def _model_dir(config: Config) -> Path:
    """Local snapshot of the active engine's repo (``models/qwen/<repo name>``).

    ``local_dir`` instead of the HF cache, like MOSS: no symlinks, which a normal
    Windows account cannot create anyway, and a folder a user can find and
    delete by hand (the only clean-up there is, decided 2026-11-12).
    """
    repo = _repo(config)
    return config.models_path / "qwen" / repo.split("/")[-1]


def _repo(config: Config) -> str:
    try:
        return str(config.qwen_models[config.tts_backend])
    except KeyError:
        raise ValueError(
            f"no Qwen model configured for {config.tts_backend!r} "
            f"(qwen_models: {', '.join(config.qwen_models)})"
        ) from None


def language_name(code: Optional[str], default: str) -> str:
    """Qwen's language name for a wrapper code; ``Auto`` for an unknown one."""
    return QWEN_LANGUAGE_NAMES.get((code or default or "").lower(), "Auto")


class _DropBelowError(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.ERROR


@contextlib.contextmanager
def _quiet_sox() -> Iterator[None]:
    """The ``sox`` package (pulled in by qwen-tts-hf) probes for the SoX binary
    when it is imported with ``os.popen('sox -h')`` and logs "SoX could not be found!".
    We never call SoX — audio is read with soundfile — so both are noise, and
    noise in a user's log costs support time.

    Two holes, two plugs, both only while the model loads (sox is imported
    lazily inside ``from_pretrained``, not by ``import faster_qwen3_tts``):
    - the shell's own "command not found" (on Windows the German
      "Der Befehl \"sox\" ist entweder falsch geschrieben …") goes straight to
      stderr, past any logger → ``os.popen`` gets ``2>`` the null device;
    - ``sox/log.py`` calls ``setLevel(WARNING)`` on import, overriding any
      level set beforehand → a filter instead, which a level change keeps.
    """
    sox_log = logging.getLogger("sox")
    filt = _DropBelowError()
    sox_log.addFilter(filt)
    real_popen = os.popen

    def popen(cmd, *args, **kwargs):
        if isinstance(cmd, str) and cmd.strip().startswith("sox"):
            cmd = f"{cmd} 2>{os.devnull}"
        return real_popen(cmd, *args, **kwargs)

    os.popen = popen
    try:
        yield
    finally:
        os.popen = real_popen
        sox_log.removeFilter(filt)  # one worker per GPU: do not pile up filters


class QwenWorker:
    """One FasterQwen3TTS model bound to one CUDA device."""

    supports_streaming = True
    supports_inline_reference = True
    # One game line = one sampling run: split chunks are sampled independently and
    # drift apart in voice (live 2026-11-15, 255 chars → 2 chunks → new voice
    # mid-line). Qwen streams tokens, so a long chunk does not delay the first audio.
    max_chars_per_chunk = 1000
    # Longer than the 15 s default (engine.inline_ref_max_seconds): two game clips
    # fit, which in a listening test (2026-11-15) removed the runs that came out in
    # a wrong voice. Qwen's prompt is a few hundred codec tokens even at 20 s.
    max_inline_ref_seconds = 20.0

    def __init__(self, config: Config, device: str, model_factory=None):
        if device != "cuda":
            # CUDA graphs need CUDA (ROCm reports itself as cuda). No silent CPU
            # path: Qwen there would be unusably slow (engines.EngineSpec.gpu_backends).
            raise RuntimeError(f"Qwen needs a CUDA device, got {device!r}")
        self.device = device
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self._lang = config.language
        self._backend = config.tts_backend
        self._non_streaming_mode = bool(config.qwen_non_streaming_mode)
        self._temperature = float(config.qwen_temperature)
        self._chunk_size = int(config.qwen_chunk_size)
        # One cache for file samples AND inline references. Sized for a voice
        # pack, not just a conversation: a prompt is a few KB of codes, and a
        # miss costs an encode in the middle of a game line.
        self._prompts: ConditioningCache = ConditioningCache(
            max(int(config.inline_ref_cache_size), 64)
        )
        if model_factory is None:
            hfcache.use_models_dir(config)  # before transformers is imported
            from faster_qwen3_tts import FasterQwen3TTS  # lazy: torch/transformers 5

            model_factory = FasterQwen3TTS.from_pretrained
        local = _model_dir(config)
        source = str(local) if (local / "config.json").is_file() else _repo(config)
        # NOT just the import: qwen_tts imports `sox` lazily inside from_pretrained
        # (seen live 2026-11-15 — wrapping only the import missed it).
        with _quiet_sox():
            self._model = model_factory(source)
        # Capture the CUDA graphs now, so the first game line does not pay for it.
        self._model.warmup(prefill_len=100)
        sr = getattr(self._model, "sample_rate", None)
        if isinstance(sr, int) and sr > 0:
            self.sample_rate = sr
        self._prime()

    def _prime(self) -> None:
        """Speak one throwaway line at startup.

        Graph capture alone is not enough: measured on Windows (RTX 5090), the
        FIRST real generation after start still took 1.6–8 s to its first audio
        (lazy kernel loading in the tokenizer/decoder), every later one 0.3 s.
        Paying that here moves it from the first game line into the start-up.
        Never fatal — a failed priming only means the first line is slow.
        """
        try:
            noise = (np.random.default_rng(0).standard_normal(self.sample_rate) * 0.01).astype(np.float32)
            prompt = self._encode(noise, self.sample_rate, "", xvec=True)
            for _ in self._model.generate_voice_clone_streaming(
                text="Hallo.", language="German", voice_clone_prompt=prompt, ref_text="",
                chunk_size=self._chunk_size, non_streaming_mode=self._non_streaming_mode,
            ):
                pass
        except Exception as exc:  # noqa: BLE001 — see docstring
            ndjson.log_once(i18n.t("engine.warmup_failed", name="Qwen", error=exc), level="warning")

    # ------------------------------------------------------------ prompt
    def _prompt(self, ref: "str | InlineReference", ref_text: str, xvec: bool):
        if isinstance(ref, InlineReference):
            key = f"inline:{ref.voice_id}:{int(xvec)}"
            load = lambda: (ref.audio, ref.sample_rate)  # noqa: E731
        else:
            path = Path(ref)
            mtime = path.stat().st_mtime_ns if path.exists() else 0
            # The transcript is part of the key: the same clip with a different
            # (corrected) text is a different ICL prompt.
            key = f"file:{path}:{mtime}:{int(xvec)}:{hash(ref_text)}"
            load = lambda: _read_mono(path)  # noqa: E731
        return self._prompts.get_or_compute(key, lambda: self._encode(*load(), ref_text, xvec))

    def _encode(self, audio: np.ndarray, sr: int, ref_text: str, xvec: bool):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if xvec:
            # No trailing silence here: the library adds none on this route either.
            return self._model.model.create_voice_clone_prompt(
                ref_audio=(audio, sr), ref_text="", x_vector_only_mode=True
            )
        silence = np.zeros(int(ICL_TRAILING_SILENCE_S * sr), dtype=np.float32)
        return self._model.model.create_voice_clone_prompt(
            ref_audio=(np.concatenate([audio, silence]), sr), ref_text=ref_text
        )

    # ---------------------------------------------------------- protocol
    def infer_stream(
        self,
        ref_file: "str | InlineReference",
        ref_text: str,
        gen_text: str,
        speed: float,
        language: Optional[str] = None,
    ) -> Iterator[np.ndarray]:
        if isinstance(ref_file, InlineReference):
            ref_text = ref_file.text
        ndjson.warn_ignored_speed("Qwen", speed)
        ref_text = (ref_text or "").strip()
        xvec = not ref_text
        if xvec:
            # ICL cannot work without the transcript. Speaker embedding only
            # sounds less like the reference, but it is audible, not a 500.
            # (No longer a user option: removed 2026-11-15, it only ever
            # sounded worse — this fallback is its one remaining use.)
            ndjson.log_once(
                i18n.t("qwen.no_ref_text"),
                level="warning",
            )
        prompt = self._prompt(ref_file, ref_text, xvec)
        for chunk, sr, _info in self._model.generate_voice_clone_streaming(
            text=gen_text,
            language=language_name(language, self._lang),
            voice_clone_prompt=prompt,
            ref_text="" if xvec else ref_text,
            chunk_size=self._chunk_size,
            temperature=self._temperature,
            non_streaming_mode=self._non_streaming_mode,
        ):
            yield _to_contract(chunk, sr, self.sample_rate)

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

    # ------------------------------------------------- speaker embedding
    @property
    def speaker_encoder(self) -> str:
        """Name of the vector space ``speaker_embedding`` answers in. The 0.6B and
        1.7B models carry different encoders, so the backend id is part of it —
        vectors of different names must never be compared."""
        return f"{self._backend}-xvector"

    def speaker_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """x-vector of one clip, from memory (``POST /speaker-embedding``).

        The model's own speaker encoder — the one ICL conditioning uses — so no
        extra download. It wants 24 kHz mono float32."""
        core = self._model.model.model
        target = int(getattr(core, "speaker_encoder_sample_rate", DEFAULT_SAMPLE_RATE))
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sr != target:
            audio = _to_contract(audio, sr, target)
        vec = core.extract_speaker_embedding(audio=audio, sr=target)
        return _as_vector(vec)

    def transcribe(self, audio_path: Path) -> str:
        """ASR fallback for a sample without sidecar — ICL needs the text."""
        from .asr import whisper_transcribe

        audio, sr = _read_mono(Path(audio_path))
        return whisper_transcribe(audio, sr, self._lang)

    def self_test(self) -> bool:
        return selftest.run(self)


def _as_vector(vec) -> np.ndarray:
    """Tensor (any device/dtype) or array → flat float32 numpy vector."""
    if hasattr(vec, "detach"):
        vec = vec.detach().float().cpu().numpy()
    return np.asarray(vec, dtype=np.float32).reshape(-1)


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf  # lazy

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, int(sr)


def _to_contract(chunk, sr: int, target_sr: int) -> np.ndarray:
    """Flat float32 at the contract rate. A different rate passed through would
    not fail loudly — it would just play at the wrong speed."""
    arr = chunk.detach().to("cpu").numpy() if hasattr(chunk, "detach") else np.asarray(chunk)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if sr and sr != target_sr:
        import torch  # lazy
        import torchaudio  # lazy

        arr = torchaudio.functional.resample(torch.from_numpy(arr), sr, target_sr).numpy()
    return arr


def download_model(config: Config) -> None:
    """Fetch the active Qwen engine's repo into ``models/qwen/<name>``. Idempotent."""
    config.models_path.mkdir(parents=True, exist_ok=True)
    hfcache.use_models_dir(config)
    repo = _repo(config)
    target = _model_dir(config)
    mp = progress.ModelProgress()
    mp.stage(f"{repo.split('/')[-1]}: ")
    ndjson.log(i18n.t("model.loading", name=repo))
    with mp.patch():
        from huggingface_hub import snapshot_download  # lazy: imported inside the patch

        snapshot_download(repo_id=repo, local_dir=str(target))
    ndjson.log(i18n.t("model.ready", name=repo))


if __name__ == "__main__":
    # Entry point used by the bootstrap (run inside .venv-qwen) for the active
    # Qwen engine: downloads its weights into models/qwen/.
    import sys
    import traceback

    try:
        download_model(load_config())
    except Exception as exc:  # noqa: BLE001 — surface as a fatal NDJSON error
        traceback.print_exc(file=sys.stderr)
        ndjson.error(i18n.t("model.download_failed", name="Qwen", error=exc), fatal=True)
        raise SystemExit(1)
