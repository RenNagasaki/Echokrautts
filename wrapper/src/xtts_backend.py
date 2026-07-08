"""XTTS-v2 backend — the wrapper's second TTS engine (SPEC addendum).

An alternative to the F5-TTS worker, selected at startup via
``config.tts_backend == "xtts"``. XTTS-v2 (Coqui) is natively multilingual
(en/de/fr/ja among 17 languages) and clones a voice from a reference sample
**without needing its transcript** — so the ASR ref-text path is a no-op here.

Design parity with the F5 worker: this exposes the same :class:`WorkerProtocol`
(``infer`` returns one float32 mono clip @ 24000 Hz per sentence chunk), so the
:class:`~src.engine.Engine` worker pool, backpressure and streaming logic are
reused unchanged. Conditioning latents are computed once per reference sample
and cached, since the engine calls ``infer`` repeatedly with the same sample.

License note: XTTS-v2 weights are under the Coqui Public Model License (CPML,
non-commercial). We use the maintained community fork ``coqui-tts`` (idiap) —
the original ``TTS`` package is unmaintained. Accepting the license
non-interactively is done via ``COQUI_TOS_AGREED=1``.

All torch / TTS imports are lazy (inside the worker / resolver) so this module —
and the unit test suite — import without those heavy deps installed.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import ndjson, progress
from .config import Config, load_config

# Coqui model id for XTTS-v2 (multilingual, multi-dataset).
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_SAMPLE_RATE = 24000  # XTTS-v2 output rate

# ISO codes XTTS-v2 accepts at inference. Unlike F5 (one finetune per language,
# fixed at startup), XTTS is natively multilingual — so the per-request
# ``language`` selects the target language with NO model reload. Superset of the
# wrapper's en/de/fr/ja language map; the server validates requests against this.
XTTS_LANGUAGES = frozenset(
    {
        "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
        "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi",
    }
)


def _should_use_fp16(config: Config, device: str) -> bool:
    """Whether to load the XTTS model in half precision.

    Only on CUDA (NVIDIA or ROCm, both reported as ``"cuda"``) and only when
    ``config.xtts_fp16`` is set. fp16 on CPU is unsupported/slower, and dml/xpu
    are mapped to CPU by the worker — so pass the *resolved* device here.
    """
    return bool(config.xtts_fp16) and device == "cuda"


def _resolve_custom_model_dir(config: Config) -> str | None:
    """A user-supplied XTTS model dropped into ``models/echokraut_custom/``.

    Returns the directory when it looks like a full XTTS-v2 model (a
    ``config.json`` plus a ``model.pth``/``model.safetensors`` weight file),
    else ``None``. F5 finetunes ship a bare checkpoint with no ``config.json``,
    so the two backends can share the same custom-model folder safely.
    """
    d = config.custom_model_path
    if (d / "config.json").is_file() and (
        (d / "model.pth").is_file() or (d / "model.safetensors").is_file()
    ):
        ndjson.log(f"Nutze eigenes XTTS-Modell: {d}")
        return str(d)
    return None


def _resolve_model_dir(config: Config) -> str:
    """Download (or reuse the cache for) the XTTS-v2 model directory.

    A user-supplied custom model (``models/echokraut_custom/``) wins over the
    base download. Otherwise, idempotent: Coqui's :class:`ModelManager` skips the
    download when the files already exist under ``models/``. Returns the local
    model directory path. Used by BOTH the bootstrap (install-time preload) and
    the worker (load) — single source of truth, mirroring ``models.resolve_model``
    for F5.
    """
    custom = _resolve_custom_model_dir(config)
    if custom is not None:
        return custom

    # Accept the CPML non-interactively so the download never blocks on a prompt.
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    from TTS.utils.manage import ModelManager  # lazy: heavy import

    manager = ModelManager(output_prefix=str(config.models_path))
    model_path, _config_path, _item = manager.download_model(XTTS_MODEL)
    return model_path


class XTTSWorker:
    """Wraps a single Coqui ``Xtts`` model bound to one device and language."""

    supports_streaming = True  # XTTS exposes token streaming via inference_stream

    def __init__(self, config: Config, device: str):
        import torch  # noqa: F401  lazy: heavy import (ensures torch is present)
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        self.device = device
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self._stream_chunk_size = config.stream_chunk_size
        # One language per process (like the F5 backend); XTTS takes ISO codes
        # (en/de/fr/ja) that match the wrapper's language keys directly.
        self._lang = config.language
        self._cond_cache: dict[str, tuple] = {}

        model_dir = _resolve_model_dir(config)
        xtts_config = XttsConfig()
        xtts_config.load_json(str(Path(model_dir) / "config.json"))
        model = Xtts.init_from_config(xtts_config)
        model.load_checkpoint(
            xtts_config, checkpoint_dir=model_dir, eval=True, use_deepspeed=False
        )
        # XTTS has no DirectML/XPU path; those fall back to CPU (engine self-test).
        resolved_device = "cpu" if device in ("dml", "xpu") else device
        model.to(resolved_device)
        # Optional fp16 speedup — CUDA only (see _should_use_fp16). The inference
        # outputs are cast back to float32 for PCM, so the HTTP contract is intact.
        self._fp16 = _should_use_fp16(config, resolved_device)
        if self._fp16:
            model.half()
            ndjson.log(f"XTTS fp16 enabled on {resolved_device}")
        self._model = model

        sr = getattr(getattr(xtts_config, "audio", None), "output_sample_rate", None)
        if isinstance(sr, int) and sr > 0:
            self.sample_rate = sr

    def _conditioning(self, ref_file: str) -> tuple:
        """(gpt_cond_latent, speaker_embedding) for a sample, cached by path."""
        cached = self._cond_cache.get(ref_file)
        if cached is not None:
            return cached
        latents = self._model.get_conditioning_latents(audio_path=[ref_file])
        self._cond_cache[ref_file] = latents
        return latents

    def infer(
        self,
        ref_file: str,
        ref_text: str,
        gen_text: str,
        nfe_step: int,
        speed: float,
        language: str | None = None,
    ) -> np.ndarray:
        # ref_text / nfe_step are F5 concepts; XTTS ignores them (clones from
        # the audio directly and has no NFE steps). ``language`` selects the
        # target language per request (XTTS is multilingual → no reload); falls
        # back to the startup language when omitted.
        gpt_cond_latent, speaker_embedding = self._conditioning(ref_file)
        out = self._model.inference(
            gen_text,
            language or self._lang,
            gpt_cond_latent,
            speaker_embedding,
            speed=speed,
            enable_text_splitting=False,
        )
        return np.asarray(out["wav"], dtype=np.float32)

    def infer_stream(
        self,
        ref_file: str,
        ref_text: str,
        gen_text: str,
        speed: float,
        language: str | None = None,
    ):
        """Yield float32 audio chunks as XTTS generates them (token streaming).

        Lower first-audio latency than :meth:`infer`: audio starts flowing before
        the whole chunk is synthesized. The engine pumps this sync generator one
        item at a time in a thread. ``ref_text`` is ignored (XTTS clones from the
        audio alone); ``language`` selects the target language per request (no
        reload), defaulting to the startup language when omitted."""
        gpt_cond_latent, speaker_embedding = self._conditioning(ref_file)
        for wav_chunk in self._model.inference_stream(
            gen_text,
            language or self._lang,
            gpt_cond_latent,
            speaker_embedding,
            speed=speed,
            enable_text_splitting=False,
            stream_chunk_size=self._stream_chunk_size,
        ):
            # XTTS yields torch tensors on the compute device; move to a flat
            # CPU float32 array for float_to_pcm16.
            arr = wav_chunk.detach().to("cpu").numpy() if hasattr(wav_chunk, "detach") else np.asarray(wav_chunk)
            yield np.asarray(arr, dtype=np.float32).reshape(-1)

    def transcribe(self, audio_path: Path) -> str:
        # XTTS clones from the reference audio directly; it never needs the
        # transcript, so ref-text resolution is a no-op for this backend.
        return ""

    def self_test(self) -> bool:
        """Tiny inference to confirm the backend works (engine runs this only
        for fragile dml/xpu devices, which XTTS anyway maps to CPU)."""
        try:
            import tempfile

            import soundfile as sf

            with tempfile.TemporaryDirectory() as tmp:
                ref = Path(tmp) / "selftest.wav"
                sf.write(ref, np.zeros(self.sample_rate, dtype=np.float32), self.sample_rate)
                self.infer(str(ref), "", "test", nfe_step=8, speed=1.0)
            return True
        except Exception:  # noqa: BLE001 — any failure means "fall back to CPU"
            return False


def download_model(config: Config) -> None:
    """Pre-download XTTS-v2 at install time. Idempotent (see resolver).

    Coqui's ``ModelManager`` fetches several files (model, config, vocab,
    speakers) — each is reported as a per-file ``progress`` bar via the tqdm
    hook in :mod:`src.progress`, on top of the coarse start/done ``log`` lines.
    """
    config.models_path.mkdir(parents=True, exist_ok=True)
    mp = progress.ModelProgress()
    mp.stage("XTTS-v2: ")
    ndjson.log("Lade XTTS-v2-Modell …")
    with mp.patch():
        _resolve_model_dir(config)
    ndjson.log("XTTS-v2-Modell bereit")


if __name__ == "__main__":
    # Entry point used by the bootstrap (run inside the venv) when the active
    # backend is XTTS: downloads the XTTS-v2 weights into models/.
    import sys
    import traceback

    try:
        download_model(load_config())
    except Exception as exc:  # noqa: BLE001 — surface as a fatal NDJSON error
        traceback.print_exc(file=sys.stderr)
        ndjson.error(f"xtts model download failed: {exc}", fatal=True)
        raise SystemExit(1)
