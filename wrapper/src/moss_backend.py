"""MOSS-TTS-Nano backend — the wrapper's third TTS engine.

Selected at startup via ``config.tts_backend == "moss"``. OpenMOSS's 0.1B model
is the first engine here that combines everything the others only have in parts,
and every claim below was measured on this machine before the code was written:

* **It does not need the GPU.** CPU and an RTX 5090 measured the same real-time
  factor (1.19-1.24 vs 1.16-1.40). For a plugin that speaks while a game renders,
  that is the whole point: set ``moss_device: "cpu"`` and the graphics card stays
  with the game.
* **Real streaming, and it is fast to first sound.** ``synthesize_stream``
  yields 41-54 audio chunks per sentence; first audio arrived after 0.12 s on
  GPU and 0.45 s on CPU, against ~2.4 s for XTTS and 6.2 s for the Qwen backend
  that was tried and dropped. Time-to-first-sound is what makes speech feel
  immediate, far more than the total.
* **Apache-2.0 for code and weights**, so synthesized audio carries no
  non-commercial restriction — unlike F5 (CC-BY-NC) and XTTS (CPML).
* **All four FFXIV client languages** (en/ja/de/fr) among 19, in one model.
* **312 MB** of weights against F5's ~5 GB and XTTS's ~2 GB.

The honest limit: **rtf ~1.2 is slower than real time.** Streaming hides it for
short lines (playback starts long before generation ends), but a long sentence
will be caught up with. It is the price for a model this small.

Cloning needs **no transcript** — reference audio alone, like XTTS.

All torch / moss imports are lazy (inside the worker) so this module — and the
unit suite — import without those heavy deps installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from . import ndjson, progress, selftest
from .config import Config, load_config

# The wrapper's HTTP contract: s16 mono at this rate. MOSS emits 48 kHz stereo,
# so every chunk is downmixed and resampled on the way out (see _to_contract).
DEFAULT_SAMPLE_RATE = 24000

# ISO codes for the 19 languages the model card lists. MOSS takes NO language
# argument — it infers the language from the text itself — so this set exists
# purely so the server can reject a request for something the model was never
# trained on, instead of returning confident nonsense.
MOSS_LANGUAGES = frozenset(
    {
        "zh", "en", "de", "es", "fr", "ja", "it", "hu", "ko", "ru",
        "fa", "ar", "pl", "pt", "cs", "da", "sv", "el", "tr",
    }
)


def _use_models_cache(config: Config) -> None:
    """Point the HuggingFace cache at the wrapper's own ``models/`` directory.

    SET, not ``setdefault``: a machine-wide ``HF_HUB_CACHE`` would otherwise send
    the worker off to fetch the same weights a second time, and in a container
    that means they land outside the ``/data/models`` volume on every start.
    """
    import os

    cache = str(config.models_path)
    os.environ["HF_HOME"] = cache
    os.environ["HF_HUB_CACHE"] = cache
    # MOSS ships its model code with the weights (`trust_remote_code`), and
    # transformers writes that code into a SEPARATE cache. Without this it lands
    # in the user's global one — outside the models volume in a container, and
    # invisible when debugging why a model loads the wrong code.
    os.environ["HF_MODULES_CACHE"] = str(Path(cache) / "modules")
    if config.hf_endpoint:
        os.environ["HF_ENDPOINT"] = config.hf_endpoint


def _model_dirs(config: Config) -> tuple[Path, Path]:
    """(checkpoint dir, audio tokenizer dir) for the two repos MOSS needs.

    Flat directories under ``models/``, not the HuggingFace cache: that cache
    links ``snapshots/`` to ``blobs/`` with symlinks, and creating one on Windows
    needs a privilege a normal account does not have — a download into it died
    with ``WinError 1314`` when this was first tried for another engine.
    """
    # UNDERSCORES, not hyphens, and that is load-bearing. MOSS ships its model
    # code with the weights (`trust_remote_code`), and transformers turns the
    # directory name into a Python module name — a hyphen becomes the literal
    # text "_hyphen_", and the module the generated code then imports by its own
    # name cannot be resolved: "No module named
    # transformers_modules.moss_hyphen_tts_hyphen_nano". Hit live on the first
    # real load.
    base = config.models_path
    return base / "moss_tts_nano", base / "moss_audio_tokenizer_nano"


def _resolve_model_dirs(config: Config) -> tuple[str, str]:
    """Local paths the worker loads from; a custom model wins over the download.

    A user-supplied model goes in ``models/echokraut_custom/`` and must bring its
    own audio tokenizer alongside the checkpoint, since MOSS needs both halves —
    a checkpoint without a matching tokenizer produces noise rather than an error.
    """
    custom = config.custom_model_path
    if (custom / "config.json").is_file() and (custom / "audio_tokenizer").is_dir():
        ndjson.log_once(f"Nutze eigenes MOSS-Modell: {custom}")
        return str(custom), str(custom / "audio_tokenizer")
    checkpoint, tokenizer = _model_dirs(config)
    return str(checkpoint), str(tokenizer)


def _resolve_device(config: Config, device: str) -> str:
    """Which device MOSS should load on.

    ``config.moss_device`` overrides the engine's choice, and "cpu" is a
    legitimate answer rather than a fallback: measured, CPU matches the GPU here,
    so running on the CPU costs nothing and leaves the graphics card to whatever
    else the user is doing. dml/xpu have no path and map to CPU like the other
    backends.
    """
    forced = (config.moss_device or "auto").strip().lower()
    if forced in ("cpu", "cuda"):
        return forced
    return "cpu" if device in ("dml", "xpu") else device


class MossWorker:
    """Wraps one MOSS-TTS-Nano service bound to a device."""

    supports_streaming = True

    def __init__(self, config: Config, device: str):
        # BEFORE the import, not after: transformers resolves its cache paths at
        # import time, so setting them afterwards is too late and the model code
        # lands in whatever machine-wide cache the user happens to have (found
        # live: a global HF_HOME sent it to G:\cache instead of models/).
        _use_models_cache(config)

        from moss_tts_nano_runtime import NanoTTSService  # lazy: heavy import

        self.sample_rate = DEFAULT_SAMPLE_RATE
        self._max_new_frames = int(config.moss_max_new_frames)

        checkpoint, tokenizer = _resolve_model_dirs(config)
        resolved = _resolve_device(config, device)
        # The RESOLVED device, not the requested one. This worker is the only
        # one that overrides the engine's choice, and `/health` and the `ready`
        # event read this attribute — reporting "cuda" while the model sits on
        # the CPU is a lie told at exactly the place a user looks to check.
        self.device = resolved
        # Generated clips are written to disk by the runtime on every request —
        # there is no flag to turn that off. Keeping them under models/ would
        # grow without bound on a server that speaks thousands of game lines, so
        # they go to a directory this worker owns and empties (see _cleanup).
        self._scratch = config.models_path / "moss-scratch"
        self._scratch.mkdir(parents=True, exist_ok=True)
        self._service = NanoTTSService(
            checkpoint_path=checkpoint,
            audio_tokenizer_path=tokenizer,
            device=resolved,
            output_dir=str(self._scratch),
        )
        self._service.preload(load_model=True)
        ndjson.log_once(f"MOSS-TTS-Nano geladen auf {resolved}")

    # ------------------------------------------------------------------ audio

    def _to_contract(self, waveform, source_rate: int) -> np.ndarray:
        """MOSS's 48 kHz stereo chunk -> the wrapper's 24 kHz mono float32.

        Done per chunk rather than once at the end, because the streaming path
        has no "end" to do it at. torchaudio does the resampling; it is already a
        pinned dependency, so this pulls nothing new in.
        """
        import torch
        import torchaudio

        tensor = waveform if hasattr(waveform, "dim") else torch.as_tensor(waveform)
        tensor = tensor.detach().to("cpu", dtype=torch.float32)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] > 1:  # stereo (or more) -> mono
            tensor = tensor.mean(dim=0, keepdim=True)
        if source_rate and source_rate != self.sample_rate:
            tensor = torchaudio.functional.resample(tensor, source_rate, self.sample_rate)
        return tensor.squeeze(0).numpy().astype(np.float32, copy=False)

    def _cleanup(self, path) -> None:
        """Drop the file the runtime insisted on writing. Never fatal.

        A failure here means one leftover file, which is worth a log line and
        nothing more — certainly not a failed request that produced good audio.
        """
        try:
            if path:
                Path(str(path)).unlink(missing_ok=True)
        except OSError as exc:  # noqa: BLE001 — leftover file, not a failure
            ndjson.log_once(f"MOSS: Zwischendatei nicht entfernt ({exc})", level="warning")

    def _events(self, ref_file: str, gen_text: str):
        """The runtime's streaming generator, with our arguments applied."""
        return self._service.synthesize_stream(
            text=gen_text,
            mode="voice_clone",
            prompt_audio_path=ref_file,
            max_new_frames=self._max_new_frames,
        )

    # ---------------------------------------------------------------- protocol

    def infer_stream(
        self,
        ref_file: str,
        ref_text: str,
        gen_text: str,
        speed: float,
        language: str | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield float32 chunks as MOSS produces them.

        ``ref_text`` is unused (MOSS clones from the audio alone) and so is
        ``language``: the model infers the language from the text it is given,
        which is why there is no language argument to pass on. ``speed`` has no
        equivalent either and is reported once rather than silently dropped.
        """
        if speed != 1.0:
            ndjson.log_once(
                "MOSS-TTS-Nano kennt keinen speed-Parameter — der Wert wird ignoriert",
                level="warning",
            )
        final_path = None
        try:
            for event in self._events(ref_file, gen_text):
                if event.get("type") == "audio":
                    chunk = self._to_contract(event["waveform"], int(event.get("sample_rate", 0)))
                    if chunk.size:
                        yield chunk
                elif event.get("type") == "result":
                    final_path = event.get("audio_path")
        finally:
            self._cleanup(final_path)

    def infer(
        self,
        ref_file: str,
        ref_text: str,
        gen_text: str,
        nfe_step: int,
        speed: float,
        language: str | None = None,
    ) -> np.ndarray:
        """One clip, for the engine's non-streaming path.

        Deliberately drains :meth:`infer_stream` instead of calling the
        runtime's one-shot method: one code path means the streamed audio and
        the buffered audio cannot drift apart, and the conversion to the
        wrapper's format lives in exactly one place.
        """
        chunks = list(
            self.infer_stream(ref_file, ref_text, gen_text, speed, language)
        )
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)

    def transcribe(self, audio_path: Path) -> str:
        # MOSS clones from the reference audio directly; the transcript is never
        # needed, so ref-text resolution is a no-op for this backend.
        return ""

    def self_test(self) -> bool:
        """Tiny inference to confirm the backend works (engine runs this only
        for fragile dml/xpu devices, which this worker maps to CPU anyway)."""
        return selftest.run(self)


def download_model(config: Config) -> None:
    """Pre-download both MOSS repos at install time. Idempotent.

    Two repos, because the model and its audio tokenizer ship separately and
    both are required. ``local_dir`` keeps them out of the symlinked HF cache
    (see :func:`_model_dirs`). Per-file progress is routed to NDJSON by the tqdm
    hook in :mod:`src.progress`.
    """
    config.models_path.mkdir(parents=True, exist_ok=True)
    _use_models_cache(config)
    checkpoint_dir, tokenizer_dir = _model_dirs(config)
    mp = progress.ModelProgress()
    mp.stage("MOSS-TTS-Nano: ")
    ndjson.log("Lade MOSS-TTS-Nano-Modell …")
    with mp.patch():
        from huggingface_hub import snapshot_download  # lazy: imported inside the patch

        for repo, target in (
            (config.moss_model, checkpoint_dir),
            (config.moss_audio_tokenizer, tokenizer_dir),
        ):
            snapshot_download(
                repo_id=repo,
                local_dir=str(target),
                endpoint=config.hf_endpoint or None,
            )
    ndjson.log("MOSS-TTS-Nano-Modell bereit")


if __name__ == "__main__":
    # Entry point used by the bootstrap (run inside the venv) when the active
    # backend is MOSS: downloads the weights into models/.
    import sys
    import traceback

    try:
        download_model(load_config())
    except Exception as exc:  # noqa: BLE001 — surface as a fatal NDJSON error
        traceback.print_exc(file=sys.stderr)
        ndjson.error(f"moss model download failed: {exc}", fatal=True)
        raise SystemExit(1)
