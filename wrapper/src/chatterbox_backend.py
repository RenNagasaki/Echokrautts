"""Chatterbox Multilingual backend — the wrapper's third TTS engine.

Resemble AI's Chatterbox, multilingual variant (``chatterbox.mtl_tts``): one
model covering 23 languages that clones a voice from a reference sample
**without a transcript** — same deal as XTTS, so the ASR ref-text path is a
no-op here too.

Design parity with the other two workers: this exposes the same
:class:`~src.engine.WorkerProtocol` (``infer`` returns one float32 mono clip at
24000 Hz per sentence chunk), so the engine's worker pool, backpressure and
cancel logic are reused unchanged. The released package has **no streaming
API**, so ``supports_streaming`` is False and the engine takes its one-shot path
(like F5). Conditioning is computed once per reference sample and cached, since
the engine calls ``infer`` repeatedly with the same sample.

Two properties of the upstream package are worth knowing:

* **Every output is watermarked.** ``generate()`` runs the clip through
  ``perth.PerthImplicitWatermarker`` — an inaudible provenance watermark by
  Resemble. It is part of the library's normal path and is left in place.
* **No speed control.** Unlike F5 and XTTS, ``generate()`` has no ``speed``
  parameter, so a request's ``speed`` is ignored here (logged once).

Dependency note (see ``bootstrap.step_deps``): ``chatterbox-tts`` declares
``torch==2.6.0``, ``transformers==5.2.0`` and ``gradio``, none of which this
wrapper can honor — the pinned torch is 2.7.0 and coqui-tts (XTTS) needs
transformers <5. It is therefore installed with ``--no-deps`` plus a curated
list of the packages it actually imports. Its transformers usage is limited to
long-stable APIs (``LlamaModel``/``LlamaConfig``/``GPT2*``/``GenerationMixin``),
identical in the release that pinned 4.46 and the one that pins 5.2.

All torch / chatterbox imports are lazy (inside the worker / resolver) so this
module — and the unit test suite — import without those heavy deps installed.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from . import ndjson, progress
from .config import Config, load_config

# HuggingFace repo holding the multilingual weights, and the exact file subset
# ``ChatterboxMultilingualTTS.from_pretrained`` fetches. Mirrored here so the
# install step can pre-download without loading the model onto a device. Drift
# is self-healing: the library re-runs its own ``snapshot_download`` at load
# time, which is a no-op for files already in the cache and fetches any we miss.
CHATTERBOX_REPO = "ResembleAI/chatterbox"
CHATTERBOX_FILES = [
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
]

DEFAULT_SAMPLE_RATE = 24000  # S3GEN_SR, same as F5 and XTTS

# Language codes the multilingual model accepts (chatterbox.mtl_tts.
# SUPPORTED_LANGUAGES). Like XTTS this is per-request — one loaded model serves
# every language, so switching costs nothing. Note Chinese is ``zh`` here, not
# XTTS's ``zh-cn``.
CHATTERBOX_LANGUAGES = frozenset(
    {
        "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it",
        "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
    }
)

# What a user-supplied Chatterbox model directory must contain (the files
# ``from_local`` opens by name). Deliberately the full set: a partial folder
# would be detected as "custom model" and then fail deep inside the loader.
CUSTOM_MODEL_FILES = (
    "ve.pt",
    "s3gen.pt",
    "t3_mtl23ls_v2.safetensors",
    "grapheme_mtl_merged_expanded_v1.json",
)


def _use_models_cache(config: Config) -> None:
    """Point the HF cache at ``models/`` for this process.

    The library calls ``snapshot_download`` with no ``cache_dir``, so unlike
    :mod:`src.models` (which passes one explicitly) the environment is the only
    lever. It is set, not defaulted: ``models_dir`` IS the user's setting for
    where weights live, and it must win over a pre-existing ``HF_HUB_CACHE``.

    Found live: on a machine with a global ``HF_HUB_CACHE`` (``G:\\cache\\…``),
    ``setdefault`` left it in place — so :func:`download_model` filled
    ``models/`` while the worker then downloaded the same 2 GB again into the
    global cache. In a container the same split means the weights land outside
    the ``/data/models`` volume and are re-fetched on every start.
    """
    config.models_path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(config.models_path)
    os.environ["HF_HUB_CACHE"] = str(config.models_path)


@contextlib.contextmanager
def _stdout_to_stderr():
    """Keep third-party ``print`` out of the NDJSON stream.

    stdout is the wrapper's protocol channel (one JSON object per line, parsed
    by the host); anything else on it is corruption. Loading the model prints
    at least one line — ``perth`` announces "loaded PerthNet (Implicit) at step
    250,000" — so model construction runs with stdout pointed at stderr, where
    free-form output belongs.

    Deliberately only around *loading*, not inference: this swaps the global
    ``sys.stdout``, so wrapping a multi-second ``generate()`` would also swallow
    NDJSON events the event loop emits meanwhile. Inference itself is quiet —
    its tqdm bars already go to stderr (verified against chatterbox 0.1.7).
    """
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _resolve_custom_model_dir(config: Config) -> Optional[str]:
    """A user-supplied Chatterbox model dropped into ``models/echokraut_custom/``.

    Returns the directory when it holds the complete file set
    (:data:`CUSTOM_MODEL_FILES`), else ``None``. The three backends can share the
    one custom-model folder because their formats are disjoint: F5 is a bare
    checkpoint, XTTS needs ``config.json`` + ``model.pth``, Chatterbox needs this
    specific quartet.
    """
    d = config.custom_model_path
    if all((d / name).is_file() for name in CUSTOM_MODEL_FILES):
        # log_once: the resolver runs once per worker (pool of n) — a plain log
        # would print the same line n times.
        ndjson.log_once(f"Nutze eigenes Chatterbox-Modell: {d}")
        return str(d)
    return None


# Name of the closure the upstream ``AlignmentStreamAnalyzer`` registers on the
# attention modules (``_add_attention_spy`` in chatterbox). Matched by name so
# only ITS hooks are dropped — a hook installed by anything else survives.
ANALYZER_HOOK_NAME = "attention_forward_hook"


def _drop_stale_analyzer_hooks(model) -> int:
    """Remove attention hooks left behind by PREVIOUS requests. Returns the count.

    Upstream leaks them: ``T3.inference`` sets ``self.compiled = False`` right
    before ``if not self.compiled:``, so a fresh ``AlignmentStreamAnalyzer`` is
    built on **every** request, and its constructor calls
    ``register_forward_hook`` on the persistent transformer without ever
    removing the previous one. Each hook copies the attention matrix to the host
    (``output[1].cpu()``) on **every decode step**, so the cost grows with the
    number of requests the process has served — invisible in a short test, fatal
    in a server that runs for hours.

    Measured on an RTX 5090: 3 hooks per request; after ~50 requests' worth
    (153 hooks) the same sentence went from rtf 1.17 to 1.29, and back to 1.13
    once they were dropped.

    Defensive by design: this reaches into library internals, so every step is
    guarded and a changed layout simply means "nothing to clean" rather than a
    failed request.
    """
    removed = 0
    layers = getattr(getattr(getattr(model, "t3", None), "tfmr", None), "layers", None)
    if layers is None:
        return 0
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        hooks = getattr(attn, "_forward_hooks", None)
        if not hooks:
            continue
        for handle_id, fn in list(hooks.items()):
            if getattr(fn, "__name__", "") == ANALYZER_HOOK_NAME:
                del hooks[handle_id]
                removed += 1
    return removed


class ChatterboxWorker:
    """Wraps one ``ChatterboxMultilingualTTS`` instance bound to one device."""

    supports_streaming = False  # the released package has no streaming API

    def __init__(self, config: Config, device: str):
        import torch  # noqa: F401  lazy: heavy import (ensures torch is present)

        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self.device = device
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self._lang = config.language
        self._exaggeration = float(config.chatterbox_exaggeration)
        self._cfg_weight = float(config.chatterbox_cfg_weight)
        self._temperature = float(config.chatterbox_temperature)
        self._conds_cache: dict = {}

        _use_models_cache(config)
        # Chatterbox has no DirectML/XPU path; those fall back to CPU, matching
        # XTTS (the engine self-tests fragile devices and rebuilds on CPU).
        resolved_device = "cpu" if device in ("dml", "xpu") else device
        custom = _resolve_custom_model_dir(config)
        with _stdout_to_stderr():
            if custom is not None:
                self._model = ChatterboxMultilingualTTS.from_local(custom, resolved_device)
            else:
                self._model = ChatterboxMultilingualTTS.from_pretrained(
                    device=resolved_device
                )

        sr = getattr(self._model, "sr", None)
        if isinstance(sr, int) and sr > 0:
            self.sample_rate = sr

    def _conditioning(self, ref_file: str) -> None:
        """Install the conditioning for ``ref_file``, computing it once.

        ``generate(audio_prompt_path=…)`` would redo the whole reference
        embedding on every call; instead we prepare it once and re-attach the
        cached :class:`Conditionals` to the model, which is what the parameter
        does internally anyway. Mirrors the latent cache in the XTTS worker.
        """
        cached = self._conds_cache.get(ref_file)
        if cached is None:
            self._model.prepare_conditionals(ref_file, exaggeration=self._exaggeration)
            cached = self._model.conds
            self._conds_cache[ref_file] = cached
        self._model.conds = cached

    def _language(self, language: Optional[str]) -> str:
        """Per-request language, falling back to the startup one (no reload)."""
        return (language or self._lang or "en").lower()

    def infer(
        self,
        ref_file: str,
        ref_text: str,
        gen_text: str,
        nfe_step: int,
        speed: float,
        language: Optional[str] = None,
    ) -> np.ndarray:
        # ref_text / nfe_step are F5 concepts and unused here (Chatterbox clones
        # from the audio and has no NFE steps). ``speed`` has no equivalent in
        # generate() either — say so once rather than silently ignoring it.
        if speed and abs(float(speed) - 1.0) > 1e-6:
            ndjson.log_once(
                "Chatterbox kennt keinen speed-Parameter — speed wird ignoriert",
                level="warning",
            )
        # Drop the hooks the previous request leaked before this one adds its own
        # (see _drop_stale_analyzer_hooks) — otherwise this worker gets slower for
        # every request it has ever served.
        _drop_stale_analyzer_hooks(self._model)
        self._conditioning(ref_file)
        wav = self._model.generate(
            gen_text,
            language_id=self._language(language),
            exaggeration=self._exaggeration,
            cfg_weight=self._cfg_weight,
            temperature=self._temperature,
        )
        # generate() returns a (1, N) torch tensor; the engine wants a flat
        # float32 array.
        arr = wav.detach().to("cpu").numpy() if hasattr(wav, "detach") else np.asarray(wav)
        return np.asarray(arr, dtype=np.float32).reshape(-1)

    def transcribe(self, audio_path: Path) -> str:
        # Chatterbox clones from the reference audio directly; it never needs the
        # transcript, so ref-text resolution is a no-op for this backend.
        return ""

    def self_test(self) -> bool:
        """Tiny inference to confirm the backend works (engine runs this only
        for fragile devices, which Chatterbox anyway maps to CPU)."""
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
    """Pre-download the Chatterbox multilingual weights. Idempotent.

    Downloads the file subset rather than loading the model, so the install step
    needs no GPU and no VRAM. Per-file progress comes from the tqdm hook in
    :mod:`src.progress` — which only works because ``huggingface_hub`` is
    imported INSIDE the patch context (it binds its tqdm class on first import).
    """
    _use_models_cache(config)
    mp = progress.ModelProgress()
    mp.stage("Chatterbox: ")
    ndjson.log("Lade Chatterbox-Multilingual-Modell …")
    with mp.patch():
        from huggingface_hub import snapshot_download  # lazy (see docstring)

        snapshot_download(
            repo_id=CHATTERBOX_REPO,
            repo_type="model",
            allow_patterns=CHATTERBOX_FILES,
            cache_dir=str(config.models_path),
        )
    ndjson.log("Chatterbox-Modell bereit")


if __name__ == "__main__":
    # Entry point used by the bootstrap (run inside the venv) when the active
    # backend is Chatterbox: downloads the weights into models/.
    import sys
    import traceback

    try:
        download_model(load_config())
    except Exception as exc:  # noqa: BLE001 — surface as a fatal NDJSON error
        traceback.print_exc(file=sys.stderr)
        ndjson.error(f"chatterbox model download failed: {exc}", fatal=True)
        raise SystemExit(1)
