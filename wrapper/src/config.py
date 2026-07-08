"""Configuration loading (SPEC §10).

Precedence: ``config.json`` < environment variables < CLI arguments.
ENV vars are upper-cased field names prefixed with ``F5W_`` (e.g. ``F5W_PORT``).
CLI flags are kebab-cased field names (e.g. ``--max-workers``).

Paths (``samples_dir``, ``models_dir``) are resolved relative to the wrapper
root (the directory that contains ``config.json``) unless given as absolute.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

# Wrapper root = parent of the ``src`` package directory.
WRAPPER_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = WRAPPER_ROOT / "config.json"

ENV_PREFIX = "F5W_"

# Sub-directory of ``models/`` where a user-supplied custom model is dropped (by
# the host's "install custom data" flow). When present it overrides the
# configured model for the ACTIVE engine at load time — F5 finetune (a bare
# checkpoint + optional vocab.txt/arch.txt) for the f5 backend, a full XTTS-v2
# model directory (config.json + model.pth/…) for the xtts backend. See
# ``models.resolve_model`` / ``xtts_backend._resolve_model_dir``.
CUSTOM_MODEL_DIRNAME = "echokraut_custom"


@dataclass
class Config:
    """Resolved runtime configuration. All SPEC §10 keys plus a few extras."""

    # Bind on all interfaces by default so the host/plugin reaches it without any
    # config edit. Set an api_key (or narrow host to 127.0.0.1) to lock it down.
    host: str = "0.0.0.0"
    port: int = 8765
    parent_pid: Optional[int] = None
    api_key: Optional[str] = None
    python_version: str = "3.11"
    # Pin torch/torchaudio to a version whose torchaudio.load still uses the
    # bundled-libsndfile *soundfile* backend (≤2.7.x). Newer torchaudio routes
    # audio loading through torchcodec, which needs system FFmpeg shared libs —
    # avoided here so the wrapper stays self-contained (no external binaries).
    # Bump deliberately and re-verify the soundfile path (SPEC §14.1).
    torch_version: str = "2.7.0"
    torchaudio_version: str = "2.7.0"
    # coqui-tts (XTTS) imports `transformers.pytorch_utils.isin_mps_friendly`,
    # which transformers REMOVED in 5.x. coqui-tts only declares `transformers>=
    # 4.57` (no upper bound), so an unconstrained resolve picks 5.x and XTTS
    # crashes on model-load ("cannot import name 'isin_mps_friendly'"). Pin to
    # the last 4.x line (4.57.x still has the symbol AND satisfies >=4.57); f5-tts
    # declares no transformers bound, so it accepts this too. Passed verbatim to
    # `uv pip install` in step_deps and asserted by `_verify_transformers`.
    transformers_constraint: str = "transformers>=4.57,<5"
    # TTS backend engine the worker pool loads at startup (one per process):
    #   "f5"   → F5-TTS finetunes (needs a ref-text per sample; CC-BY-NC weights)
    #   "xtts" → Coqui XTTS-v2 (clones from audio only, no ref-text; CPML weights)
    # The bootstrap installs BOTH engines + all their models, so switching is a
    # restart with a different --tts-backend (no reinstall). See xtts_backend.py.
    tts_backend: str = "f5"
    # Active language at startup; selects which model the worker pool loads.
    # The per-request ``language`` field must match this (or be omitted).
    language: str = "de"
    # Per-language model definitions. ``en`` is the official multilingual base
    # (auto-downloaded, no ckpt/vocab needed); de/fr/ja are community finetunes
    # (CC-BY-NC) resolved from HuggingFace at install time. Verify repo/file
    # names against F5-TTS SHARED.md when bumping (SPEC §14.3).
    languages: dict = field(
        default_factory=lambda: {
            "en": {"arch": "F5TTS_v1_Base"},
            "de": {
                "arch": "F5TTS_Base",
                "hf_repo": "hvoss-techfak/F5-TTS-German",
                "ckpt_file": "model_f5tts_german.safetensors",
                "vocab_file": "vocab.txt",
            },
            "fr": {
                "arch": "F5TTS_Base",
                "hf_repo": "RASPIAUDIO/F5-French-MixedSpeakers-reduced",
                "ckpt_file": "model_last_reduced.pt",
                "vocab_file": "vocab.txt",
            },
            "ja": {
                "arch": "F5TTS_Base",
                "hf_repo": "Jmica/F5TTS",
                "ckpt_file": "JA_21999120/model_21999120.pt",
                "vocab_file": "JA_21999120/vocab_japanese.txt",
            },
        }
    )
    samples_dir: str = "samples"
    models_dir: str = "models"
    max_workers: Optional[int] = None
    vram_reserve_gb: float = 1.5
    per_job_gb: float = 3.0
    max_queue: int = 64
    max_chars_per_chunk: int = 250
    # XTTS token-streaming granularity: audio tokens per streamed chunk handed to
    # `inference_stream` (lower = lower first-audio latency, slightly more
    # overhead). XTTS default is 20. F5 has no token streaming and ignores this.
    stream_chunk_size: int = 20
    # Load the XTTS model in half precision (fp16) for a ~1.3-1.8x inference
    # speedup on a GPU. Opt-in (default off) and **only applied on a CUDA device**
    # (covers NVIDIA and ROCm; ignored on CPU/dml/xpu, where fp16 is unsupported
    # or slower). F5 ignores this. Experimental — verify audio quality per voice.
    xtts_fp16: bool = False
    asr_for_missing_ref_text: bool = True
    allowed_sample_ext: list[str] = field(
        default_factory=lambda: [".wav", ".flac", ".mp3"]
    )
    hf_endpoint: Optional[str] = None
    torch_index_override: Optional[str] = None
    log_level: str = "info"

    # ---- resolved absolute paths (computed, not part of the JSON schema) ----
    @property
    def root(self) -> Path:
        return WRAPPER_ROOT

    @property
    def samples_path(self) -> Path:
        return self._resolve(self.samples_dir)

    @property
    def models_path(self) -> Path:
        return self._resolve(self.models_dir)

    @property
    def custom_model_path(self) -> Path:
        """Directory a user-installed custom model lives in (may not exist)."""
        return self.models_path / CUSTOM_MODEL_DIRNAME

    def _resolve(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (WRAPPER_ROOT / p)

    @property
    def normalized_exts(self) -> set[str]:
        """Lower-cased extensions with a leading dot, for matching."""
        out: set[str] = set()
        for e in self.allowed_sample_ext:
            e = e.lower()
            out.add(e if e.startswith(".") else f".{e}")
        return out


_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


def _coerce(name: str, raw: Any, current: Any) -> Any:
    """Coerce a string (from ENV/CLI) to the type of the existing field value."""
    if raw is None or not isinstance(raw, str):
        return raw
    # Choose the target type from the dataclass default when current is None.
    if name in ("parent_pid", "max_workers"):
        return None if raw == "" or raw.lower() == "null" else int(raw)
    if name in ("port", "max_queue", "max_chars_per_chunk", "stream_chunk_size"):
        return int(raw)
    if name in ("vram_reserve_gb", "per_job_gb"):
        return float(raw)
    if name in ("asr_for_missing_ref_text", "xtts_fp16"):
        low = raw.lower()
        if low in _BOOL_TRUE:
            return True
        if low in _BOOL_FALSE:
            return False
        raise ValueError(f"invalid bool for {name}: {raw!r}")
    if name == "allowed_sample_ext":
        return [s.strip() for s in raw.split(",") if s.strip()]
    if name == "languages":
        return json.loads(raw)
    if name in ("api_key", "hf_endpoint", "torch_index_override"):
        return None if raw == "" or raw.lower() == "null" else raw
    return raw


def _field_names() -> list[str]:
    return [f.name for f in fields(Config)]


def load_config(
    argv: Optional[list[str]] = None,
    config_path: Optional[Path] = None,
    env: Optional[dict[str, str]] = None,
) -> Config:
    """Build a :class:`Config` applying JSON < ENV < CLI precedence."""
    env = dict(os.environ if env is None else env)
    names = _field_names()

    # 1) JSON base
    path = config_path or DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        data = {k: v for k, v in data.items() if k in names}

    cfg = Config(**data)

    # 2) ENV overrides
    for name in names:
        env_key = ENV_PREFIX + name.upper()
        if env_key in env:
            setattr(cfg, name, _coerce(name, env[env_key], getattr(cfg, name)))

    # 3) CLI overrides
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--start", action="store_true", help="Start the server.")
    parser.add_argument("--config", dest="config", default=None)
    for name in names:
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, default=None)
    ns, _unknown = parser.parse_known_args(argv)

    for name in names:
        val = getattr(ns, name)
        if val is not None:
            setattr(cfg, name, _coerce(name, val, getattr(cfg, name)))

    return cfg
