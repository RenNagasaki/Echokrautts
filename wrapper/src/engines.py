"""The one list of engines this build can run — data only, no heavy imports.

Before this module the set of engines lived in several places that had to agree
by convention: ``engine.KNOWN_BACKENDS``, the branches of the bootstrap's install
and download steps, and ``engines.json`` in the repo root (what the Echokraut
plugin offers in its dropdown). A test now holds ``engines.json`` against this
list, and the bootstrap asks it which venv and which download a given engine
needs (plan ``docs/plans/qwen3-und-lazy-install.md`` §3.1).

Importable from the bootstrap (stdlib-only interpreter) and from both venvs, so
it must stay free of torch/transformers imports.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineSpec:
    id: str
    """Value of ``--tts-backend``; identical to the ``id`` in ``engines.json``."""
    name: str
    """Display name for user-facing messages (log lines, HTTP errors)."""
    group: str
    """venv group the engine lives in. Engines in one group share a venv and
    must therefore agree on their dependencies (transformers above all)."""
    model_module: str
    """``python -m <module>`` downloads this engine's weights (idempotent)."""
    per_language: bool = False
    """True when the engine loads one model per language (F5's finetunes), so a
    language switch needs a download of its own (``--active-language``)."""
    optional_model_modules: tuple[str, ...] = ()
    """Further downloads whose failure is NOT fatal (MOSS's ONNX graphs: without
    them the engine falls back to its slower PyTorch runtime and says so)."""
    model_variant: tuple[str, ...] = ()
    """Config fields that select WHICH files the model download fetches (Pocket:
    ``pocket_layers`` 6/24 and ``pocket_revision``). They are part of the model
    marker, so changing one downloads again instead of trusting the old marker —
    live 2026-11-13: a mirror switch kept ``model-pocket-de-24``, skipped the
    download and the server died on the empty new revision folder."""
    gpu_backends: tuple[str, ...] = ()
    """Detection backends (``gpu_detect.Detection.backend``) the engine can run
    on; empty = any, CPU included. A non-empty list also means NO CPU fallback:
    the bootstrap refuses the install up front, and a GPU worker that fails to
    build stops the server instead of quietly rebuilding on the CPU — where
    such an engine would be unusably slow."""


CLASSIC_GROUP = "classic"
"""f5 + xtts + moss: transformers >=4.57,<5 (coqui needs <5), the historical ``.venv``."""
QWEN_GROUP = "qwen"
"""Qwen3-TTS: faster-qwen3-tts needs transformers 5 → its own ``.venv-qwen``."""

# CUDA graphs need a CUDA device. ROCm reports itself as one; whether the graphs
# capture under HIP is untested — allowed as experimental (user, 2026-11-13).
_QWEN_GPUS = ("cuda", "rocm", "rocm_win")

ENGINES: dict[str, EngineSpec] = {
    spec.id: spec
    for spec in (
        EngineSpec("f5", "F5-TTS", CLASSIC_GROUP, "src.models", per_language=True),
        EngineSpec("xtts", "XTTS", CLASSIC_GROUP, "src.xtts_backend"),
        EngineSpec(
            "moss", "MOSS-TTS-Nano", CLASSIC_GROUP, "src.moss_backend",
            optional_model_modules=("src.moss_onnx_backend",),
        ),
        # CPU engine, one model per language (like F5); fits the classic venv
        # (no transformers needed, hub/tokenizers/numpy ranges agree).
        EngineSpec("pocket", "Pocket TTS", CLASSIC_GROUP, "src.pocket_backend", per_language=True,
                   model_variant=("pocket_layers", "pocket_revision")),
        EngineSpec("qwen-0.6b", "Qwen 0.6B", QWEN_GROUP, "src.qwen_backend", gpu_backends=_QWEN_GPUS),
        EngineSpec("qwen-1.7b", "Qwen 1.7B", QWEN_GROUP, "src.qwen_backend", gpu_backends=_QWEN_GPUS),
    )
}

KNOWN_BACKENDS: tuple[str, ...] = tuple(ENGINES)

GROUPS: tuple[str, ...] = tuple(dict.fromkeys(spec.group for spec in ENGINES.values()))


def engines_in(group: str) -> list[EngineSpec]:
    """Every engine that lives in venv group ``group``, in registry order."""
    return [spec for spec in ENGINES.values() if spec.group == group]


def unsupported_device_reason(spec: EngineSpec, backend: str) -> str | None:
    """Why ``spec`` cannot run on detection ``backend`` — None when it can."""
    if not spec.gpu_backends or backend in spec.gpu_backends:
        return None
    from . import i18n

    return i18n.t("boot.unsupported_device", name=spec.name, backend=backend)


def spec_for(backend: str) -> EngineSpec:
    """The spec of ``backend``; ``ValueError`` listing the valid names otherwise.

    Refusing is deliberate (2026-09-09): F5 used to be both the default and the
    last branch, so an unknown name silently became F5 — wrong voice, no error.
    """
    try:
        return ENGINES[backend]
    except KeyError:
        raise ValueError(
            f"unknown tts_backend {backend!r} — known backends: {', '.join(KNOWN_BACKENDS)}"
        ) from None
