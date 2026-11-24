"""Per-file download progress → NDJSON, by hooking ``tqdm`` (SPEC §3.1).

The model downloads run inside the bootstrap's "model" step as separate
sub-processes (``python -m src.models`` / ``python -m src.xtts_backend``), whose
stdout NDJSON is forwarded live to the host. Until now each model only emitted a
coarse start/done ``log`` line — no per-file percentage.

Both download backends render their byte-level progress through **tqdm**:

- ``huggingface_hub`` (F5 checkpoints en/de/fr/ja) — ``class tqdm(old_tqdm)``
  with ``old_tqdm = from tqdm.auto import tqdm``, byte bars ``unit="B"`` and the
  filename as ``desc``.
- coqui ``ModelManager`` (XTTS-v2) — ``from tqdm import tqdm``, byte bars
  ``unit="iB"`` (no ``desc``).

Neither exposes a clean byte-callback, but both bind their ``tqdm`` reference at
*import* time, and this wrapper imports them lazily (inside the download
functions). So we swap the ``tqdm`` class for a thin subclass *before* that
import binds it, turning every per-file download bar into throttled
``ndjson.progress`` events the host already knows how to render — then restore
the original class afterwards.

Kept dependency-free at import time (no top-level ``tqdm`` import) so the unit
test suite — which mocks the heavy deps and doesn't install tqdm — imports this
module fine. When tqdm is genuinely absent at runtime the hook is a silent no-op
(the coarse start/done ``log`` lines still appear; only the animated bar is
lost).
"""

from __future__ import annotations

import contextlib
import os
from typing import Callable, Iterator

from . import ndjson

# Emit at most one progress event per this many percent (plus a final 100%), so
# a multi-GB download doesn't flood the NDJSON stream with thousands of lines.
_PCT_STEP = 3

Emit = Callable[[str, int], None]


def _make_subclass(base: type, emit: Emit) -> type:
    """Build a ``tqdm`` subclass of *base* that reports byte progress to *emit*.

    Only *byte* bars (``unit`` containing ``b``) are reported — tqdm's file-count
    / iteration bars are ignored. Reporting is throttled to ``_PCT_STEP`` and
    guarded so a failing ``emit`` never breaks the actual download.
    """

    class _NdjsonTqdm(base):  # type: ignore[valid-type,misc]
        def __init__(self, *args, **kwargs):
            # coqui creates bars with tqdm's default ``disable=None``, which
            # auto-disables on a non-tty (the bootstrap pipes stdout/stderr) →
            # ``n`` would never advance and we'd report nothing. Force it on when
            # tqdm would auto-decide; respect an explicit True/False (e.g. HF's
            # globally-disabled case).
            if kwargs.get("disable", None) is None:
                kwargs["disable"] = False
            super().__init__(*args, **kwargs)
            self._nd_last = -_PCT_STEP - 1
            self._nd_report()

        def update(self, n=1):
            ret = super().update(n)
            self._nd_report()
            return ret

        def close(self):
            self._nd_report(final=True)
            super().close()

        def _nd_report(self, final: bool = False) -> None:
            try:
                if getattr(self, "disable", False):
                    return
                unit = getattr(self, "unit", "") or ""
                if "b" not in unit.lower():  # only byte-download bars
                    return
                total = getattr(self, "total", None)
                if not total:  # unknown size → can't compute a percentage
                    return
                n = total if final else getattr(self, "n", 0)
                pct = min(100, int(n * 100 / total))
                if pct <= self._nd_last:  # never re-emit the same or lower percent
                    return
                if not final and pct - self._nd_last < _PCT_STEP:
                    return
                self._nd_last = pct
                label = (getattr(self, "desc", "") or "").strip() or "Download"
                emit(label, pct)
            except Exception:  # noqa: BLE001 — progress must never break a download
                pass

    return _NdjsonTqdm


@contextlib.contextmanager
def patched_downloads(emit: Emit) -> Iterator[None]:
    """Route byte-level download progress (tqdm) to ``emit(label, pct)``.

    Swaps the ``tqdm`` class on the ``tqdm`` / ``tqdm.auto`` / ``tqdm.std``
    modules for a reporting subclass, so both huggingface_hub and coqui pick it
    up when they bind their ``tqdm`` reference on first import. Restores the
    originals on exit. No-op (still yields) if tqdm isn't importable.
    """
    try:
        import tqdm as pkg
        from tqdm import auto as auto_mod
        from tqdm import std as std_mod
    except Exception:  # noqa: BLE001 — tqdm absent → progress simply won't animate
        yield
        return

    sub = _make_subclass(std_mod.tqdm, emit)
    targets = [(pkg, "tqdm"), (auto_mod, "tqdm"), (std_mod, "tqdm")]
    saved = [(mod, name, getattr(mod, name)) for mod, name in targets]
    for mod, name in targets:
        setattr(mod, name, sub)
    try:
        yield
    finally:
        for mod, name, orig in saved:
            setattr(mod, name, orig)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


class ModelProgress:
    """Per-file download reporter for the bootstrap "model" step.

    Wrap the downloads in ``with mp.patch():`` and call ``mp.stage(prefix)``
    before each model so its byte bars are labelled (e.g. ``"de: "``). The
    ``index``/``total`` default to the bootstrap model step but honor
    ``F5W_STEP_INDEX`` / ``F5W_STEP_TOTAL`` (set by ``bootstrap.step_model``) so
    these sub-process bars land on the SAME "Step 5/6 · model" bar the host
    already shows around them — bootstrap stays the single source of the step
    numbering.
    """

    def __init__(self, step: str = "model") -> None:
        self._step = step
        self._prefix = ""
        self._index = _env_int("F5W_STEP_INDEX", 5)
        self._total = _env_int("F5W_STEP_TOTAL", 6)

    def stage(self, prefix: str) -> None:
        """Label subsequent byte bars (call before each model download)."""
        self._prefix = prefix

    def _emit(self, label: str, pct: int) -> None:
        message = f"{self._prefix}{label}" if self._prefix else label
        ndjson.progress(self._index, self._total, self._step, message=message, percent=pct)

    def patch(self):
        """Context manager that routes tqdm byte progress to this reporter."""
        return patched_downloads(self._emit)
