"""Structured NDJSON event sink for stdout (SPEC §8.1).

Every line written here is exactly one JSON object terminated by ``\\n`` and
immediately flushed, so the C# host can read events line-by-line as they happen.
This module is the *single* place allowed to write to stdout — everything else
(logs, tracebacks) goes to stderr — so the stream stays machine-parseable.

The host expects ``PYTHONUNBUFFERED=1`` and ``PYTHONUTF8=1`` in the environment
(SPEC §8.1); we still flush explicitly and force UTF-8 here so a missing env var
cannot corrupt the protocol.
"""

from __future__ import annotations

import json
import sys
import threading
from typing import Any, Optional

# A single lock so concurrent threads (worker pool, watchdog, request handlers)
# never interleave half-written JSON lines on stdout.
_lock = threading.Lock()


def _write(obj: dict[str, Any]) -> None:
    line = json.dumps(obj, ensure_ascii=False)
    with _lock:
        # Reconfigure lazily in case the parent did not set PYTHONUTF8.
        stream = sys.stdout
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
        try:
            stream.write(line + "\n")
            stream.flush()
        except (BrokenPipeError, OSError):
            # The host/bootstrap that reads our stdout has gone away (e.g. window
            # closed). Nothing to do — we're shutting down anyway; don't crash.
            pass


def starting() -> None:
    """First event emitted by the bootstrap before any work begins."""
    _write({"event": "starting"})


def progress(
    index: int,
    total: int,
    step: str,
    message: Optional[str] = None,
    percent: Optional[int] = None,
    done: bool = False,
    skipped: bool = False,
) -> None:
    """Install/synthesis progress event (SPEC §3.1).

    ``index`` is 1-based, ``total`` is the fixed step count, ``step`` is a stable
    key (e.g. ``"deps"``), ``message`` is human-readable. ``percent`` (0-100)
    animates a download bar; ``done``/``skipped`` mark step completion.
    """
    obj: dict[str, Any] = {
        "event": "progress",
        "index": index,
        "total": total,
        "step": step,
    }
    if message is not None:
        obj["message"] = message
    if percent is not None:
        obj["percent"] = int(percent)
    if done:
        obj["done"] = True
    if skipped:
        obj["skipped"] = True
    _write(obj)


def ready(host: str, port: int, backend: str, device: str, workers: int) -> None:
    """Server is listening on host:port (SPEC §8.1)."""
    _write(
        {
            "event": "ready",
            "host": host,
            "port": port,
            "backend": backend,
            "device": device,
            "workers": workers,
        }
    )


def log(message: str, level: str = "info") -> None:
    """Informational log line surfaced to the host UI."""
    _write({"event": "log", "level": level, "message": message})


def error(message: str, fatal: bool = False) -> None:
    """Error event. ``fatal=True`` means the process will exit non-zero."""
    _write({"event": "error", "message": message, "fatal": fatal})


def shutdown() -> None:
    """Emitted right before a graceful exit."""
    _write({"event": "shutdown"})
