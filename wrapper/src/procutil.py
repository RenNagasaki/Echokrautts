"""Subprocess helpers that never flash a console window on Windows (SPEC §8.5).

The host runs us hidden (``CreateNoWindow=true``). Any child process we spawn
(``nvidia-smi``, ``uv``, ``pip`` …) would otherwise pop a cmd window for a split
second. Routing every spawn through here with ``CREATE_NO_WINDOW`` prevents that.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Sequence

IS_WINDOWS = sys.platform.startswith("win")

# Combine into kwargs we can splat into any subprocess call.
NO_WINDOW_KWARGS: dict = {}
if IS_WINDOWS:
    NO_WINDOW_KWARGS["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]


def run(
    args: Sequence[str],
    timeout: float | None = None,
    check: bool = False,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    """Run ``args`` capturing text output, hidden, with no window."""
    return subprocess.run(
        list(args),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
        env=env,
        **NO_WINDOW_KWARGS,
    )


def try_run(args: Sequence[str], timeout: float | None = 10.0) -> str | None:
    """Run a probe command; return stdout on success, ``None`` on any failure."""
    try:
        proc = run(args, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout
