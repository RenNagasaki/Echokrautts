"""One place for how this wrapper talks to the HuggingFace cache.

Two engines download weights from the hub and both used to decide this for
themselves, which is how a single policy turns into two slightly different ones.
The rules are short but each was learned from a failure, so they are stated once
here and imported rather than repeated.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from .config import Config


def silence_symlink_warning() -> None:
    """Stop huggingface_hub from warning that it cannot create symlinks.

    Its cache stores each file once under ``blobs/`` and links it into
    ``snapshots/``. Creating a symlink on Windows needs a privilege a normal
    account does not have, so on most user machines the library prints a
    paragraph about Developer Mode and then quietly falls back to copying —
    which works. The warning is therefore noise in front of a non-problem, and
    noise in an installer log is not free: a user who sees red text asks whether
    their install is broken, exactly as happened with the unrelated ``pydub``
    ffmpeg warning.

    The cost of the fallback is real but small: a file shared by two models is
    stored twice. That is a disk trade, not a failure, and nothing here can
    change it without asking users to enable Developer Mode.
    """
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


_probe_lock = threading.Lock()


def serialize_symlink_probe() -> None:
    """Make huggingface_hub's symlink probe safe under a parallel download.

    ``are_symlinks_supported`` caches its answer per directory, but writes
    ``True`` into that cache BEFORE it has tested anything and only flips it to
    ``False`` once the test symlink failed. ``snapshot_download`` fetches with
    eight threads; every thread that asks during that window reads the
    provisional ``True``, calls ``os.symlink`` for real and dies with
    ``WinError 1314`` on a Windows account without Developer Mode. Single-file
    downloads (``hf_hub_download``, used for F5) never hit it — the XTTS
    download through Coqui's ModelManager does, as a user's install log showed
    (2026-11-01): the "cannot create symlinks" warning and the fatal symlink
    error came out of the same second.

    The fix is to hold a lock around the whole probe, so the second thread waits
    for the real answer. ``_create_symlink`` looks the function up as a module
    global, so replacing that attribute is enough. Idempotent.
    """
    try:
        from huggingface_hub import file_download  # lazy: heavy import
    except ImportError:
        return  # nothing downloads from the hub then, so nothing can race

    original = file_download.are_symlinks_supported
    if getattr(original, "_echokraut_serialized", False):
        return

    def locked(cache_dir=None):
        with _probe_lock:
            return original(cache_dir)

    locked._echokraut_serialized = True  # type: ignore[attr-defined]
    file_download.are_symlinks_supported = locked


def silence_xet_warning() -> None:
    """Stop the hub nagging about ``hf_xet`` on every single downloaded file.

    Repos on Xet storage make huggingface_hub warn, per file, that installing
    ``hf_xet`` would download faster. With a few dozen files in a model that is
    a wall of red text during an install — the same kind of noise as the symlink
    warning, and the kind that makes users write in asking whether their install
    is broken.

    **Only when ``hf_xet`` is genuinely absent.** ``HF_HUB_DISABLE_XET`` also
    switches the feature off, so setting it unconditionally would rob a user who
    later installs the package of the speed-up they installed it for. When it is
    missing, the download already falls back to plain HTTP — the flag then
    changes nothing except whether the fallback is announced forty times.

    The hub reads this constant at IMPORT time, so this has to run before
    huggingface_hub is imported. :func:`use_models_dir` already has to, for the
    same reason.
    """
    try:
        import hf_xet  # noqa: F401 — presence is the whole question
    except ImportError:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def use_models_dir(config: Config) -> None:
    """Point every HuggingFace cache at the wrapper's own ``models/`` directory.

    **SET, not setdefault** — that distinction is the whole point. A machine-wide
    ``HF_HUB_CACHE`` (plenty of people have one) would otherwise send a worker
    off to download the same weights a second time, and in a container they
    would land outside the ``/data/models`` volume on every single start. Found
    live, twice.

    ``HF_MODULES_CACHE`` is separate on purpose: a model shipping its own code
    (``trust_remote_code``) is written there, not into the hub cache, and it also
    has to stay inside the volume. Callers that load such a model must call this
    **before importing transformers**, which resolves these paths at import time.
    """
    silence_symlink_warning()
    silence_xet_warning()
    cache = str(config.models_path)
    os.environ["HF_HOME"] = cache
    os.environ["HF_HUB_CACHE"] = cache
    os.environ["HF_MODULES_CACHE"] = str(Path(cache) / "modules")
    if config.hf_endpoint:
        os.environ["HF_ENDPOINT"] = config.hf_endpoint
