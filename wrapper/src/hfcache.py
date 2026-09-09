"""One place for how this wrapper talks to the HuggingFace cache.

Two engines download weights from the hub and both used to decide this for
themselves, which is how a single policy turns into two slightly different ones.
The rules are short but each was learned from a failure, so they are stated once
here and imported rather than repeated.
"""

from __future__ import annotations

import os
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
    cache = str(config.models_path)
    os.environ["HF_HOME"] = cache
    os.environ["HF_HUB_CACHE"] = cache
    os.environ["HF_MODULES_CACHE"] = str(Path(cache) / "modules")
    if config.hf_endpoint:
        os.environ["HF_ENDPOINT"] = config.hf_endpoint
