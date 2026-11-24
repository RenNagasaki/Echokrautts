"""How this wrapper configures the HuggingFace cache.

Short rules, each learned from a failure, which is why they are pinned: the
redirect must OVERRIDE a machine-wide cache rather than defer to it, and the
symlink warning must not be printed at users who cannot do anything about it.
"""

from __future__ import annotations

import builtins
import os
import sys
import types
from pathlib import Path

import pytest

from src import hfcache
from src.config import Config


def test_symlink_warning_is_silenced(monkeypatch):
    """It describes a fallback that works, so it is noise in front of a
    non-problem — and red text in an installer log makes users write in."""
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS_WARNING", raising=False)
    hfcache.silence_symlink_warning()
    assert os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] == "1"


def test_an_explicit_setting_is_respected(monkeypatch):
    """setdefault, not set: somebody who deliberately wants the warning keeps it."""
    monkeypatch.setenv("HF_HUB_DISABLE_SYMLINKS_WARNING", "0")
    hfcache.silence_symlink_warning()
    assert os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] == "0"


def test_a_machine_wide_cache_is_overridden(monkeypatch, tmp_path):
    """SET, not setdefault. Deferring to an existing HF_HUB_CACHE made a worker
    download the same weights a second time — outside the models volume, on
    every container start."""
    monkeypatch.setenv("HF_HUB_CACHE", "G:/somewhere/else")
    monkeypatch.setenv("HF_HOME", "G:/somewhere/else")
    config = Config(models_dir=str(tmp_path))

    hfcache.use_models_dir(config)

    assert os.environ["HF_HUB_CACHE"] == str(config.models_path)
    assert os.environ["HF_HOME"] == str(config.models_path)


def test_remote_code_cache_stays_in_the_volume(monkeypatch, tmp_path):
    """Models that ship their own code are written to a SEPARATE cache; it has
    to live inside the models directory too, or it lands outside the volume."""
    monkeypatch.setenv("HF_MODULES_CACHE", "G:/elsewhere")
    config = Config(models_dir=str(tmp_path))

    hfcache.use_models_dir(config)

    assert Path(os.environ["HF_MODULES_CACHE"]).parent == config.models_path


def test_a_configured_endpoint_is_exported(monkeypatch, tmp_path):
    config = Config(models_dir=str(tmp_path), hf_endpoint="https://mirror.example")
    hfcache.use_models_dir(config)
    assert os.environ["HF_ENDPOINT"] == "https://mirror.example"


def test_no_endpoint_is_not_invented(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    hfcache.use_models_dir(Config(models_dir=str(tmp_path)))
    assert "HF_ENDPOINT" not in os.environ


def test_redirecting_also_silences_the_warning(monkeypatch, tmp_path):
    """The two belong together — every caller that downloads wants both."""
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS_WARNING", raising=False)
    hfcache.use_models_dir(Config(models_dir=str(tmp_path)))
    assert os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] == "1"


# ---------------------------------------------------------------- Xet warning

def test_xet_warning_is_silenced_when_the_package_is_absent(monkeypatch):
    """Per-FILE nagging turns a model download into a wall of red text.

    Without `hf_xet` the download already falls back to plain HTTP, so the flag
    changes nothing except whether that fallback is announced once per file.
    """
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    real_import = builtins.__import__

    def without_hf_xet(name, *args, **kwargs):
        if name == "hf_xet":
            raise ImportError("no hf_xet")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_hf_xet)
    hfcache.silence_xet_warning()
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"


def test_xet_stays_enabled_when_the_package_IS_installed(monkeypatch):
    """The flag also switches the feature OFF.

    Setting it unconditionally would rob a user who installed `hf_xet` of
    exactly the speed-up they installed it for.
    """
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.setitem(sys.modules, "hf_xet", types.ModuleType("hf_xet"))
    hfcache.silence_xet_warning()
    assert "HF_HUB_DISABLE_XET" not in os.environ


def test_an_explicit_xet_choice_is_respected(monkeypatch):
    """setdefault, not set: whoever set it meant it."""
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    monkeypatch.delitem(sys.modules, "hf_xet", raising=False)
    hfcache.silence_xet_warning()
    assert os.environ["HF_HUB_DISABLE_XET"] == "0"


def test_use_models_dir_also_silences_xet(tmp_path, monkeypatch):
    """One call has to establish the whole policy, or callers drift apart."""
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.delitem(sys.modules, "hf_xet", raising=False)
    hfcache.use_models_dir(Config(models_dir=str(tmp_path)))
    assert os.environ.get("HF_HUB_DISABLE_XET") == "1"


def _fake_hub_with_racy_probe(monkeypatch):
    """A stand-in for huggingface_hub.file_download whose probe has the real
    library's flaw: it caches ``True`` first and corrects it after a slow test."""
    import threading
    import time

    cache: dict = {}
    fd = types.ModuleType("huggingface_hub.file_download")

    def are_symlinks_supported(cache_dir=None):
        if cache_dir not in cache:
            cache[cache_dir] = True
            time.sleep(0.05)  # the test symlink being attempted
            cache[cache_dir] = False  # it failed: no Developer Mode
        return cache[cache_dir]

    fd.are_symlinks_supported = are_symlinks_supported
    hub = types.ModuleType("huggingface_hub")
    hub.file_download = fd
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.file_download", fd)
    return fd, threading


def _ask_from_threads(fd, threading, n=8):
    answers = []
    barrier = threading.Barrier(n)

    def ask():
        barrier.wait()
        answers.append(fd.are_symlinks_supported("models--tts-hub--XTTS-v2"))

    threads = [threading.Thread(target=ask) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return answers


def test_the_unguarded_probe_lies_to_parallel_threads(monkeypatch):
    """Pins the failure itself: without the guard, threads asking while the
    probe runs are told symlinks work — and then die with WinError 1314."""
    fd, threading = _fake_hub_with_racy_probe(monkeypatch)
    assert True in _ask_from_threads(fd, threading)


def test_the_serialized_probe_gives_every_thread_the_real_answer(monkeypatch):
    """XTTS downloads through an eight-thread snapshot_download. A user's
    install died on exactly this (2026-11-01)."""
    fd, threading = _fake_hub_with_racy_probe(monkeypatch)
    hfcache.serialize_symlink_probe()
    assert _ask_from_threads(fd, threading) == [False] * 8


def test_serializing_twice_wraps_once(monkeypatch):
    fd, _ = _fake_hub_with_racy_probe(monkeypatch)
    hfcache.serialize_symlink_probe()
    first = fd.are_symlinks_supported
    hfcache.serialize_symlink_probe()
    assert fd.are_symlinks_supported is first


def test_without_the_hub_installed_nothing_happens(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    hfcache.serialize_symlink_probe()  # must not raise
