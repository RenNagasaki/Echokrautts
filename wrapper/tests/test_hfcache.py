"""How this wrapper configures the HuggingFace cache.

Short rules, each learned from a failure, which is why they are pinned: the
redirect must OVERRIDE a machine-wide cache rather than defer to it, and the
symlink warning must not be printed at users who cannot do anything about it.
"""

from __future__ import annotations

import os
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
