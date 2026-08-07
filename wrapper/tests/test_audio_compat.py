"""Tests for the torchaudio.load shim (see src/audio_compat.py).

torch is mocked away in this suite, so these drive the module against fake
torchaudio objects — which is also the point of keeping `_needs_shim` pure.
"""

import types

import pytest

from src import audio_compat, ndjson


@pytest.fixture(autouse=True)
def _reset_log_once():
    ndjson.reset_log_once()
    yield
    ndjson.reset_log_once()


def _fake_torchaudio(version):
    mod = types.SimpleNamespace(__version__=version)
    mod.load = lambda *a, **k: ("stock", 0)
    return mod


# --------------------------------------------------------------- _needs_shim
@pytest.mark.parametrize(
    "version,codec,expected",
    [
        ("2.9.1", False, True),  # the AMD/Windows case: 2.9 without torchcodec
        ("2.10.0", False, True),
        ("2.9.1", True, False),  # torchcodec present → stock path works
        ("2.8.0", False, False),  # still decodes natively
        ("2.7.0", False, False),  # the pinned version everywhere else
        ("not-a-version", False, False),  # never patch what we cannot parse
    ],
)
def test_needs_shim(version, codec, expected):
    assert audio_compat._needs_shim(version, codec) is expected


# ------------------------------------------------------------------- patching
def test_patches_torchaudio_29(monkeypatch):
    monkeypatch.setattr(audio_compat, "_needs_shim", lambda *a: True)
    ta = _fake_torchaudio("2.9.1")
    reason = audio_compat.ensure_native_audio_loading(ta)
    assert reason and "soundfile" in reason
    assert ta.load is audio_compat._soundfile_load


def test_leaves_pinned_torchaudio_alone():
    ta = _fake_torchaudio("2.7.0")
    stock = ta.load
    assert audio_compat.ensure_native_audio_loading(ta) is None
    assert ta.load is stock


def test_patching_is_idempotent(monkeypatch):
    monkeypatch.setattr(audio_compat, "_needs_shim", lambda *a: True)
    ta = _fake_torchaudio("2.9.1")
    assert audio_compat.ensure_native_audio_loading(ta) is not None
    # Second call must be a no-op, so repeated engine starts don't re-log or
    # wrap the shim in itself.
    assert audio_compat.ensure_native_audio_loading(ta) is None
    assert ta.load is audio_compat._soundfile_load


def test_version_with_local_suffix_is_parsed():
    # AMD's wheels report "2.9.1+rocm7.2.1" — the local part must not defeat the
    # version check (torchcodec is genuinely absent in this venv).
    ta = _fake_torchaudio("2.9.1+rocm7.2.1")
    assert audio_compat.ensure_native_audio_loading(ta) is not None
    assert ta.load is audio_compat._soundfile_load


def test_missing_torchaudio_is_not_an_error(monkeypatch):
    # The unit venv has no torch at all; calling this must stay harmless.
    assert audio_compat.ensure_native_audio_loading() is None
