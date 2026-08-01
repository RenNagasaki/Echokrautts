"""The bootstrap → server config handover.

The server runs as a subprocess that re-runs ``load_config`` from scratch, so a
flag the host passed to ``bootstrap.py`` only reaches it via the forwarded
``F5W_*`` environment. These tests pin that handover as a round trip: whatever
the bootstrap resolved must come back out of ``load_config`` unchanged.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from src.config import Config, load_config

WRAPPER_ROOT = Path(__file__).resolve().parent.parent


def _load_bootstrap():
    """Import bootstrap.py by path (it is a script, not part of the package)."""
    spec = importlib.util.spec_from_file_location(
        "bootstrap_under_test", WRAPPER_ROOT / "bootstrap" / "bootstrap.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bootstrap():
    return _load_bootstrap()


def _roundtrip(bootstrap, config: Config, tmp_path: Path) -> Config:
    """Config → forwarded env → what the server subprocess would resolve.

    Uses a non-existent config path so the result comes purely from the env,
    exactly like a server whose config.json still holds the old defaults.
    """
    env = bootstrap._config_env(config)
    return load_config(argv=[], config_path=tmp_path / "absent.json", env=env)


def test_cli_only_flags_reach_the_server(bootstrap, tmp_path):
    # The regression: the host starts the wrapper with --xtts-fp16 true, but the
    # server used to re-read config.json (fp16 false) and silently ran without
    # it. Every field must survive, not just the few once listed by hand.
    resolved = Config(
        tts_backend="xtts",
        language="fr",
        xtts_fp16=True,
        stream_chunk_size=7,
        max_chars_per_chunk=120,
        api_key="secret",
    )
    out = _roundtrip(bootstrap, resolved, tmp_path)

    assert out.xtts_fp16 is True
    assert out.stream_chunk_size == 7
    assert out.max_chars_per_chunk == 120
    assert out.tts_backend == "xtts"
    assert out.language == "fr"
    assert out.api_key == "secret"


def test_every_field_survives_the_roundtrip(bootstrap, tmp_path):
    # A blanket check so a newly added config field cannot quietly fall out of
    # the handover (and, if its ENV coercion is missing, fails loudly here).
    resolved = Config()
    out = _roundtrip(bootstrap, resolved, tmp_path)

    for name in vars(resolved):
        assert getattr(out, name) == getattr(resolved, name), f"{name} did not survive"


def test_falsy_values_survive(bootstrap, tmp_path):
    # False must arrive as False, not as the truthy string "False", and None
    # must stay None rather than becoming the empty string.
    resolved = Config(xtts_fp16=False, asr_for_missing_ref_text=False, api_key=None)
    out = _roundtrip(bootstrap, resolved, tmp_path)

    assert out.xtts_fp16 is False
    assert out.asr_for_missing_ref_text is False
    assert out.api_key is None


def test_parent_pid_is_not_clobbered_by_bulk_forwarding(bootstrap, monkeypatch, tmp_path):
    # _server_env applies the bulk forwarding first and then sets F5W_PARENT_PID,
    # which falls back to the bootstrap's own pid when none was given. Order
    # matters: the bulk pass would otherwise leave it empty.
    monkeypatch.setattr(bootstrap.os, "getpid", lambda: 4242)
    config = Config(models_dir=str(tmp_path / "models"))

    env = bootstrap._server_env(config)
    assert env["F5W_PARENT_PID"] == "4242"

    explicit = bootstrap._server_env(Config(parent_pid=99, models_dir=str(tmp_path / "models")))
    assert explicit["F5W_PARENT_PID"] == "99"
