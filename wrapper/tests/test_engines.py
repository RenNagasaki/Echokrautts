"""The engine registry (`src/engines.py`) — the one list of engines this build runs."""

from pathlib import Path

import pytest

from src import engines
from src.engine import KNOWN_BACKENDS

SRC = Path(__file__).resolve().parent.parent / "src"


def test_engine_module_reexports_the_registry():
    assert KNOWN_BACKENDS == engines.KNOWN_BACKENDS == tuple(engines.ENGINES)


@pytest.mark.parametrize("spec", list(engines.ENGINES.values()), ids=lambda s: s.id)
def test_every_engine_names_an_existing_download_module(spec):
    """The bootstrap runs `python -m <model_module>`; a typo there only shows up
    on a user's machine, mid-install."""
    package, _, module = spec.model_module.partition(".")
    assert package == "src"
    assert (SRC / f"{module}.py").is_file(), spec.model_module


def test_spec_ids_match_their_keys():
    assert all(key == spec.id for key, spec in engines.ENGINES.items())


def test_per_language_engines():
    """F5 (finetunes) and Pocket load one model per language; the others are multilingual."""
    assert [s.id for s in engines.ENGINES.values() if s.per_language] == ["f5", "pocket"]


def test_spec_for_rejects_an_unknown_engine_with_the_valid_names():
    with pytest.raises(ValueError) as err:
        engines.spec_for("xtts2")
    message = str(err.value)
    assert "'xtts2'" in message
    for known in KNOWN_BACKENDS:
        assert known in message


def test_registry_is_importable_without_torch():
    """The bootstrap imports it before any venv exists."""
    source = (SRC / "engines.py").read_text(encoding="utf-8")
    for heavy in ("import torch", "import transformers", "import numpy"):
        assert heavy not in source
