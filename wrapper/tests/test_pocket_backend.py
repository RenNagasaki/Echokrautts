"""Pocket TTS backend (`src/pocket_backend.py`) — without torch or pocket-tts."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from src import engines
from src.config import Config, load_config
from src.engine import _default_factory
from src.inline_ref import InlineReference
from src.pocket_backend import PocketWorker, _language_files, _model_dir, cpu_threads, decode_steps, language_folder, write_config

TEMPLATE = """weights_path: hf://kyutai/pocket-tts/languages/german_24l/model.safetensors@abc
weights_path_without_voice_cloning: hf://kyutai/pocket-tts-without-voice-cloning/languages/german_24l/model.safetensors@def

flow_lm:
  lookup_table:
    dim: 1024
    tokenizer: tokenizers
    tokenizer_path: hf://kyutai/pocket-tts-without-voice-cloning/languages/german_24l/tokenizer.json@def
  #weights_path: partial.safetensors
"""


class _FakeModel:
    sample_rate = 24000

    def __init__(self, config, sampler_decode_steps):
        self.config = config
        self.steps = sampler_decode_steps
        self.encoded = []
        self.generated = []

    def get_state_for_audio_prompt(self, ref):
        self.encoded.append(ref)
        return {"state": len(self.encoded)}

    def generate_audio_stream(self, state, text):
        self.generated.append((state, text))
        yield np.ones(4, dtype=np.float64)
        yield np.zeros((1, 3))


@pytest.fixture
def setup(tmp_path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "german_24l.yaml").write_text(TEMPLATE, encoding="utf-8")
    (template_dir / "english.yaml").write_text(TEMPLATE, encoding="utf-8")
    config = Config(tts_backend="pocket", language="de", models_dir=str(tmp_path / "models"))
    _model_dir(config).mkdir(parents=True)
    return config, template_dir


def _worker(config, template_dir):
    holder = {}

    def loader(config, sampler_decode_steps):
        holder["model"] = _FakeModel(config, sampler_decode_steps)
        return holder["model"]

    worker = PocketWorker(config, "cuda", model_loader=loader, template_dir=template_dir)
    holder["model"].encoded.clear()  # priming (no torch here → it fails softly)
    holder["model"].generated.clear()
    return worker, holder["model"]


def test_the_config_points_at_the_mirror_files_not_the_gated_repo(setup):
    config, template_dir = setup
    cfg = write_config(config, "german_24l", template_dir).read_text(encoding="utf-8")
    files = _language_files(config, "german_24l")
    assert f"weights_path: {files['model.safetensors'].as_posix()}" in cfg
    # The mirror is an older revision: only its own sentencepiece model fits.
    assert "tokenizer: sentencepiece" in cfg
    assert f"tokenizer_path: {files['tokenizer.model'].as_posix()}" in cfg
    assert "kyutai/pocket-tts/" not in cfg, "nothing may reach for the gated repo"
    assert "#weights_path: partial.safetensors" in cfg, "only the first weights_path is replaced"


def test_it_runs_on_the_cpu_whatever_was_detected(setup):
    worker, model = _worker(*setup)
    assert worker.device == "cpu"
    assert model.steps == 4, "user default: no reverb, ~15 % slower than 1"


def test_chunks_follow_the_contract(setup, tmp_path):
    worker, _ = _worker(*setup)
    ref = tmp_path / "voice.wav"
    ref.write_bytes(b"RIFF")
    parts = list(worker.infer_stream(str(ref), "", "Hallo.", 1.0, "de"))
    assert [p.dtype for p in parts] == [np.float32, np.float32]
    assert [p.shape for p in parts] == [(4,), (3,)]
    assert worker.infer(str(ref), "", "Hallo.", 32, 1.0).shape == (7,)


def test_the_voice_state_is_encoded_once(setup, tmp_path):
    worker, model = _worker(*setup)
    ref = tmp_path / "voice.wav"
    ref.write_bytes(b"RIFF")
    for _ in range(3):
        list(worker.infer_stream(str(ref), "", "Satz.", 1.0))
    assert len(model.encoded) == 1


def test_inline_reference_goes_in_as_a_tensor(setup, monkeypatch):
    fake_torch = types.SimpleNamespace(from_numpy=lambda a: types.SimpleNamespace(reshape=lambda *s: ("tensor", s, a.shape)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    worker, model = _worker(*setup)
    ref = InlineReference(voice_id="v1", audio=np.zeros(2400, dtype=np.float32), text="egal")
    list(worker.infer_stream(ref, "", "Satz.", 1.0))
    list(worker.infer_stream(ref, "", "Noch einer.", 1.0))
    assert model.encoded == [("tensor", (1, -1), (2400,))]


def test_no_japanese_model(setup):
    config, template_dir = setup
    with pytest.raises(ValueError) as err:
        language_folder(Config(tts_backend="pocket", language="ja"))
    assert "ja" in str(err.value) and "de" in str(err.value)
    with pytest.raises(ValueError):
        _worker(Config(tts_backend="pocket", language="ja", models_dir=config.models_dir), template_dir)


@pytest.mark.parametrize("code, size, folder", [
    ("de", 24, "german_24l"), ("fr", 24, "french_24l"), ("de", 6, "german"), ("it", 6, "italian"),
    ("en", 6, "english"),
])
def test_language_folders(code, size, folder):
    assert language_folder(Config(language=code, pocket_layers=size)) == folder


@pytest.mark.parametrize("code, size, folder", [("en", 24, "english"), ("fr", 6, "french_24l")])
def test_a_missing_size_falls_back_to_the_other(code, size, folder):
    """The mirror has no English 24-layer and no French 6-layer model."""
    assert language_folder(Config(language=code, pocket_layers=size)) == folder


def test_an_unknown_size_means_24():
    assert language_folder(Config(language="de", pocket_layers=12)) == "german_24l"


@pytest.mark.parametrize("size, configured, effective", [(24, 1, 1), (24, 8, 8), (6, 1, 4), (6, 4, 4), (6, 8, 8)])
def test_six_layers_never_decode_with_one_step(size, configured, effective):
    """User decision 2026-11-13: the small model only with 4 and 8 (WER 4.6 % at 1)."""
    assert decode_steps(Config(pocket_layers=size, pocket_decode_steps=configured)) == effective


def test_the_worker_loads_the_six_layer_model_with_the_guarded_steps(setup):
    config, template_dir = setup
    config.pocket_layers, config.pocket_decode_steps = 6, 1
    (template_dir / "german.yaml").write_text(TEMPLATE, encoding="utf-8")
    worker, model = _worker(config, template_dir)
    assert model.steps == 4
    assert model.config.endswith("german.yaml")


def test_the_model_marker_includes_size_and_revision():
    """Switching 24 → 6 or the mirror revision must download again, not trust the
    old marker (live 2026-11-13: a revision switch crashed on an empty folder)."""
    from bootstrap.bootstrap import _model_marker

    spec = engines.spec_for("pocket")
    rev = "f" * 40
    assert _model_marker(spec, "de", Config(pocket_layers=24, pocket_revision=rev)) == "model-pocket-de-24-ffffffffffff"
    assert _model_marker(spec, "de", Config(pocket_layers=6, pocket_revision=rev)) == "model-pocket-de-6-ffffffffffff"
    assert _model_marker(spec, "de", Config(pocket_revision="a" * 40)) != _model_marker(spec, "de", Config(pocket_revision=rev))
    assert _model_marker(engines.spec_for("f5"), "de", Config()) == "model-f5-de"


def test_registry_and_factory():
    spec = engines.spec_for("pocket")
    assert spec.group == engines.CLASSIC_GROUP and spec.per_language and not spec.gpu_backends
    assert _default_factory(Config(tts_backend="pocket")).__name__ == "make_pocket"


def test_pocket_is_locked_to_the_startup_language():
    """One model per process, like F5: a per-request language is ignored."""
    from src.server import _request_languages

    assert _request_languages(Config(tts_backend="pocket")) is None


def test_options_coerce_from_env():
    config = load_config(argv=[], env={"F5W_POCKET_DECODE_STEPS": "8", "F5W_POCKET_THREADS": "4", "F5W_POCKET_LAYERS": "6"})
    assert config.pocket_decode_steps == 8 and config.pocket_threads == 4 and config.pocket_layers == 6


def test_the_source_is_pinned_to_a_commit():
    """A moving branch would change what users get without a line changing here."""
    rev = Config().pocket_revision
    assert len(rev) == 40 and all(c in "0123456789abcdef" for c in rev)


def test_model_dir_is_per_revision(tmp_path):
    a = _model_dir(Config(models_dir=str(tmp_path), pocket_revision="a" * 40))
    b = _model_dir(Config(models_dir=str(tmp_path), pocket_revision="b" * 40))
    assert a != b and a.parent == Path(tmp_path) / "pocket"


@pytest.mark.parametrize("configured, cores, expected", [(0, 32, 4), (0, 2, 2), (0, 1, 1), (8, 32, 8)])
def test_threads_are_always_explicit_and_modest(configured, cores, expected):
    """A launched server was seen starting torch with ONE thread (rtf 0.95)."""
    assert cpu_threads(configured, cores) == expected
