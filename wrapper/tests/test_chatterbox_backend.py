"""Tests for the Chatterbox Multilingual backend, without torch/chatterbox.

The heavy imports (torch, chatterbox.mtl_tts) live inside ChatterboxWorker.
__init__, so the module imports and its plumbing is testable with fakes — same
approach as test_xtts_backend.py.
"""

import io
import sys
import types

import numpy as np
import pytest

from src import chatterbox_backend as cb
from src import ndjson
from src.config import Config


def test_module_imports_without_heavy_deps():
    assert cb.DEFAULT_SAMPLE_RATE == 24000
    assert cb.CHATTERBOX_REPO == "ResembleAI/chatterbox"
    # The engine is one-shot: the released package exposes no streaming API, so
    # the engine must take its _one_shot_chunk path (like F5), not _stream_chunk.
    assert cb.ChatterboxWorker.supports_streaming is False


def test_language_set_matches_upstream():
    # 23 languages, and Chinese is "zh" here — NOT XTTS's "zh-cn". Getting that
    # wrong would 400 a perfectly valid request.
    assert len(cb.CHATTERBOX_LANGUAGES) == 23
    assert {"de", "en", "fr", "ja", "zh"} <= cb.CHATTERBOX_LANGUAGES
    assert "zh-cn" not in cb.CHATTERBOX_LANGUAGES


# ------------------------------------------------------------- model location
def test_use_models_cache_points_hf_at_models_dir(tmp_path, monkeypatch):
    # The library calls snapshot_download without cache_dir, so the environment
    # is the only lever — without it the weights land outside the container's
    # models volume and are re-downloaded on every start.
    cfg = Config(models_dir=str(tmp_path / "models"))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)

    cb._use_models_cache(cfg)

    import os

    assert os.environ["HF_HUB_CACHE"] == str(cfg.models_path)
    assert os.environ["HF_HOME"] == str(cfg.models_path)
    assert cfg.models_path.exists()


def test_use_models_cache_overrides_a_global_hf_cache(tmp_path, monkeypatch):
    # Regression (found live): with a machine-wide HF_HUB_CACHE, a setdefault
    # here left it in place — download_model filled models/ while the worker
    # then re-downloaded the same 2 GB into the global cache. models_dir is the
    # user's setting for where weights live, so it wins.
    cfg = Config(models_dir=str(tmp_path / "models"))
    monkeypatch.setenv("HF_HUB_CACHE", "G:/cache/huggingface/hub")

    cb._use_models_cache(cfg)

    import os

    assert os.environ["HF_HUB_CACHE"] == str(cfg.models_path)


def _write_custom_model(cfg, names):
    custom = cfg.custom_model_path
    custom.mkdir(parents=True, exist_ok=True)
    for name in names:
        (custom / name).write_bytes(b"x")
    return custom


def test_custom_model_dir_needs_the_complete_file_set(tmp_path):
    cfg = Config(models_dir=str(tmp_path / "models"))
    custom = _write_custom_model(cfg, cb.CUSTOM_MODEL_FILES)

    assert cb._resolve_custom_model_dir(cfg) == str(custom)


def test_custom_model_dir_rejects_a_partial_folder(tmp_path):
    # A half-populated folder would be detected as "custom model" and then fail
    # deep inside from_local — better to fall back to the pretrained download.
    cfg = Config(models_dir=str(tmp_path / "models"))
    _write_custom_model(cfg, cb.CUSTOM_MODEL_FILES[:-1])

    assert cb._resolve_custom_model_dir(cfg) is None


def test_custom_model_dir_ignores_other_engines_models(tmp_path):
    # The three backends share models/echokraut_custom/; their formats must stay
    # disjoint. An XTTS model dir is not a Chatterbox one.
    cfg = Config(models_dir=str(tmp_path / "models"))
    _write_custom_model(cfg, ["config.json", "model.pth"])

    assert cb._resolve_custom_model_dir(cfg) is None


def test_stdout_is_kept_out_of_the_ndjson_stream():
    # Loading prints ("loaded PerthNet (Implicit) at step 250,000"), and stdout
    # is the protocol channel — anything but NDJSON on it is corruption.
    out, err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        with cb._stdout_to_stderr():
            print("loaded PerthNet (Implicit) at step 250,000")
    finally:
        sys.stdout, sys.stderr = old_out, old_err

    assert out.getvalue() == ""
    assert "PerthNet" in err.getvalue()


# -------------------------------------------------------------------- worker
class _FakeTensor:
    """Stands in for the (1, N) torch tensor generate() returns."""

    def __init__(self, data):
        self._data = np.asarray(data, dtype=np.float64).reshape(1, -1)

    def detach(self):
        return self

    def to(self, _device):
        return self

    def numpy(self):
        return self._data


class _FakeModel:
    def __init__(self):
        self.conds = None
        self.prepared: list[tuple] = []
        self.calls: list[dict] = []

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        self.prepared.append((wav_fpath, exaggeration))
        self.conds = f"conds:{wav_fpath}"

    def generate(self, text, language_id, **kw):
        self.calls.append({"text": text, "language_id": language_id,
                           "conds": self.conds, **kw})
        return _FakeTensor([0.5, -0.5, 0.25])


def _worker(model=None, config=None):
    cfg = config or Config(language="de")
    worker = object.__new__(cb.ChatterboxWorker)
    worker._model = model or _FakeModel()
    worker._lang = cfg.language
    worker._exaggeration = cfg.chatterbox_exaggeration
    worker._cfg_weight = cfg.chatterbox_cfg_weight
    worker._temperature = cfg.chatterbox_temperature
    worker._conds_cache = {}
    worker.device = "cpu"
    worker.sample_rate = cb.DEFAULT_SAMPLE_RATE
    return worker


def test_conditioning_is_computed_once_per_sample_and_reattached():
    # The engine calls infer() repeatedly with the same sample; redoing the
    # reference embedding on every call is the cost this cache exists to avoid.
    model = _FakeModel()
    worker = _worker(model)

    worker._conditioning("voice.wav")
    worker._conditioning("voice.wav")

    assert model.prepared == [("voice.wav", 0.5)]
    assert model.conds == "conds:voice.wav"


def test_conditioning_switches_back_to_a_cached_sample():
    # Two voices alternating must not silently keep the last one's conditioning:
    # `conds` is single-slot state on the model, so the cached object has to be
    # re-attached, not just kept in the dict.
    model = _FakeModel()
    worker = _worker(model)

    worker._conditioning("a.wav")
    worker._conditioning("b.wav")
    worker._conditioning("a.wav")

    assert [p[0] for p in model.prepared] == ["a.wav", "b.wav"]
    assert model.conds == "conds:a.wav"


def test_infer_returns_flat_float32_and_passes_the_knobs():
    model = _FakeModel()
    cfg = Config(language="de", chatterbox_exaggeration=0.7,
                 chatterbox_cfg_weight=0.3, chatterbox_temperature=0.9)
    worker = _worker(model, cfg)

    out = worker.infer("voice.wav", "", "Hallo.", nfe_step=32, speed=1.0)

    assert out.dtype == np.float32 and out.ndim == 1
    assert np.allclose(out, [0.5, -0.5, 0.25])
    call = model.calls[0]
    assert call["language_id"] == "de"  # startup language
    assert call["exaggeration"] == 0.7
    assert call["cfg_weight"] == 0.3
    assert call["temperature"] == 0.9
    # Conditioning is installed before generating, and NOT passed as
    # audio_prompt_path (which would recompute it on every call).
    assert call["conds"] == "conds:voice.wav"
    assert "audio_prompt_path" not in call


def test_infer_honors_the_per_request_language_case_insensitively():
    model = _FakeModel()
    worker = _worker(model)

    worker.infer("voice.wav", "", "Hello.", nfe_step=32, speed=1.0, language="EN")

    assert model.calls[0]["language_id"] == "en"


def test_infer_accepts_a_plain_array_result():
    # Defensive: a chatterbox version returning a numpy array instead of a
    # tensor must not break the PCM conversion.
    model = _FakeModel()
    model.generate = lambda text, language_id, **kw: np.array([[0.1, 0.2]])
    worker = _worker(model)

    out = worker.infer("voice.wav", "", "x", nfe_step=32, speed=1.0)

    assert out.dtype == np.float32 and out.tolist() == pytest.approx([0.1, 0.2])


def test_speed_is_reported_as_unsupported_once(capsys):
    # Chatterbox's generate() has no speed parameter. Silently ignoring the
    # request field would look like a bug in the wrapper; log_once keeps it from
    # repeating per request.
    ndjson.reset_log_once()
    worker = _worker()

    worker.infer("voice.wav", "", "a", nfe_step=32, speed=1.5)
    worker.infer("voice.wav", "", "b", nfe_step=32, speed=1.5)
    worker.infer("voice.wav", "", "c", nfe_step=32, speed=1.0)

    lines = [l for l in capsys.readouterr().out.splitlines() if "speed" in l]
    assert len(lines) == 1
    ndjson.reset_log_once()


def test_transcribe_is_a_noop():
    # Chatterbox clones from audio alone; there is no ref-text path to feed.
    from pathlib import Path

    assert _worker().transcribe(Path("voice.wav")) == ""


# --------------------------------------------------------- leaked attn hooks
def _hooked_model(per_layer):
    """Fake model shaped like model.t3.tfmr.layers[i].self_attn._forward_hooks."""

    def spy(module, inp, out):  # name must match ANALYZER_HOOK_NAME
        return None

    spy.__name__ = cb.ANALYZER_HOOK_NAME

    def foreign(module, inp, out):
        return None

    layers = []
    for _ in range(3):
        hooks = {}
        for i in range(per_layer):
            hooks[len(hooks)] = spy
        hooks[99] = foreign  # something else's hook must survive
        layers.append(types.SimpleNamespace(self_attn=types.SimpleNamespace(_forward_hooks=hooks)))
    return types.SimpleNamespace(t3=types.SimpleNamespace(tfmr=types.SimpleNamespace(layers=layers))), foreign


def test_stale_analyzer_hooks_are_dropped():
    # Upstream builds a fresh AlignmentStreamAnalyzer per request and never
    # removes the previous one's hooks; each copies the attention matrix to the
    # host on every decode step. Measured: 153 leaked hooks (~50 requests) took
    # the same sentence from rtf 1.17 to 1.29.
    model, foreign = _hooked_model(per_layer=2)

    removed = cb._drop_stale_analyzer_hooks(model)

    assert removed == 6  # 3 layers x 2
    for layer in model.t3.tfmr.layers:
        assert list(layer.self_attn._forward_hooks.values()) == [foreign]


def test_dropping_hooks_survives_a_changed_library_layout():
    # It reaches into library internals, so a renamed attribute must mean
    # "nothing to clean", never a failed request.
    assert cb._drop_stale_analyzer_hooks(types.SimpleNamespace()) == 0
    assert cb._drop_stale_analyzer_hooks(
        types.SimpleNamespace(t3=types.SimpleNamespace(tfmr=None))
    ) == 0
    assert cb._drop_stale_analyzer_hooks(
        types.SimpleNamespace(t3=types.SimpleNamespace(
            tfmr=types.SimpleNamespace(layers=[types.SimpleNamespace()])))
    ) == 0


def test_infer_cleans_hooks_before_generating():
    # Order matters: clean first, then let the fresh analyzer register — so the
    # steady state is one request's worth of hooks, not zero and not N.
    model, _foreign = _hooked_model(per_layer=1)
    worker = _worker(_FakeModel())
    worker._model.t3 = model.t3

    worker.infer("voice.wav", "", "x", nfe_step=32, speed=1.0)

    assert all(
        cb.ANALYZER_HOOK_NAME not in [getattr(f, "__name__", "") for f in l.self_attn._forward_hooks.values()]
        for l in model.t3.tfmr.layers
    )


# ------------------------------------------------------------------ download
def test_download_model_fetches_the_file_subset(tmp_path, monkeypatch):
    cfg = Config(models_dir=str(tmp_path / "models"))
    captured = {}

    def fake_snapshot_download(**kw):
        captured.update(kw)
        return str(tmp_path / "snapshot")

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    cb.download_model(cfg)

    assert captured["repo_id"] == cb.CHATTERBOX_REPO
    assert captured["cache_dir"] == str(cfg.models_path)
    # Only the multilingual subset — the repo also holds the English-only model.
    assert "t3_mtl23ls_v2.safetensors" in captured["allow_patterns"]
    assert cfg.models_path.exists()
