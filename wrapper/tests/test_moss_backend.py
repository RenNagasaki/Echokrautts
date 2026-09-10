"""The MOSS-TTS-Nano backend.

Driven against a fake ``moss_tts_nano_runtime`` module: the worker imports it
lazily, so the unit suite needs neither the runtime nor torch. What is pinned
here is everything the spike measured or discovered the hard way — the audio
contract conversion (MOSS emits 48 kHz stereo, the wrapper promises 24 kHz
mono), the device override that lets the GPU stay with the game, and the
cleanup of the files the runtime writes on every single request.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from src import moss_backend
from src.config import Config


class _FakeService:
    """Stands in for ``NanoTTSService``; yields the event shape MOSS yields."""

    def __init__(self, chunks=3, sample_rate=48000, channels=2, written=None):
        self.calls = []
        self.preloaded = False
        self._chunks = chunks
        self._sr = sample_rate
        self._ch = channels
        self.written = written

    def preload(self, **kwargs):
        self.preloaded = True
        return {}

    def synthesize_stream(self, **kwargs):
        self.calls.append(kwargs)
        for i in range(self._chunks):
            frame = np.full((self._ch, 480), 0.25 * (i + 1), dtype=np.float32)
            yield {
                "type": "audio",
                "waveform": frame,
                "sample_rate": self._sr,
                "chunk_index": i,
                "emitted_audio_seconds": 0.01 * (i + 1),
            }
        yield {"type": "result", "audio_path": str(self.written) if self.written else "", "sample_rate": self._sr}


@pytest.fixture
def fake_moss(monkeypatch, tmp_path):
    """Install a fake runtime module and hand back the service it creates."""
    holder = {}

    def factory(**kwargs):
        holder["init"] = kwargs
        svc = _FakeService(written=holder.get("written"))
        holder["service"] = svc
        return svc

    module = types.ModuleType("moss_tts_nano_runtime")
    module.NanoTTSService = factory
    monkeypatch.setitem(sys.modules, "moss_tts_nano_runtime", module)

    torch_stub = types.ModuleType("torch")
    torch_stub.float32 = "float32"

    class _T:
        def __init__(self, a):
            self.a = np.asarray(a, dtype=np.float32)

        def dim(self):
            return self.a.ndim

        def detach(self):
            return self

        def to(self, *a, **k):
            return self

        @property
        def shape(self):
            return self.a.shape

        def unsqueeze(self, _d):
            return _T(self.a[None, :])

        def mean(self, dim=0, keepdim=False):
            return _T(self.a.mean(axis=dim, keepdims=keepdim))

        def squeeze(self, _d):
            return _T(self.a.squeeze(_d))

        def numpy(self):
            return self.a

        # The seam-free resampling carries a tail across chunks, so the stub has
        # to support the handful of tensor operations that needs.
        def numel(self):
            return self.a.size

        def clone(self):
            return _T(self.a.copy())

        def __getitem__(self, item):
            return _T(self.a[item])

    torch_stub.as_tensor = lambda a, **k: _T(a)
    torch_stub.cat = lambda parts: _T(np.concatenate([p.a for p in parts]))
    torch_stub.from_numpy = lambda a: _T(a)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    torchaudio = types.ModuleType("torchaudio")
    holder["resampled"] = []

    def resample(t, orig, target):
        holder["resampled"].append((orig, target))
        return t

    torchaudio.functional = types.SimpleNamespace(resample=resample)
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio)
    return holder


def _lay_down_weights(config) -> None:
    """Create what the worker now insists on finding.

    The worker refuses to start without the model files, because letting
    transformers reinterpret a missing path as a repo id produced an error that
    named neither the cause nor the fix. Tests therefore have to put something
    there — which is the point: the check is real, not decorative.
    """
    for directory in moss_backend._model_dirs(config):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text("{}", encoding="utf-8")


def _worker(config=None, device="cpu", tmp_path=None):
    cfg = config or Config(models_dir=str(tmp_path) if tmp_path else None)
    _lay_down_weights(cfg)
    return moss_backend.MossWorker(cfg, device)


# --------------------------------------------------------------------- filter

def test_all_four_ffxiv_client_languages_are_supported():
    for code in ("en", "ja", "de", "fr"):
        assert code in moss_backend.MOSS_LANGUAGES, code


def test_streaming_is_declared_on(fake_moss, tmp_path):
    """The reason this engine was chosen over Qwen: it really streams."""
    worker = _worker(tmp_path=tmp_path)
    assert hasattr(worker, "infer_stream")
    assert moss_backend.MossWorker.supports_streaming is True
    assert worker.supports_streaming is True


def test_moss_stream_can_be_turned_off(fake_moss, tmp_path):
    """The escape hatch for a consumer that cannot bridge a slow producer.

    MOSS generates slower than real time, so anything that starts playing on the
    first piece has to make up the shortfall itself. Bridging it belongs to the
    consumer, which is the side that knows about playback; this flag is for one
    that cannot, and pays the full generation time before the first sound.
    """
    cfg = Config(models_dir=str(tmp_path), moss_stream=False)
    assert _worker(config=cfg).supports_streaming is False


def test_no_transcript_is_required(fake_moss, tmp_path):
    """Clones from audio alone, like XTTS — the sample service needs no sidecar."""
    from pathlib import Path

    assert _worker(tmp_path=tmp_path).transcribe(Path("x.wav")) == ""


# ------------------------------------------------------------- audio contract

def test_stereo_48k_becomes_mono_24k(fake_moss, tmp_path):
    """MOSS emits 48 kHz stereo; the wrapper promises 24 kHz mono s16.

    Passing stereo through would be heard as garbage, and passing 48 kHz off as
    24 kHz would play at half speed — both silent failures, not crashes.
    """
    worker = _worker(tmp_path=tmp_path)
    chunks = list(worker.infer_stream("r.wav", "", "Text.", 1.0))
    assert chunks, "no audio produced"
    for chunk in chunks:
        assert chunk.ndim == 1, "must be mono"
        assert chunk.dtype == np.float32
    assert fake_moss["resampled"] == [(48000, 24000)] * len(chunks)


def test_matching_rate_is_not_resampled(fake_moss, tmp_path, monkeypatch):
    """A no-op conversion on every chunk of every request is worth avoiding."""
    worker = _worker(tmp_path=tmp_path)
    worker.sample_rate = 48000
    list(worker.infer_stream("r.wav", "", "Text.", 1.0))
    assert fake_moss["resampled"] == []


def test_infer_concatenates_the_same_stream(fake_moss, tmp_path):
    """One code path for both: streamed and buffered audio cannot drift apart."""
    worker = _worker(tmp_path=tmp_path)
    streamed = np.concatenate(list(worker.infer_stream("r.wav", "", "Text.", 1.0)))
    oneshot = worker.infer("r.wav", "", "Text.", nfe_step=32, speed=1.0)
    assert oneshot.dtype == np.float32
    assert oneshot.shape == streamed.shape


# -------------------------------------------------------------------- device

def test_config_can_pin_moss_to_the_cpu(fake_moss, tmp_path):
    """The whole point of this engine: measured, CPU matches the GPU, so the
    graphics card can stay with the game."""
    cfg = Config(models_dir=str(tmp_path), moss_device="cpu")
    _lay_down_weights(cfg)
    moss_backend.MossWorker(cfg, "cuda")
    assert fake_moss["init"]["device"] == "cpu"


def test_auto_follows_the_engine_device(fake_moss, tmp_path):
    cfg = Config(models_dir=str(tmp_path), moss_device="auto")
    _lay_down_weights(cfg)
    moss_backend.MossWorker(cfg, "cuda")
    assert fake_moss["init"]["device"] == "cuda"


@pytest.mark.parametrize("device", ["dml", "xpu"])
def test_unsupported_devices_map_to_cpu(fake_moss, tmp_path, device):
    cfg = Config(models_dir=str(tmp_path), moss_device="auto")
    _lay_down_weights(cfg)
    moss_backend.MossWorker(cfg, device)
    assert fake_moss["init"]["device"] == "cpu"


# ------------------------------------------------------------------- hygiene

def test_generated_files_are_deleted_after_each_request(fake_moss, tmp_path):
    """The runtime writes every clip to disk and offers no way to stop it.

    On a server speaking thousands of game lines that is unbounded growth, so
    the worker removes what it caused.
    """
    leftover = tmp_path / "generated.wav"
    leftover.write_bytes(b"RIFF")
    fake_moss["written"] = leftover

    worker = _worker(tmp_path=tmp_path)
    worker._service.written = leftover
    list(worker.infer_stream("r.wav", "", "Text.", 1.0))

    assert not leftover.exists(), "the runtime's output file was left behind"


def test_cleanup_failure_does_not_fail_the_request(fake_moss, tmp_path, monkeypatch):
    """A leftover file is a log line; audio that was already produced is not
    worth throwing away over it."""
    worker = _worker(tmp_path=tmp_path)

    def boom(*_a, **_k):
        raise OSError("locked by another process")

    monkeypatch.setattr(moss_backend.Path, "unlink", boom)
    worker._service.written = tmp_path / "x.wav"
    chunks = list(worker.infer_stream("r.wav", "", "Text.", 1.0))
    assert chunks, "audio must survive a failed cleanup"


def test_model_is_preloaded_at_construction(fake_moss, tmp_path):
    """Loading on the first request would put a cold start inside a user's
    first line of dialogue."""
    assert _worker(tmp_path=tmp_path)._service.preloaded is True


def test_weights_go_to_flat_directories_not_the_hf_cache(tmp_path):
    """The HF cache symlinks snapshots to blobs, and Windows forbids that to a
    normal account — it killed the first download attempt of another engine."""
    cfg = Config(models_dir=str(tmp_path))
    checkpoint, tokenizer = moss_backend._model_dirs(cfg)
    assert checkpoint.parent == cfg.models_path
    assert tokenizer.parent == cfg.models_path
    assert checkpoint != tokenizer


def test_the_default_keeps_the_gpu_free(fake_moss, tmp_path):
    """The default must be CPU, not "auto".

    MOSS was added so a plugin can speak while a game renders, and it is the one
    engine where the GPU buys nothing (measured: CPU 1.25 vs GPU 1.22 rtf).
    Defaulting to "auto" would silently take the graphics card — the exact thing
    this backend exists to avoid.
    """
    assert Config().moss_device == "cpu"
    cfg = Config(models_dir=str(tmp_path))
    _lay_down_weights(cfg)
    moss_backend.MossWorker(cfg, "cuda")
    assert fake_moss["init"]["device"] == "cpu"


def test_worker_reports_the_device_it_actually_uses(fake_moss, tmp_path):
    """`/health` and the `ready` event read `worker.device`.

    Reporting the requested device while the model sits elsewhere is a lie told
    at exactly the place a user looks to check — it was reported live: "MOSS
    geladen auf cpu" one line above `"device": "cuda"`.
    """
    cfg = Config(models_dir=str(tmp_path), moss_device="cpu")
    _lay_down_weights(cfg)
    worker = moss_backend.MossWorker(cfg, "cuda")
    assert worker.device == "cpu"


def test_missing_weights_say_so_instead_of_becoming_a_repo_id(fake_moss, tmp_path):
    """transformers treats a path that is not a directory as a HuggingFace repo
    id and then fails its name validation — an error that names neither the
    missing weights nor what to do. Seen live after an upgrade installed the
    engine but skipped its download."""
    config = Config(models_dir=str(tmp_path))  # nothing downloaded
    with pytest.raises(RuntimeError) as err:
        moss_backend.MossWorker(config, "cpu")

    message = str(err.value)
    assert "moss_tts_nano" in message, "must name the directory it looked in"
    assert "src.moss_backend" in message, "must say how to fix it"


# ---------------------------------------------------------------- seam-free audio
#
# A user heard crackling. Measured on the real model: resampling each streamed
# chunk on its own left the largest sample-to-sample jump at the seams at 1.48x
# the signal's own typical step, across ~60 seams per sentence. Carrying the tail
# of the previous chunk into the next resample brought that to 0.71x — exactly
# the value of resampling the whole thing at once.

def test_the_resampler_gets_context_across_chunk_boundaries(fake_moss, tmp_path):
    """Without it the filter starts cold on every chunk, which is audible."""
    worker = _worker(tmp_path=tmp_path)
    lengths = []

    real = moss_backend.torchaudio if hasattr(moss_backend, "torchaudio") else None
    import sys as _sys

    def spy(tensor, orig, target):
        lengths.append(tensor.a.shape[-1])
        return tensor

    _sys.modules["torchaudio"].functional.resample = spy
    list(worker.infer_stream("r.wav", "", "Text.", 1.0))

    assert len(lengths) >= 2, "needs several chunks to have a seam at all"
    assert lengths[0] < lengths[1], (
        "later chunks must be resampled WITH the previous tail prepended; "
        f"got {lengths[:3]}"
    )
    assert lengths[1] - lengths[0] == moss_backend.MossWorker.RESAMPLE_OVERLAP


def test_the_carried_tail_does_not_leak_between_requests(fake_moss, tmp_path):
    """Two requests are two signals. Splicing the end of one sentence onto the
    start of the next would be a defect that only shows up in production."""
    worker = _worker(tmp_path=tmp_path)
    list(worker.infer_stream("r.wav", "", "Erster Satz.", 1.0))
    assert worker._resampler.tail is not None, "a tail is kept within a request"

    lengths = []
    import sys as _sys

    _sys.modules["torchaudio"].functional.resample = lambda t, o, n: (
        lengths.append(t.a.shape[-1]) or t
    )
    list(worker.infer_stream("r.wav", "", "Zweiter Satz.", 1.0))
    assert lengths[0] < lengths[1], "the second request must start without a tail"


def test_no_overlap_work_when_no_resampling_is_needed(fake_moss, tmp_path):
    """If the engine already emits the contract rate there is nothing to smooth."""
    worker = _worker(tmp_path=tmp_path)
    worker.sample_rate = 48000
    list(worker.infer_stream("r.wav", "", "Text.", 1.0))
    assert worker._resampler.tail is None
