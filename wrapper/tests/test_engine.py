import asyncio
import re

import numpy as np
import pytest

from conftest import FakeWorker, make_engine
from src.config import Config
from src.engine import (
    KNOWN_BACKENDS,
    Engine,
    InferenceError,
    QueueFull,
    TtsParams,
    _default_factory,
    float_to_pcm16,
)
from src.gpu_detect import Detection
from src.jobs import CANCELLED, DONE, ERROR, JobRegistry


def _engine_on(config: Config, device: str) -> Engine:
    det = Detection(backend=device, device=device, torch_index_url="x", max_workers_hint=1)
    return Engine(config, det, JobRegistry(), worker_factory=lambda i, d: FakeWorker())


def test_health_reports_effective_xtts_fp16():
    # Enabled + xtts + CUDA → effective.
    on = _engine_on(Config(tts_backend="xtts", xtts_fp16=True), "cuda")
    assert on.health()["xtts_fp16"] is True
    # Same flag but CPU device → not effective.
    cpu = _engine_on(Config(tts_backend="xtts", xtts_fp16=True), "cpu")
    assert cpu.health()["xtts_fp16"] is False
    # f5 backend ignores the flag entirely, even on CUDA.
    f5 = _engine_on(Config(tts_backend="f5", xtts_fp16=True), "cuda")
    assert f5.health()["xtts_fp16"] is False


def test_default_factory_defaults_to_f5():
    # The default backend must remain F5; we only assert selection here (calling
    # the closure would need f5-tts installed).
    cfg = Config()
    assert cfg.tts_backend == "f5"
    assert callable(_default_factory(cfg))


def test_default_factory_selects_xtts(monkeypatch):
    from src import xtts_backend

    made = {}

    class FakeXTTS:
        def __init__(self, config, device):
            made["config"] = config
            made["device"] = device

    monkeypatch.setattr(xtts_backend, "XTTSWorker", FakeXTTS)
    factory = _default_factory(Config(tts_backend="xtts"))
    worker = factory(0, "cpu")
    assert isinstance(worker, FakeXTTS)
    assert made["device"] == "cpu"


def test_moss_uses_the_onnx_runtime_when_it_is_available(monkeypatch):
    """ONNX is the default: same weights, ~2.5x faster on the CPU."""
    from src import moss_onnx_backend

    class FakeOnnx:
        def __init__(self, config, device):
            self.device = device

    monkeypatch.setattr(moss_onnx_backend, "is_available", lambda cfg: (True, ""))
    monkeypatch.setattr(moss_onnx_backend, "MossOnnxWorker", FakeOnnx)
    worker = _default_factory(Config(tts_backend="moss", moss_runtime="onnx"))(0, "cpu")
    assert isinstance(worker, FakeOnnx)


def test_moss_falls_back_to_pytorch_and_says_so(monkeypatch):
    """A missing ONNX install must not cost the user the engine.

    Unlike an unknown BACKEND — which is silently the WRONG voice and therefore
    fails hard — this substitution is the same model at a different speed, so it
    degrades. It is logged as a warning because a user who wonders why MOSS got
    slow deserves to find the answer in the log rather than guess.
    """
    from src import moss_backend, moss_onnx_backend, ndjson

    said = []
    monkeypatch.setattr(ndjson, "log_once", lambda msg, level="info": said.append((level, msg)))
    monkeypatch.setattr(moss_onnx_backend, "is_available",
                        lambda cfg: (False, "onnxruntime ist nicht installiert"))

    class FakeMoss:
        def __init__(self, config, device):
            self.device = device

    monkeypatch.setattr(moss_backend, "MossWorker", FakeMoss)
    worker = _default_factory(Config(tts_backend="moss", moss_runtime="onnx"))(0, "cpu")
    assert isinstance(worker, FakeMoss)
    assert any(level == "warning" and "onnxruntime" in msg for level, msg in said)


def test_moss_runtime_pytorch_never_touches_onnx(monkeypatch):
    """Choosing the old runtime must not import or probe the new one."""
    from src import moss_backend, moss_onnx_backend

    def explode(cfg):
        raise AssertionError("is_available must not be called for moss_runtime=pytorch")

    monkeypatch.setattr(moss_onnx_backend, "is_available", explode)

    class FakeMoss:
        def __init__(self, config, device):
            self.device = device

    monkeypatch.setattr(moss_backend, "MossWorker", FakeMoss)
    worker = _default_factory(Config(tts_backend="moss", moss_runtime="pytorch"))(0, "cpu")
    assert isinstance(worker, FakeMoss)


def test_float_to_pcm16_known_values():
    wav = np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    pcm = float_to_pcm16(wav)
    samples = np.frombuffer(pcm, dtype="<i2")
    # 2.0/-2.0 clipped to 1.0/-1.0 → 32767 / -32767.
    assert list(samples) == [0, 32767, -32767, 32767, -32767]


@pytest.mark.asyncio
async def test_stream_produces_pcm_and_completes(config):
    engine = make_engine(config)
    await engine.start()
    job = engine._jobs.create()
    params = TtsParams(sample="anna_de.wav", text="Eins. Zwei. Drei.")
    path = engine.samples.resolve_path("anna_de.wav")
    engine.admit()

    chunks = [c async for c in engine.stream(job, params, path)]
    assert len(chunks) >= 1
    assert all(isinstance(c, (bytes, bytearray)) for c in chunks)
    assert sum(len(c) for c in chunks) > 0
    assert job.state == DONE
    assert job.sentences_done == job.sentences_total


@pytest.mark.asyncio
async def test_stream_emits_request_timing_logs(config, monkeypatch):
    # A completed request logs a start line and a done line with generation
    # time, audio length, and the real-time factor (efficiency).
    from src import engine as engine_mod

    logs: list[str] = []
    monkeypatch.setattr(engine_mod.ndjson, "log", lambda msg, level="info": logs.append(msg))

    config.max_chars_per_chunk = 5  # force "Eins." / "Zwei." into 2 chunks
    engine = make_engine(config)
    await engine.start()
    job = engine._jobs.create()
    params = TtsParams(sample="anna_de.wav", text="Eins. Zwei.")
    path = engine.samples.resolve_path("anna_de.wav")
    engine.admit()

    _ = [c async for c in engine.stream(job, params, path)]

    start = next(m for m in logs if m.startswith("tts request start:"))
    done = next(m for m in logs if m.startswith("tts request done:"))
    assert f"job={job.job_id}" in start and "sample=anna_de.wav" in start
    assert "generated=" in done and "audio=" in done and "rtf=" in done
    # Two sentences × 50 ms of fake silence → ~0.10 s of audio reported.
    assert "audio=0.10s" in done
    # One-shot path: one PCM part per sentence, and the first-audio timestamp is
    # present (it is a wall-clock measurement, so only its shape is asserted).
    assert "parts=2" in done
    assert re.search(r"first=\d+\.\d\ds", done)


@pytest.mark.asyncio
async def test_timing_log_reports_first_audio_and_parts_when_streaming(config, monkeypatch):
    # The streaming metric: a token-streaming worker delivers many parts, so
    # `parts` >> chunks. Together with `first` this makes it observable that
    # audio left the engine before the request finished — i.e. a consumer that
    # only starts playing at the end is buffering on its own side.
    from src import engine as engine_mod

    logs: list[str] = []
    monkeypatch.setattr(engine_mod.ndjson, "log", lambda msg, level="info": logs.append(msg))

    config.max_chars_per_chunk = 5  # force "Eins." / "Zwei." into 2 chunks
    engine = make_engine(config, streaming=True)
    await engine.start()
    job = engine._jobs.create()
    params = TtsParams(sample="anna_de.wav", text="Eins. Zwei.")
    path = engine.samples.resolve_path("anna_de.wav")
    engine.admit()

    _ = [c async for c in engine.stream(job, params, path)]

    done = next(m for m in logs if m.startswith("tts request done:"))
    # 2 chunks × FakeStreamWorker.PARTS_PER_CHUNK parts.
    assert "parts=6" in done
    assert re.search(r"first=\d+\.\d\ds", done)


@pytest.mark.asyncio
async def test_timing_log_reports_no_first_audio_for_empty_text(config, monkeypatch):
    # Empty text chunks to nothing → no PCM ever yielded. `first` must say so
    # rather than claiming a bogus 0.00s.
    from src import engine as engine_mod

    logs: list[str] = []
    monkeypatch.setattr(engine_mod.ndjson, "log", lambda msg, level="info": logs.append(msg))

    engine = make_engine(config)
    await engine.start()
    job = engine._jobs.create()
    params = TtsParams(sample="anna_de.wav", text="   ")
    path = engine.samples.resolve_path("anna_de.wav")
    engine.admit()

    _ = [c async for c in engine.stream(job, params, path)]

    done = next(m for m in logs if m.startswith("tts request done:"))
    assert "first=n/a" in done and "parts=0" in done


@pytest.mark.asyncio
async def test_cancel_before_first_chunk(config):
    engine = make_engine(config)
    await engine.start()
    job = engine._jobs.create()
    job.cancel_event.set()
    params = TtsParams(sample="anna_de.wav", text="Eins. Zwei.")
    path = engine.samples.resolve_path("anna_de.wav")
    engine.admit()

    chunks = [c async for c in engine.stream(job, params, path)]
    assert chunks == []
    assert job.state == CANCELLED


@pytest.mark.asyncio
async def test_queue_full(config):
    config.max_queue = 0
    engine = make_engine(config, workers=1)
    await engine.start()
    engine.admit()  # fills the only slot
    with pytest.raises(QueueFull):
        engine.admit()


@pytest.mark.asyncio
async def test_token_streaming_yields_parts_per_sentence(config):
    # A streaming worker (XTTS-like) emits several PCM parts per sentence,
    # finer-grained than the one-shot path — but job progress still advances
    # per sentence.
    config.max_chars_per_chunk = 5  # force "Eins." / "Zwei." into 2 chunks
    engine = make_engine(config, streaming=True)
    await engine.start()
    job = engine._jobs.create()
    params = TtsParams(sample="anna_de.wav", text="Eins. Zwei.")  # 2 chunks
    path = engine.samples.resolve_path("anna_de.wav")
    engine.admit()

    chunks = [c async for c in engine.stream(job, params, path)]
    # 2 chunks × 3 parts each — streaming granularity is finer than chunks.
    assert len(chunks) == 2 * 3
    assert all(isinstance(c, (bytes, bytearray)) and len(c) > 0 for c in chunks)
    assert job.state == DONE
    assert job.sentences_done == job.sentences_total == 2


@pytest.mark.asyncio
async def test_language_forwarded_to_worker(config):
    # The per-request language reaches the worker on both the one-shot and the
    # streaming path (XTTS uses it to pick the target language, no reload).
    for streaming in (False, True):
        engine = make_engine(config, streaming=streaming)
        await engine.start()
        job = engine._jobs.create()
        params = TtsParams(sample="anna_de.wav", text="Hallo.", language="fr")
        path = engine.samples.resolve_path("anna_de.wav")
        engine.admit()

        _ = [c async for c in engine.stream(job, params, path)]
        assert engine._workers[0].languages == ["fr"]


@pytest.mark.asyncio
async def test_streaming_cancel_between_parts(config):
    engine = make_engine(config, streaming=True)
    await engine.start()
    job = engine._jobs.create()
    job.cancel_event.set()  # cancelled before the first part
    params = TtsParams(sample="anna_de.wav", text="Eins. Zwei.")
    path = engine.samples.resolve_path("anna_de.wav")
    engine.admit()

    chunks = [c async for c in engine.stream(job, params, path)]
    assert chunks == []
    assert job.state == CANCELLED


@pytest.mark.asyncio
async def test_streaming_inference_error_rebuilds_worker(config):
    engine = make_engine(config, fail=True, streaming=True)
    await engine.start()
    job = engine._jobs.create()
    params = TtsParams(sample="anna_de.wav", text="Hallo.")
    path = engine.samples.resolve_path("anna_de.wav")
    engine.admit()

    with pytest.raises(InferenceError):
        async for _ in engine.stream(job, params, path):
            pass
    assert job.state == ERROR
    assert engine.queue_depth == 0
    assert engine._free.qsize() == 1


@pytest.mark.asyncio
async def test_inference_error_rebuilds_worker(config):
    engine = make_engine(config, fail=True)
    await engine.start()
    job = engine._jobs.create()
    params = TtsParams(sample="anna_de.wav", text="Hallo.")
    path = engine.samples.resolve_path("anna_de.wav")
    engine.admit()

    with pytest.raises(InferenceError):
        async for _ in engine.stream(job, params, path):
            pass
    assert job.state == ERROR
    # A healthy worker is back in the pool and pending was released.
    assert engine.queue_depth == 0
    assert engine._free.qsize() == 1


def test_default_factory_selects_moss(monkeypatch):
    from src import moss_backend

    class FakeMoss:
        supports_streaming = True

        def __init__(self, config, device):
            self.config, self.device = config, device

    monkeypatch.setattr(moss_backend, "MossWorker", FakeMoss)
    factory = _default_factory(Config(tts_backend="moss"))
    assert isinstance(factory(0, "cpu"), FakeMoss)


def test_health_reports_the_device_the_workers_really_use(config):
    """A backend may override the engine's choice (MOSS pins itself to the CPU).

    Until the engine read this back, `ready` and `/health` announced "cuda"
    while the model ran on the CPU — reported live from a real server log.
    """
    class CpuPinnedWorker:
        supports_streaming = False
        sample_rate = 24000
        device = "cpu"          # resolved elsewhere than requested

        def __init__(self, worker_id, device):
            pass

        def self_test(self):
            return True

    config.tts_backend = "moss"
    detection = Detection(backend="cuda", device="cuda", torch_index_url="x", detail="test")
    engine = Engine(
        config,
        detection,
        JobRegistry(),
        worker_factory=lambda i, d: CpuPinnedWorker(i, d),
    )
    asyncio.run(engine.start())

    assert engine.health()["device"] == "cpu", "must report where the model runs"
    assert engine.health()["backend"] == "cuda", "the machine still has that GPU"


@pytest.mark.parametrize("backend", ["chatterbox", "xttts", "", "Moss", "xtts-v2"])
def test_unknown_backend_fails_loudly(backend):
    """F5 is the default AND the last branch, so anything unrecognised used to
    become F5 in silence — wrong voice, no error, and `/health` still reporting
    the name that was asked for. The realistic source is a typo in the
    documented `F5W_TTS_BACKEND`, or an id a plugin stored before a rename.
    """
    with pytest.raises(ValueError) as err:
        _default_factory(Config(tts_backend=backend))
    message = str(err.value)
    assert backend or "''" in message, "the offending value must appear"
    for known in KNOWN_BACKENDS:
        assert known in message, "the message must list what IS valid"


@pytest.mark.parametrize("backend", ["f5", "xtts", "moss"])
def test_every_known_backend_still_resolves(backend):
    """The guard must not lock out a backend that genuinely exists — and this
    list is what engines.json publishes to the plugin."""
    assert _default_factory(Config(tts_backend=backend)) is not None


def test_engines_json_matches_the_backends_this_build_knows():
    """`engines.json` (repo root) is what the Echokraut plugin puts in its
    dropdown, and `KNOWN_BACKENDS` is what this wrapper will actually start.

    Drift either way is a user-visible bug with no other alarm: an id only in
    the file is selectable and then refused at startup, and an id only in the
    code is invisible to everyone using the plugin. The file is published, not
    imported, so nothing else would ever catch this.
    """
    import json
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent.parent
    engines_file = repo_root / "engines.json"
    if not engines_file.is_file():
        pytest.skip("engines.json is only present in a full checkout")

    listed = {e["id"] for e in json.loads(engines_file.read_text(encoding="utf-8"))["engines"]}
    assert listed == set(KNOWN_BACKENDS), (
        f"engines.json lists {sorted(listed)}, the engine knows {sorted(KNOWN_BACKENDS)}"
    )
