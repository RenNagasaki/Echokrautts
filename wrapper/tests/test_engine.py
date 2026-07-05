import numpy as np
import pytest

from conftest import FakeWorker, make_engine
from src.config import Config
from src.engine import Engine, InferenceError, QueueFull, TtsParams, _default_factory, float_to_pcm16
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
