import numpy as np
import pytest

from conftest import make_engine
from src.engine import InferenceError, QueueFull, TtsParams, float_to_pcm16
from src.jobs import CANCELLED, DONE, ERROR


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
