"""Fatal device errors (CUDA "device-side assert" & co.) end the process.

After such an error the CUDA context of the whole process is unusable: an
in-process worker rebuild always fails and every later request 500s (User-Log
2026-11-08: 543 follow-up errors). The engine must instead refuse new work and
exit with EXIT_RESTART_REQUIRED so the host starts a fresh process.
"""

import pytest

from conftest import make_engine
from src import engine as engine_mod
from src.engine import (
    EXIT_RESTART_REQUIRED,
    EngineFailed,
    InferenceError,
    TtsParams,
    is_fatal_device_error,
)
from src.jobs import ERROR

DEVICE_ASSERT = (
    "CUDA error: device-side assert triggered\n"
    "CUDA kernel errors might be asynchronously reported at some other API call"
)


@pytest.mark.parametrize(
    "message",
    [
        DEVICE_ASSERT,
        "CUDA error: an illegal memory access was encountered",
        "CUDA error: unspecified launch failure",
        "HIP error: an illegal memory access was encountered",
        "cudaErrorAssert",
    ],
)
def test_fatal_device_errors_are_recognised(message):
    assert is_fatal_device_error(RuntimeError(message))


@pytest.mark.parametrize(
    "message",
    [
        "CUDA out of memory. Tried to allocate 2.00 GiB",
        "CUDA error: out of memory",
        "simulated CUDA OOM",
        "Input type (torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor) should be the same",
        "expected scalar type Float but found Half",
        "",
    ],
)
def test_recoverable_errors_are_not_fatal(message):
    assert not is_fatal_device_error(RuntimeError(message))


def test_fatal_error_is_found_through_the_cause_chain():
    try:
        try:
            raise RuntimeError(DEVICE_ASSERT)
        except RuntimeError as inner:
            raise InferenceError("wrapped") from inner
    except InferenceError as outer:
        assert is_fatal_device_error(outer)


def test_exit_code_is_ex_tempfail():
    # Mirrored by EchokrautLocalInstaller (Constants.ECHOKRAUTTS_EXIT_RESTART_REQUIRED).
    assert EXIT_RESTART_REQUIRED == 75


async def _run_one(engine, text="Hallo."):
    job = engine._jobs.create()
    params = TtsParams(sample="anna_de.wav", text=text)
    path = engine.samples.resolve_path("anna_de.wav")
    engine.admit()
    with pytest.raises(InferenceError):
        async for _ in engine.stream(job, params, path):
            pass
    return job


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_device_assert_triggers_exit_instead_of_rebuild(config, streaming, monkeypatch):
    calls = []
    engine = make_engine(
        config, fail=True, streaming=streaming, fail_message=DEVICE_ASSERT,
        on_fatal=lambda: calls.append("exit"),
    )
    await engine.start()
    original = engine._workers[0]

    async def no_rebuild(dead):  # pragma: no cover — must not be reached
        raise AssertionError("must not rebuild a worker after a fatal device error")

    monkeypatch.setattr(engine, "_rebuild_worker", no_rebuild)

    job = await _run_one(engine)

    assert job.state == ERROR
    assert calls == ["exit"]
    assert engine._workers[0] is original
    assert engine.queue_depth == 0
    health = engine.health()
    assert health["status"] == "failed"
    assert "device-side assert" in health["fatal_error"]


@pytest.mark.asyncio
async def test_failed_engine_refuses_new_requests(config):
    engine = make_engine(config, fail=True, fail_message=DEVICE_ASSERT)
    await engine.start()
    await _run_one(engine)

    with pytest.raises(EngineFailed):
        engine.admit()


@pytest.mark.asyncio
async def test_fatal_is_reported_once(config, monkeypatch):
    calls, events = [], []
    monkeypatch.setattr(engine_mod.ndjson, "error", lambda msg, fatal=False: events.append((msg, fatal)))
    engine = make_engine(
        config, fail=True, workers=2, fail_message=DEVICE_ASSERT,
        on_fatal=lambda: calls.append("exit"),
    )
    await engine.start()
    # Both requests were admitted before the first one failed.
    j1, j2 = engine._jobs.create(), engine._jobs.create()
    params = TtsParams(sample="anna_de.wav", text="Hallo.")
    path = engine.samples.resolve_path("anna_de.wav")
    engine.admit()
    engine.admit()
    for job in (j1, j2):
        with pytest.raises(InferenceError):
            async for _ in engine.stream(job, params, path):
                pass

    assert calls == ["exit"]
    assert len(events) == 1
    msg, fatal = events[0]
    assert fatal is True
    assert str(EXIT_RESTART_REQUIRED) in msg


@pytest.mark.asyncio
async def test_recoverable_error_still_rebuilds(config):
    calls = []
    engine = make_engine(config, fail=True, on_fatal=lambda: calls.append("exit"))
    await engine.start()
    original = engine._workers[0]

    await _run_one(engine)

    assert calls == []
    assert engine._workers[0] is not original
    assert engine.health()["status"] == "ok"
    engine.admit()  # still accepting work


@pytest.mark.asyncio
async def test_default_on_fatal_schedules_hard_exit(config, monkeypatch):
    exits = []
    monkeypatch.setattr(engine_mod, "_hard_exit", lambda code: exits.append(code))
    monkeypatch.setattr(engine_mod, "FATAL_EXIT_DELAY_S", 0.0)
    from conftest import FakeWorker, cpu_detection
    from src.engine import Engine
    from src.jobs import JobRegistry

    engine = Engine(
        config, cpu_detection(), JobRegistry(),
        worker_factory=lambda i, d: FakeWorker(fail=True, fail_message=DEVICE_ASSERT),
        asr=lambda a, sr, lang: "",
    )
    await engine.start()
    await _run_one(engine)

    import asyncio

    await asyncio.sleep(0.05)
    assert exits == [EXIT_RESTART_REQUIRED]


def test_hard_exit_uses_os_exit_with_the_code(monkeypatch):
    seen = []
    monkeypatch.setattr(engine_mod.os, "_exit", lambda code: seen.append(code))
    engine_mod._hard_exit(EXIT_RESTART_REQUIRED)
    assert seen == [EXIT_RESTART_REQUIRED]
