"""Response cache: hits, expiry, sharing a running generation, refcounted cancel."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from conftest import make_engine
from src.jobs import CANCELLED, DONE, ERROR, JobRegistry
from src.response_cache import BYPASS, HIT, MISS, OFF, SHARED, ResponseCache
from src.server import create_app


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeRun:
    """Engine-like source: yields what the test feeds, honours the job's cancel
    token between parts exactly like ``Engine.stream`` does."""

    def __init__(self, jobs: JobRegistry, job) -> None:
        self.jobs, self.job = jobs, job
        self.queue: asyncio.Queue = asyncio.Queue()
        self.started = 0

    async def stream(self):
        self.started += 1
        while True:
            item = await self.queue.get()
            if self.job.cancelled:
                self.jobs.finish(self.job, CANCELLED)
                return
            if item is None:
                self.jobs.finish(self.job, DONE)
                return
            if isinstance(item, Exception):
                self.jobs.finish(self.job, ERROR, str(item))
                raise item
            yield item

    def feed(self, *items) -> None:
        for item in items:
            self.queue.put_nowait(item)


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


def _new(cache, jobs, key="k", no_cache=False):
    job = jobs.create()
    run = FakeRun(jobs, job)
    gen, how = cache.start(key, job, run.stream(), no_cache=no_cache)
    return gen, how, run


async def _drain(cache, gen, client, out: list) -> None:
    async for pcm in cache.subscribe(gen, client):
        out.append(pcm)


def _client(cache, jobs, gen):
    job = jobs.create()
    cache.attach(gen)
    return job


@pytest.fixture
def jobs():
    return JobRegistry()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def cache(jobs, clock):
    return ResponseCache(minutes=5, max_mb=1, jobs=jobs, clock=clock)


@pytest.mark.asyncio
async def test_finished_run_is_a_hit_with_the_same_pcm(cache, jobs):
    assert cache.lookup("k") == (None, MISS)
    gen, how, run = _new(cache, jobs)
    assert how == MISS
    first = []
    task = asyncio.create_task(_drain(cache, gen, _client(cache, jobs, gen), first))
    run.feed(b"ab", b"cd", None)
    await task
    again, how = cache.lookup("k")
    assert how == HIT and again is gen
    second = []
    client = _client(cache, jobs, again)
    await _drain(cache, again, client, second)
    assert first == second == [b"ab", b"cd"]
    assert client.state == DONE
    assert run.started == 1


@pytest.mark.asyncio
async def test_entry_expires(cache, jobs, clock):
    gen, _, run = _new(cache, jobs)
    run.feed(b"x", None)
    await _drain(cache, gen, _client(cache, jobs, gen), [])
    clock.now += 4 * 60
    assert cache.lookup("k")[1] == HIT
    clock.now += 61
    assert cache.lookup("k") == (None, MISS)
    assert cache.snapshot()["entries"] == 0


@pytest.mark.asyncio
async def test_late_client_gets_the_buffer_then_live(cache, jobs):
    gen, _, run = _new(cache, jobs)
    early = []
    t1 = asyncio.create_task(_drain(cache, gen, _client(cache, jobs, gen), early))
    run.feed(b"1", b"2")
    await _settle()
    joined, how = cache.lookup("k")
    assert how == SHARED and joined is gen
    late = []
    t2 = asyncio.create_task(_drain(cache, gen, _client(cache, jobs, gen), late))
    await _settle()
    assert late == [b"1", b"2"]  # the buffer, before anything new arrived
    run.feed(b"3", None)
    await asyncio.gather(t1, t2)
    assert early == late == [b"1", b"2", b"3"]
    assert run.started == 1


@pytest.mark.asyncio
async def test_cancel_one_of_two_keeps_the_generation(cache, jobs):
    gen, _, run = _new(cache, jobs)
    a, b = _client(cache, jobs, gen), _client(cache, jobs, gen)
    out_a, out_b = [], []
    ta = asyncio.create_task(_drain(cache, gen, a, out_a))
    tb = asyncio.create_task(_drain(cache, gen, b, out_b))
    run.feed(b"1")
    await _settle()
    jobs.cancel(a.job_id)
    await ta
    assert a.state == CANCELLED
    assert not gen.job.cancelled  # B still listens
    run.feed(b"2", None)
    await tb
    assert out_b == [b"1", b"2"] and b.state == DONE
    assert cache.lookup("k")[1] == HIT  # complete clip, stored


@pytest.mark.asyncio
async def test_cancel_of_all_clients_stops_and_stores_nothing(cache, jobs):
    gen, _, run = _new(cache, jobs)
    a, b = _client(cache, jobs, gen), _client(cache, jobs, gen)
    ta = asyncio.create_task(_drain(cache, gen, a, []))
    tb = asyncio.create_task(_drain(cache, gen, b, []))
    run.feed(b"1")
    await _settle()
    jobs.cancel(a.job_id)
    jobs.cancel(b.job_id)
    await asyncio.gather(ta, tb)
    assert gen.job.cancelled
    run.feed(b"2")  # the engine reaches its next cancel check
    await gen.task
    assert gen.job.state == CANCELLED
    assert cache.lookup("k") == (None, MISS)
    assert cache.snapshot()["running"] == 0


@pytest.mark.asyncio
async def test_disconnect_detaches_like_a_cancel(cache, jobs):
    gen, _, run = _new(cache, jobs)
    client = _client(cache, jobs, gen)
    stream = cache.subscribe(gen, client)
    run.feed(b"1")
    assert await stream.__anext__() == b"1"
    await stream.aclose()  # what StreamingResponse does on a dropped connection
    assert gen.subscribers == 0 and gen.job.cancelled


@pytest.mark.asyncio
async def test_no_joining_a_run_that_is_being_cancelled(cache, jobs):
    gen, _, _ = _new(cache, jobs)
    client = _client(cache, jobs, gen)
    cache.detach(gen)  # last client gone, engine not at its check yet
    assert gen.job.cancelled and client
    assert cache.lookup("k") == (None, MISS)


@pytest.mark.asyncio
async def test_failed_run_reaches_every_client_and_is_not_stored(cache, jobs):
    gen, _, run = _new(cache, jobs)
    a = _client(cache, jobs, gen)
    run.feed(b"1", RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await _drain(cache, gen, a, [])
    assert a.state == ERROR
    assert cache.lookup("k") == (None, MISS)


@pytest.mark.asyncio
async def test_no_cache_replaces_the_entry(cache, jobs):
    old, _, run = _new(cache, jobs)
    run.feed(b"old", None)
    await _drain(cache, old, _client(cache, jobs, old), [])
    new, how, run2 = _new(cache, jobs, no_cache=True)
    assert how == BYPASS
    assert cache.lookup("k")[1] == SHARED  # the running new one, not the old entry
    run2.feed(b"new", None)
    await _drain(cache, new, _client(cache, jobs, new), [])
    hit, how = cache.lookup("k")
    assert how == HIT and hit.parts == [b"new"]


@pytest.mark.asyncio
async def test_superseded_run_finishes_but_is_not_stored(cache, jobs):
    old, _, run = _new(cache, jobs)
    out = []
    t = asyncio.create_task(_drain(cache, old, _client(cache, jobs, old), out))
    new, _, run2 = _new(cache, jobs, no_cache=True)
    run.feed(b"old", None)
    await t
    assert out == [b"old"]  # its client still got the whole clip
    run2.feed(b"new", None)
    await _drain(cache, new, _client(cache, jobs, new), [])
    assert cache.lookup("k")[0].parts == [b"new"]


@pytest.mark.asyncio
async def test_lru_keeps_the_byte_budget(jobs, clock):
    cache = ResponseCache(minutes=5, max_mb=3 / 1024 / 1024, jobs=jobs, clock=clock)  # 3 bytes
    for key in ("a", "b"):
        gen, _, run = _new(cache, jobs, key)
        run.feed(b"xx", None)
        await _drain(cache, gen, _client(cache, jobs, gen), [])
    assert cache.lookup("a") == (None, MISS)  # evicted
    assert cache.lookup("b")[1] == HIT
    gen, _, run = _new(cache, jobs, "big")
    run.feed(b"xxxx", None)  # larger than the whole budget: never stored
    await _drain(cache, gen, _client(cache, jobs, gen), [])
    assert cache.lookup("big") == (None, MISS)


@pytest.mark.asyncio
async def test_zero_minutes_disables_sharing_and_storing(jobs, clock):
    cache = ResponseCache(minutes=0, max_mb=64, jobs=jobs, clock=clock)
    assert cache.lookup("k") == (None, OFF)
    gen, how, run = _new(cache, jobs)
    assert how == OFF
    assert cache.lookup("k") == (None, OFF)
    run.feed(b"1", None)
    await _drain(cache, gen, _client(cache, jobs, gen), [])
    assert cache.snapshot()["enabled"] is False
    assert cache.snapshot()["entries"] == 0


# ------------------------------------------------------------ through HTTP
def _calls(engine) -> int:
    return sum(len(w.languages) for w in engine._workers)


def test_http_second_request_is_a_hit(config):
    engine = make_engine(config)
    with TestClient(create_app(config=config, engine=engine)) as c:
        body = {"sample": "anna_de.wav", "text": "Hallo."}
        r1 = c.post("/tts", json=body)
        r2 = c.post("/tts", json={**body, "format": "wav"})
        assert r1.headers["x-cache"] == MISS and r2.headers["x-cache"] == HIT
        assert r2.content.endswith(r1.content)  # same PCM, RIFF header in front
        assert r1.headers["x-job-id"] != r2.headers["x-job-id"]
        assert c.get(f"/jobs/{r2.headers['x-job-id']}").json()["state"] == "done"
        health = c.get("/health").json()["response_cache"]
        assert health["enabled"] and health["entries"] == 1 and health["hits"] == 1
    assert _calls(engine) == 1


def test_http_no_cache_generates_anew(config):
    engine = make_engine(config)
    with TestClient(create_app(config=config, engine=engine)) as c:
        body = {"sample": "anna_de.wav", "text": "Hallo."}
        c.post("/tts", json=body)
        r = c.post("/tts", json={**body, "no_cache": True})
        assert r.headers["x-cache"] == BYPASS
    assert _calls(engine) == 2


def test_http_different_options_are_different_entries(config):
    engine = make_engine(config)
    with TestClient(create_app(config=config, engine=engine)) as c:
        body = {"sample": "anna_de.wav", "text": "Hallo."}
        c.post("/tts", json=body)
        assert c.post("/tts", json={**body, "speed": 1.2}).headers["x-cache"] == MISS
        assert c.post("/tts", json={**body, "text": "Tschüss."}).headers["x-cache"] == MISS
        assert c.post("/tts", json={**body, "ref_text": "anders"}).headers["x-cache"] == MISS
        assert c.post("/tts", json={**body, "sample": "anna_de"}).headers["x-cache"] == HIT
    assert _calls(engine) == 4


def test_http_cache_off(config):
    config.response_cache_minutes = 0
    engine = make_engine(config)
    with TestClient(create_app(config=config, engine=engine)) as c:
        body = {"sample": "anna_de.wav", "text": "Hallo."}
        assert c.post("/tts", json=body).headers["x-cache"] == OFF
        assert c.post("/tts", json=body).headers["x-cache"] == OFF
        assert c.get("/health").json()["response_cache"]["enabled"] is False
    assert _calls(engine) == 2
