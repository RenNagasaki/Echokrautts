"""Response cache: one generation per identical ``/tts`` request, shared.

Two things in one place, because they are the same mechanism:

* **Sharing a running generation.** A second client asking for the same line a
  few milliseconds after the first (the plugin does that, e.g. a dialogue line
  and its replay) does not occupy a second worker. It joins the first run: it
  first gets everything buffered so far, then the rest live.
* **Keeping a finished one.** For ``response_cache_minutes`` after it finished,
  the same request gets the same PCM again, with no generation at all.

Every client keeps its OWN job (``X-Job-Id``), so ``/cancel/{id}`` and a dropped
connection detach only that client. The generation itself — its own internal
job — is cancelled only when the LAST attached client has gone (refcount). A run
that ended cancelled or failed is never stored: a cache entry is always a
complete clip.

MOSS is not deterministic, so a hit is "the same sound again" — intended, not a
bug. ``no_cache`` bypasses the lookup and replaces the stored entry with the new
run; a run that was superseded that way still finishes for the clients already
attached to it, it just is not stored.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import OrderedDict
from typing import AsyncIterator, Callable, Hashable, Optional

from .jobs import CANCELLED, DONE, ERROR, Job, JobRegistry

# How often a waiting client re-checks its own cancel token. A cancel arrives
# through the job registry (a threading.Event), which cannot wake an asyncio
# waiter, so the wait is sliced. 50 ms is far below anything audible.
CANCEL_POLL_S = 0.05

HIT = "hit"          # finished entry from the cache
SHARED = "shared"    # joined a generation that is still running
MISS = "miss"        # new generation, will be stored when it completes
BYPASS = "bypass"    # no_cache: new generation, replaces the entry
OFF = "off"          # cache disabled: new generation, never stored


class SharedGeneration:
    """One generation, its PCM buffer and the clients attached to it."""

    def __init__(self, key: Hashable, job: Job, storable: bool) -> None:
        self.key = key
        self.job = job  # the generation's own job — never handed to a client
        self.parts: list[bytes] = []
        self.size = 0
        self.done = False
        self.error: Optional[BaseException] = None
        self.subscribers = 0
        self.storable = storable
        self.finished_at: Optional[float] = None
        self.task: Optional[asyncio.Task] = None
        self._changed = asyncio.Event()

    @property
    def complete(self) -> bool:
        """Finished normally, i.e. a full clip (what may be cached)."""
        return self.done and self.error is None and self.job.state == DONE

    def _notify(self) -> None:
        # Swap the event so every waiter that grabbed the old one wakes once.
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()

    def append(self, pcm: bytes) -> None:
        self.parts.append(pcm)
        self.size += len(pcm)
        self._notify()

    def finish(self, error: Optional[BaseException] = None) -> None:
        self.error = error
        self.done = True
        self._notify()

    async def wait_changed(self, timeout: float) -> None:
        changed = self._changed
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(changed.wait(), timeout)


class ResponseCache:
    def __init__(
        self,
        minutes: float,
        max_mb: float,
        jobs: JobRegistry,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_s = max(0.0, float(minutes)) * 60.0
        self.max_bytes = int(max(0.0, float(max_mb)) * 1024 * 1024)
        self._jobs = jobs
        self._clock = clock
        self._running: dict[Hashable, SharedGeneration] = {}
        self._done: "OrderedDict[Hashable, SharedGeneration]" = OrderedDict()
        self._bytes = 0
        self.stats = {HIT: 0, SHARED: 0, MISS: 0, BYPASS: 0}

    @property
    def enabled(self) -> bool:
        return self.ttl_s > 0 and self.max_bytes > 0

    # ------------------------------------------------------------- lookup
    def lookup(self, key: Hashable) -> tuple[Optional[SharedGeneration], str]:
        """A generation to join for ``key`` (and how it was found), or None."""
        if not self.enabled:
            return None, OFF
        self._expire()
        gen = self._done.get(key)
        if gen is not None:
            self._done.move_to_end(key)
            self.stats[HIT] += 1
            return gen, HIT
        gen = self._running.get(key)
        # A run whose last client already left is being cancelled: joining it
        # would hand the newcomer a truncated clip. Start a fresh one instead.
        if gen is not None and not gen.job.cancelled:
            self.stats[SHARED] += 1
            return gen, SHARED
        return None, MISS

    def start(
        self,
        key: Hashable,
        job: Job,
        source: AsyncIterator[bytes],
        no_cache: bool = False,
    ) -> tuple[SharedGeneration, str]:
        """Run ``source`` (the engine stream of ``job``) as a shared generation.

        Call only after ``lookup`` found nothing (or for ``no_cache``) and after
        the engine admitted the request — ``source`` must be iterated exactly once.
        """
        storable = self.enabled
        gen = SharedGeneration(key, job, storable)
        if storable:
            previous = self._running.get(key)
            if previous is not None:
                previous.storable = False  # superseded by this newer run
            self._running[key] = gen
            if no_cache:
                self._drop(key)
            self.stats[BYPASS if no_cache else MISS] += 1
            how = BYPASS if no_cache else MISS
        else:
            how = OFF
        gen.task = asyncio.create_task(self._produce(gen, source))
        return gen, how

    async def _produce(self, gen: SharedGeneration, source: AsyncIterator[bytes]) -> None:
        error: Optional[BaseException] = None
        try:
            async for pcm in source:
                gen.append(pcm)
        except asyncio.CancelledError as exc:
            error = exc
            raise
        except Exception as exc:  # noqa: BLE001 — handed to every attached client
            error = exc
        finally:
            with contextlib.suppress(Exception):
                await source.aclose()  # type: ignore[attr-defined]
            gen.finish(error)
            gen.finished_at = self._clock()
            self._finished(gen)

    def _finished(self, gen: SharedGeneration) -> None:
        if self._running.get(gen.key) is gen:
            del self._running[gen.key]
        if not (gen.storable and gen.complete) or gen.size > self.max_bytes:
            return
        self._drop(gen.key)
        self._done[gen.key] = gen
        self._bytes += gen.size
        while self._bytes > self.max_bytes and self._done:
            _, old = self._done.popitem(last=False)
            self._bytes -= old.size

    def _drop(self, key: Hashable) -> None:
        old = self._done.pop(key, None)
        if old is not None:
            self._bytes -= old.size

    def _expire(self) -> None:
        now = self._clock()
        for key in [k for k, g in self._done.items() if now - (g.finished_at or now) >= self.ttl_s]:
            self._drop(key)

    # ------------------------------------------------------------ clients
    def attach(self, gen: SharedGeneration) -> None:
        """Count a client in. Synchronous on purpose: between the lookup and
        the first read nothing may see the generation without this client."""
        gen.subscribers += 1

    async def subscribe(self, gen: SharedGeneration, client: Job) -> AsyncIterator[bytes]:
        """PCM of ``gen`` for one client: the buffer first, then live.

        Must follow :meth:`attach`. Leaving — normally, by ``/cancel`` of the
        client's job or by the connection closing — detaches; the last client
        to leave cancels a generation that is still running.
        """
        jobs = self._jobs
        index = 0
        try:
            jobs.mark_running(client, gen.job.worker_id or 0)
            while True:
                if client.cancelled:
                    jobs.finish(client, CANCELLED)
                    return
                if index < len(gen.parts):
                    pcm = gen.parts[index]
                    index += 1
                    self._mirror(gen, client)
                    yield pcm
                    continue
                if gen.done:
                    self._mirror(gen, client)
                    if gen.error is not None:
                        jobs.finish(client, ERROR, str(gen.error))
                        raise gen.error
                    jobs.finish(client, DONE if gen.complete else CANCELLED)
                    return
                await gen.wait_changed(CANCEL_POLL_S)
        finally:
            self.detach(gen)

    def detach(self, gen: SharedGeneration) -> None:
        gen.subscribers -= 1
        if gen.subscribers <= 0 and not gen.done:
            # Nobody listens any more: stop the engine at its next check.
            gen.job.cancel_event.set()

    @staticmethod
    def _mirror(gen: SharedGeneration, client: Job) -> None:
        # /jobs/{client id} reports the generation's progress.
        client.sentences_total = gen.job.sentences_total
        client.sentences_done = gen.job.sentences_done

    # -------------------------------------------------------------- health
    def snapshot(self) -> dict:
        if self.enabled:
            self._expire()
        return {
            "enabled": self.enabled,
            "minutes": self.ttl_s / 60.0,
            "max_mb": round(self.max_bytes / 1024 / 1024, 1),
            "entries": len(self._done),
            "mb": round(self._bytes / 1024 / 1024, 2),
            "running": len(self._running),
            "hits": self.stats[HIT],
            "shared": self.stats[SHARED],
            "misses": self.stats[MISS],
            "bypassed": self.stats[BYPASS],
        }
