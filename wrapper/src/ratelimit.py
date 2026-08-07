"""Sliding-window request limits — one global, one per caller IP.

What this is *not*: back-pressure. ``engine.admit()`` already answers 503 when
the queue is full, which says "busy right now, retry shortly". This module
answers 429, which says "you have had your share for this hour". They protect
different things and both stay in place.

Design decisions worth keeping:

* **Sliding window, not hourly buckets.** With fixed buckets a caller can spend
  the whole quota at 10:59 and the whole next one at 11:01. Timestamps in a
  deque cost a little memory and behave the way people expect.
* **Only accepted requests count.** If rejections counted too, a client that
  keeps hammering would keep refilling its own window and stay locked out
  forever — the limit would punish retrying rather than usage.
* **The IP table ages and is capped.** Anything else is a memory leak with a
  rotating (or spoofed) source address.
* **Time is injected.** The tests must not sleep an hour.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from typing import Callable, Optional

# Upper bound on tracked client addresses. Reached only under a rotating or
# spoofed source address, where the per-IP limit is useless anyway — the global
# one still applies. Least-recently-seen entries are dropped first.
MAX_TRACKED_CLIENTS = 4096

# How often the full table is swept for clients whose window has expired. A
# sweep is O(tracked clients); doing it on every request would make a cheap
# check walk thousands of entries, doing it never would let the table age only
# by eviction. 1 = sweep every request (used by the tests).
SWEEP_EVERY = 64

GLOBAL_KEY = "*"


class RateLimitExceeded(Exception):
    """Raised by :meth:`RateLimiter.check`. Carries the wait in seconds."""

    def __init__(self, scope: str, limit: int, retry_after: int):
        self.scope = scope  # "global" | "ip"
        self.limit = limit
        self.retry_after = max(1, int(retry_after))
        super().__init__(
            f"{scope} rate limit of {limit} requests per hour reached; "
            f"retry in {self.retry_after}s"
        )


class RateLimiter:
    """Counts requests in a rolling window (default one hour).

    ``per_hour`` and ``per_ip_per_hour`` of 0 (or less) disable the respective
    limit, which is the default — a local wrapper serving one game client has no
    reason to ration itself.
    """

    def __init__(
        self,
        per_hour: int = 0,
        per_ip_per_hour: int = 0,
        window_seconds: int = 3600,
        clock: Optional[Callable[[], float]] = None,
        sweep_every: int = SWEEP_EVERY,
    ):
        self.per_hour = max(0, int(per_hour))
        self.per_ip_per_hour = max(0, int(per_ip_per_hour))
        self.window = max(1, int(window_seconds))
        # Monotonic: a wall-clock jump (NTP, DST) must not hand out free quota
        # or lock everyone out.
        self._clock = clock or time.monotonic
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()
        self._sweep_countdown = sweep_every
        self._sweep_every = sweep_every

    @property
    def enabled(self) -> bool:
        return self.per_hour > 0 or self.per_ip_per_hour > 0

    def _prune(self, key: str, now: float) -> deque:
        window_start = now - self.window
        hits = self._hits.get(key)
        if hits is None:
            hits = deque()
            self._hits[key] = hits
        while hits and hits[0] <= window_start:
            hits.popleft()
        return hits

    def _forget_idle(self, now: float) -> None:
        """Drop clients whose window has run out, then cap what is left.

        A client is only pruned when it is *checked*, so an address that goes
        quiet keeps its stale timestamps forever — the table would age only by
        hitting the cap. Hence this sweep. It walks every entry, so it runs on a
        schedule rather than on every request; the cap below is the hard bound
        in between.
        """
        window_start = now - self.window
        self._sweep_countdown -= 1
        if self._sweep_countdown <= 0:
            self._sweep_countdown = self._sweep_every
            stale = [
                key
                for key, hits in self._hits.items()
                if key != GLOBAL_KEY and (not hits or hits[-1] <= window_start)
            ]
            for key in stale:
                del self._hits[key]
        else:
            for key in [k for k, v in self._hits.items() if not v and k != GLOBAL_KEY]:
                del self._hits[key]

        while len(self._hits) > MAX_TRACKED_CLIENTS:
            oldest, _ = next(iter(self._hits.items()))
            if oldest == GLOBAL_KEY:  # never evict the global counter
                self._hits.move_to_end(GLOBAL_KEY)
                continue
            del self._hits[oldest]

    def check(self, client: str) -> None:
        """Record one request, or raise :class:`RateLimitExceeded`.

        Nothing is recorded when the call is rejected (see module docstring).
        """
        if not self.enabled:
            return
        now = self._clock()
        with self._lock:
            checks = []
            if self.per_hour > 0:
                checks.append(("global", GLOBAL_KEY, self.per_hour))
            if self.per_ip_per_hour > 0:
                checks.append(("ip", f"ip:{client}", self.per_ip_per_hour))

            for scope, key, limit in checks:
                hits = self._prune(key, now)
                if len(hits) >= limit:
                    # The oldest hit leaving the window is when a slot frees up.
                    raise RateLimitExceeded(
                        scope, limit, retry_after=self.window - (now - hits[0])
                    )

            for _scope, key, _limit in checks:
                self._hits[key].append(now)
                self._hits.move_to_end(key)
            self._forget_idle(now)

    def snapshot(self) -> dict:
        """Current usage, for /health. Cheap and lock-protected."""
        if not self.enabled:
            return {"enabled": False}
        now = self._clock()
        with self._lock:
            used = len(self._prune(GLOBAL_KEY, now)) if self.per_hour > 0 else 0
            clients = sum(1 for k in self._hits if k.startswith("ip:"))
        return {
            "enabled": True,
            "per_hour": self.per_hour,
            "per_ip_per_hour": self.per_ip_per_hour,
            "used_this_window": used,
            "tracked_clients": clients,
        }


def client_address(request, trust_forwarded_for: bool = False) -> str:
    """Best available caller address for a Starlette/FastAPI request.

    ``X-Forwarded-For`` is only read when explicitly trusted. Behind a reverse
    proxy the socket address is the proxy's and every caller would share one
    bucket — but honouring the header by default would be worse: anyone could
    then invent an address per request and the per-IP limit would mean nothing.
    """
    if trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            # Left-most entry is the original client; the rest are proxies.
            first = forwarded.split(",")[0].strip()
            if first:
                return first
    client = getattr(request, "client", None)
    return getattr(client, "host", None) or "unknown"
