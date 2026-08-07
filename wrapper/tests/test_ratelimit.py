"""Sliding-window rate limits (src/ratelimit.py).

The clock is injected, so a one-hour window is tested in microseconds.
"""

from __future__ import annotations

import types

import pytest

from src.ratelimit import (
    GLOBAL_KEY,
    MAX_TRACKED_CLIENTS,
    RateLimitExceeded,
    RateLimiter,
    client_address,
)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


# ------------------------------------------------------------------- disabled
def test_disabled_by_default(clock):
    limiter = RateLimiter(clock=clock)
    assert limiter.enabled is False
    for _ in range(1000):
        limiter.check("1.2.3.4")  # never raises
    assert limiter.snapshot() == {"enabled": False}


# --------------------------------------------------------------------- global
def test_global_limit_blocks_the_n_plus_first(clock):
    limiter = RateLimiter(per_hour=3, clock=clock)
    for _ in range(3):
        limiter.check("1.2.3.4")
    with pytest.raises(RateLimitExceeded) as excinfo:
        limiter.check("1.2.3.4")
    assert excinfo.value.scope == "global"
    assert excinfo.value.limit == 3


def test_global_limit_counts_across_clients(clock):
    limiter = RateLimiter(per_hour=2, clock=clock)
    limiter.check("1.1.1.1")
    limiter.check("2.2.2.2")
    with pytest.raises(RateLimitExceeded):
        limiter.check("3.3.3.3")


def test_window_slides(clock):
    limiter = RateLimiter(per_hour=2, window_seconds=3600, clock=clock)
    limiter.check("a")
    clock.advance(1800)
    limiter.check("a")
    with pytest.raises(RateLimitExceeded):
        limiter.check("a")
    # The first hit leaves the window → exactly one slot opens, not two.
    clock.advance(1801)
    limiter.check("a")
    with pytest.raises(RateLimitExceeded):
        limiter.check("a")


def test_retry_after_points_at_the_freeing_slot(clock):
    limiter = RateLimiter(per_hour=1, window_seconds=3600, clock=clock)
    limiter.check("a")
    clock.advance(600)
    with pytest.raises(RateLimitExceeded) as excinfo:
        limiter.check("a")
    assert excinfo.value.retry_after == 3000  # 3600 - 600
    assert excinfo.value.retry_after >= 1


def test_rejections_do_not_extend_the_lockout(clock):
    # If rejected calls counted, hammering would refill the window forever.
    limiter = RateLimiter(per_hour=1, window_seconds=100, clock=clock)
    limiter.check("a")
    for _ in range(50):
        clock.advance(1)
        with pytest.raises(RateLimitExceeded):
            limiter.check("a")
    clock.advance(51)  # 101s after the single accepted call
    limiter.check("a")


# --------------------------------------------------------------------- per IP
def test_per_ip_limit_is_independent_per_client(clock):
    limiter = RateLimiter(per_ip_per_hour=2, clock=clock)
    limiter.check("1.1.1.1")
    limiter.check("1.1.1.1")
    with pytest.raises(RateLimitExceeded) as excinfo:
        limiter.check("1.1.1.1")
    assert excinfo.value.scope == "ip"
    limiter.check("2.2.2.2")  # a different caller is unaffected


def test_global_and_per_ip_together(clock):
    limiter = RateLimiter(per_hour=3, per_ip_per_hour=2, clock=clock)
    limiter.check("a")
    limiter.check("a")
    with pytest.raises(RateLimitExceeded) as excinfo:
        limiter.check("a")
    assert excinfo.value.scope == "ip"
    limiter.check("b")  # 3rd globally, 1st for b
    with pytest.raises(RateLimitExceeded) as excinfo:
        limiter.check("b")
    assert excinfo.value.scope == "global"


def test_a_rejected_global_check_does_not_consume_ip_quota(clock):
    # Both counters must move together or neither: otherwise the per-IP window
    # silently drains while the global limit is doing the rejecting.
    limiter = RateLimiter(per_hour=1, per_ip_per_hour=5, clock=clock)
    limiter.check("a")
    with pytest.raises(RateLimitExceeded) as excinfo:
        limiter.check("b")
    assert excinfo.value.scope == "global"
    assert not limiter._hits.get("ip:b")  # b was never charged for it


# ------------------------------------------------------------------- bookkeeping
def test_idle_clients_are_forgotten(clock):
    # A client that goes quiet is never pruned by its own check — the periodic
    # sweep is what ages it out. sweep_every=1 makes that deterministic here.
    limiter = RateLimiter(per_ip_per_hour=1, window_seconds=10, clock=clock, sweep_every=1)
    limiter.check("1.1.1.1")
    clock.advance(11)
    limiter.check("2.2.2.2")
    assert "ip:1.1.1.1" not in limiter._hits
    assert "ip:2.2.2.2" in limiter._hits


def test_sweep_is_not_run_on_every_request(clock):
    # The sweep walks the whole table; doing it per request would make a cheap
    # check O(tracked clients).
    limiter = RateLimiter(per_ip_per_hour=1, window_seconds=10, clock=clock, sweep_every=100)
    limiter.check("1.1.1.1")
    clock.advance(11)
    limiter.check("2.2.2.2")
    assert "ip:1.1.1.1" in limiter._hits  # still there, and bounded by the cap


def test_client_table_is_capped(clock):
    limiter = RateLimiter(per_ip_per_hour=5, clock=clock)
    for i in range(MAX_TRACKED_CLIENTS + 200):
        limiter.check(f"10.0.{i // 256}.{i % 256}")
    assert len(limiter._hits) <= MAX_TRACKED_CLIENTS


def test_global_counter_survives_eviction(clock):
    limiter = RateLimiter(per_hour=10**9, per_ip_per_hour=5, clock=clock)
    for i in range(MAX_TRACKED_CLIENTS + 50):
        limiter.check(f"10.0.{i // 256}.{i % 256}")
    assert GLOBAL_KEY in limiter._hits


def test_snapshot_reports_usage(clock):
    limiter = RateLimiter(per_hour=5, per_ip_per_hour=2, clock=clock)
    limiter.check("a")
    limiter.check("b")
    snap = limiter.snapshot()
    assert snap["enabled"] is True
    assert snap["per_hour"] == 5
    assert snap["used_this_window"] == 2
    assert snap["tracked_clients"] == 2


# ---------------------------------------------------------------- client_address
def _request(host, headers=None):
    return types.SimpleNamespace(
        client=types.SimpleNamespace(host=host), headers=headers or {}
    )


def test_socket_address_by_default():
    request = _request("5.5.5.5", {"x-forwarded-for": "1.1.1.1"})
    # Untrusted header must NOT win — otherwise any caller invents an address
    # per request and the per-IP limit is worthless.
    assert client_address(request, trust_forwarded_for=False) == "5.5.5.5"


def test_forwarded_for_when_trusted():
    request = _request("10.0.0.1", {"x-forwarded-for": "1.1.1.1, 10.0.0.9"})
    assert client_address(request, trust_forwarded_for=True) == "1.1.1.1"


def test_trusted_but_header_absent_falls_back():
    assert client_address(_request("5.5.5.5"), trust_forwarded_for=True) == "5.5.5.5"


def test_missing_client_is_not_a_crash():
    request = types.SimpleNamespace(client=None, headers={})
    assert client_address(request) == "unknown"
