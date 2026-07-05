"""Tests for the tqdm→NDJSON download-progress hook.

No real tqdm is installed in the test venv (by design — the heavy deps are
mocked), so these drive the subclass against a tiny fake tqdm base and inject a
fake ``tqdm`` package into ``sys.modules`` for the patch/restore test.
"""

import sys
import types

from src import ndjson, progress


class FakeTqdm:
    """Minimal stand-in for tqdm's std class (enough for the byte-bar path)."""

    def __init__(self, *args, total=None, unit="", desc="", disable=None, initial=0, **kw):
        self.total = total
        self.unit = unit
        self.desc = desc
        self.disable = bool(disable)
        self.n = initial

    def update(self, n=1):
        self.n += n

    def close(self):
        pass


def _sink():
    emitted = []
    return emitted, lambda label, pct: emitted.append((label, pct))


def test_byte_bar_reports_throttled_progress():
    emitted, emit = _sink()
    Sub = progress._make_subclass(FakeTqdm, emit)
    bar = Sub(total=100, unit="B", desc="model.bin")  # __init__ emits 0%

    bar.update(2)   # 2%  → below 3% step, suppressed
    bar.update(2)   # 4%  → emitted
    bar.update(50)  # 54% → emitted
    bar.close()     # final → 100%

    assert emitted[0] == ("model.bin", 0)
    pcts = [p for _, p in emitted]
    assert 4 in pcts and 54 in pcts
    assert pcts[-1] == 100
    # Throttled: never one event per single update.
    assert len(emitted) < 6
    assert all(label == "model.bin" for label, _ in emitted)


def test_non_byte_bar_is_ignored():
    emitted, emit = _sink()
    Sub = progress._make_subclass(FakeTqdm, emit)
    bar = Sub(total=4, unit="it", desc="fetching files")  # iteration bar, not bytes
    bar.update(1)
    bar.update(3)
    bar.close()
    assert emitted == []


def test_unknown_total_is_ignored():
    emitted, emit = _sink()
    Sub = progress._make_subclass(FakeTqdm, emit)
    bar = Sub(total=None, unit="B", desc="stream")  # size unknown → no percentage
    bar.update(1234)
    bar.close()
    assert emitted == []


def test_explicit_disable_is_respected():
    emitted, emit = _sink()
    Sub = progress._make_subclass(FakeTqdm, emit)
    bar = Sub(total=100, unit="B", desc="x", disable=True)  # globally disabled bar
    bar.update(50)
    bar.close()
    assert emitted == []


def test_auto_disable_is_forced_on():
    # tqdm's default disable=None auto-disables on a non-tty; the subclass must
    # force it on so coqui's byte bars still advance under piped stdout.
    emitted, emit = _sink()
    Sub = progress._make_subclass(FakeTqdm, emit)
    bar = Sub(total=100, unit="iB")  # disable omitted → None
    assert bar.disable is False
    bar.update(100)
    bar.close()
    assert emitted and emitted[-1][1] == 100


def test_empty_desc_falls_back_to_download():
    emitted, emit = _sink()
    Sub = progress._make_subclass(FakeTqdm, emit)
    bar = Sub(total=10, unit="iB")  # coqui bars have no desc
    bar.close()
    assert emitted[-1][0] == "Download"


def test_emit_failure_never_breaks_download():
    def boom(label, pct):
        raise RuntimeError("host went away")

    Sub = progress._make_subclass(FakeTqdm, boom)
    bar = Sub(total=100, unit="B")  # must not raise despite emit raising
    bar.update(50)
    bar.close()


def test_patched_downloads_swaps_and_restores(monkeypatch):
    # Inject a fake tqdm package so patched_downloads has something to swap.
    pkg = types.ModuleType("tqdm")
    auto_mod = types.ModuleType("tqdm.auto")
    std_mod = types.ModuleType("tqdm.std")
    pkg.tqdm = FakeTqdm
    auto_mod.tqdm = FakeTqdm
    std_mod.tqdm = FakeTqdm
    pkg.auto = auto_mod
    pkg.std = std_mod
    monkeypatch.setitem(sys.modules, "tqdm", pkg)
    monkeypatch.setitem(sys.modules, "tqdm.auto", auto_mod)
    monkeypatch.setitem(sys.modules, "tqdm.std", std_mod)

    emitted, emit = _sink()
    with progress.patched_downloads(emit):
        assert std_mod.tqdm is not FakeTqdm  # swapped for the reporting subclass
        bar = std_mod.tqdm(total=10, unit="B", desc="f")
        bar.update(10)
        bar.close()
    # Restored on exit.
    assert std_mod.tqdm is FakeTqdm
    assert auto_mod.tqdm is FakeTqdm
    assert pkg.tqdm is FakeTqdm
    assert emitted[-1] == ("f", 100)


def test_patched_downloads_noop_without_tqdm(monkeypatch):
    # Simulate tqdm not being importable → context must still yield cleanly.
    for name in ("tqdm", "tqdm.auto", "tqdm.std"):
        monkeypatch.setitem(sys.modules, name, None)  # None → ImportError on import
    ran = []
    with progress.patched_downloads(lambda label, pct: ran.append((label, pct))):
        ran.append("body")
    assert "body" in ran


def test_model_progress_emits_progress_event(monkeypatch):
    monkeypatch.setenv("F5W_STEP_INDEX", "5")
    monkeypatch.setenv("F5W_STEP_TOTAL", "6")
    events = []
    monkeypatch.setattr(
        ndjson, "progress",
        lambda index, total, step, message=None, percent=None, **kw: events.append(
            (index, total, step, message, percent)
        ),
    )
    mp = progress.ModelProgress()
    mp.stage("de: ")
    mp._emit("model.safetensors", 42)
    assert events == [(5, 6, "model", "de: model.safetensors", 42)]


def test_model_progress_defaults_without_env(monkeypatch):
    monkeypatch.delenv("F5W_STEP_INDEX", raising=False)
    monkeypatch.delenv("F5W_STEP_TOTAL", raising=False)
    events = []
    monkeypatch.setattr(
        ndjson, "progress",
        lambda index, total, step, message=None, percent=None, **kw: events.append(
            (index, total, step, message, percent)
        ),
    )
    mp = progress.ModelProgress()
    mp._emit("x", 10)
    assert events == [(5, 6, "model", "x", 10)]  # bootstrap model step defaults
