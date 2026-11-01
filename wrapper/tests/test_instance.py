"""One install folder, one owner (src/instance.py).

Driven by a tester log: the installer was started twice, the second bootstrap
ran ``uv venv --clear`` on a venv the first was still installing into, and the
leftovers kept ``.venv\\Scripts`` locked for good.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src import instance
from src.instance import InstanceLock, ProcessInfo, stale_processes, take_over

WRAPPER_ROOT = Path(__file__).resolve().parent.parent

VENV = Path(r"C:\LocalTTS\echokrautts\.venv") if instance.IS_WINDOWS else Path("/i/.venv")
UV = Path(r"C:\LocalTTS\echokrautts\.uv") if instance.IS_WINDOWS else Path("/i/.uv")
BASE_PY = r"C:\Users\x\AppData\Roaming\uv\python\python.exe" if instance.IS_WINDOWS else "/usr/bin/python3"


def _exe(root: Path, name: str) -> str:
    return str(root / name)


# ------------------------------------------------------------------ the lock
def test_a_second_owner_is_refused_while_the_first_holds_it(tmp_path):
    first, second = InstanceLock(tmp_path), InstanceLock(tmp_path)
    assert first.try_acquire()
    assert not second.try_acquire()
    first.release()
    assert second.try_acquire()
    second.release()


def test_the_holder_leaves_its_pid_for_the_one_taking_over(tmp_path):
    lock = InstanceLock(tmp_path)
    assert lock.try_acquire()
    assert InstanceLock(tmp_path).holder_pid() == os.getpid()
    lock.release()


def test_a_dead_holder_leaves_no_stale_lock(tmp_path):
    """The whole point of an OS lock over a marker file: however the holder
    dies, the next bootstrap is not locked out by what it left behind."""
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "from pathlib import Path; from src.instance import InstanceLock;"
        "assert InstanceLock(Path(sys.argv[2])).try_acquire();"
        "import os; os._exit(0)"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(WRAPPER_ROOT), str(tmp_path)], check=True
    )
    lock = InstanceLock(tmp_path)
    assert lock.try_acquire(), "a lock outlived the process that took it"
    lock.release()


def test_taking_over_stops_the_holder_then_owns_the_folder(tmp_path):
    holder = InstanceLock(tmp_path)
    assert holder.try_acquire()
    holder.pid_path.write_text("4242", encoding="utf-8")
    stopped = []

    def terminate(pid):
        stopped.append(pid)
        holder.release()  # what the holder's death does
        return True

    mine = InstanceLock(tmp_path)
    assert take_over(mine, terminate, sleep=lambda _s: None) == 4242
    assert stopped == [4242]
    mine.release()


def test_a_free_folder_stops_nobody(tmp_path):
    lock = InstanceLock(tmp_path)
    assert take_over(lock, lambda pid: pytest.fail("stopped someone")) is None
    lock.release()


def test_a_holder_that_will_not_die_is_reported_not_waited_on_forever(tmp_path):
    holder = InstanceLock(tmp_path)
    assert holder.try_acquire()
    holder.pid_path.write_text("4242", encoding="utf-8")
    now = [0.0]

    def sleep(s):
        now[0] += s

    with pytest.raises(TimeoutError) as err:
        take_over(
            InstanceLock(tmp_path), lambda pid: False,
            timeout=2.0, sleep=sleep, clock=lambda: now[0],
        )
    assert err.value.args[0] == 4242
    holder.release()


def test_we_never_stop_ourselves(tmp_path):
    """A pid file pointing at us (pid reuse, a half-written file) must not make
    the bootstrap kill itself."""
    holder = InstanceLock(tmp_path)
    assert holder.try_acquire()
    holder.pid_path.write_text(str(os.getpid()), encoding="utf-8")
    now = [0.0]
    with pytest.raises(TimeoutError):
        take_over(
            InstanceLock(tmp_path), lambda pid: pytest.fail("killed itself"),
            timeout=1.0, sleep=lambda s: now.__setitem__(0, now[0] + s),
            clock=lambda: now[0],
        )
    holder.release()


# ---------------------------------------------------------- stale processes
def test_an_orphaned_server_and_its_real_interpreter_are_selected():
    """A venv python.exe is a launcher; the interpreter holding the venv's files
    is its CHILD and runs from the base Python — it has to go too."""
    procs = [
        ProcessInfo(1, 0, None),
        ProcessInfo(100, 1, _exe(VENV, "Scripts/python.exe")),  # launcher
        ProcessInfo(101, 100, BASE_PY),                          # interpreter
        ProcessInfo(500, 1, BASE_PY),                            # us
    ]
    assert stale_processes(procs, [VENV, UV], own_pid=500) == [100, 101]


def test_a_leftover_uv_install_is_selected():
    procs = [
        ProcessInfo(200, 1, _exe(UV, "uv.exe")),
        ProcessInfo(500, 1, BASE_PY),
    ]
    assert stale_processes(procs, [VENV, UV], own_pid=500) == [200]


def test_our_own_uv_run_parent_is_never_selected():
    """The host starts us through `.uv/uv.exe run` — stopping that kills us."""
    procs = [
        ProcessInfo(300, 1, _exe(UV, "uv.exe")),   # our parent
        ProcessInfo(500, 300, BASE_PY),            # us
        ProcessInfo(301, 1, _exe(UV, "uv.exe")),   # an old install's uv
    ]
    assert stale_processes(procs, [VENV, UV], own_pid=500) == [301]


def test_unrelated_processes_are_left_alone():
    """Only the runtime venv and uv count — the developer's .venv-test sits in
    the same wrapper folder and must survive."""
    other = VENV.parent / ".venv-test" / "Scripts" / "python.exe"
    sibling = str(VENV) + "-old" + os.sep + "python.exe"
    procs = [
        ProcessInfo(400, 1, str(other)),
        ProcessInfo(401, 1, sibling),
        ProcessInfo(402, 1, None),  # image unreadable (system process)
        ProcessInfo(500, 1, BASE_PY),
    ]
    assert stale_processes(procs, [VENV, UV], own_pid=500) == []


def test_matching_ignores_case_on_windows():
    if not instance.IS_WINDOWS:
        pytest.skip("case-insensitive paths are a Windows property")
    procs = [ProcessInfo(100, 1, str(VENV).upper() + r"\SCRIPTS\PYTHON.EXE")]
    assert stale_processes(procs, [VENV], own_pid=500) == [100]


@pytest.mark.skipif(not instance.IS_WINDOWS, reason="process listing is Windows-only")
def test_the_real_process_list_finds_ourselves():
    """The ctypes plumbing, against the live system: we must be in it, with our
    parent and our own image path."""
    procs = {p.pid: p for p in instance.list_processes()}
    me = procs[os.getpid()]
    assert me.ppid == os.getppid()
    # Not compared with sys.executable: in a venv that is the launcher's path,
    # while the image actually running is the base interpreter.
    assert me.exe and os.path.basename(me.exe).lower().startswith("python")


@pytest.mark.skipif(not instance.IS_WINDOWS, reason="Windows termination path")
def test_terminate_stops_a_real_process():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert instance.terminate(child.pid)
        assert child.wait(timeout=10) is not None
    finally:
        if child.poll() is None:
            child.kill()


# ------------------------------------------------------ bootstrap wiring
def _load_bootstrap():
    spec = importlib.util.spec_from_file_location(
        "bootstrap_instance_under_test", WRAPPER_ROOT / "bootstrap" / "bootstrap.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_bootstrap_claims_the_folder_before_touching_anything(monkeypatch):
    bootstrap = _load_bootstrap()
    order = []
    monkeypatch.setattr(bootstrap, "_install_kill_on_close_job", lambda: None)
    monkeypatch.setattr(bootstrap, "_claim_install_folder", lambda: order.append("claim"))

    def step_uv():
        order.append("uv")
        raise bootstrap.FatalError("stop here")

    monkeypatch.setattr(bootstrap, "step_uv", step_uv)
    monkeypatch.setattr(bootstrap.ndjson, "error", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap.ndjson, "starting", lambda *a, **k: None)
    assert bootstrap.main([]) == 1
    assert order == ["claim", "uv"]


def test_a_holder_that_will_not_die_is_a_readable_fatal_error(monkeypatch):
    bootstrap = _load_bootstrap()

    def refuse(*_a, **_k):
        raise TimeoutError(4242)

    monkeypatch.setattr(bootstrap.instance, "take_over", refuse)
    with pytest.raises(bootstrap.FatalError) as err:
        bootstrap._claim_install_folder()
    assert "4242" in str(err.value)


def test_the_sweep_only_looks_at_the_venv_and_uv(monkeypatch):
    bootstrap = _load_bootstrap()
    seen, stopped = {}, []
    monkeypatch.setattr(bootstrap.instance, "list_processes", lambda: [])

    def fake_select(procs, roots, own_pid):
        seen["roots"] = list(roots)
        seen["own"] = own_pid
        return [77] if not stopped else []

    monkeypatch.setattr(bootstrap.instance, "stale_processes", fake_select)
    monkeypatch.setattr(bootstrap.instance, "terminate", lambda pid: stopped.append(pid))
    monkeypatch.setattr(bootstrap.ndjson, "log", lambda *a, **k: None)
    bootstrap._stop_stale_processes(wait=0)
    assert seen["roots"] == [bootstrap.VENV_DIR, bootstrap.UV_DIR]
    assert seen["own"] == os.getpid()
    assert stopped == [77]


def test_a_failing_sweep_never_aborts_the_install(monkeypatch):
    bootstrap = _load_bootstrap()

    def boom():
        raise OSError("snapshot failed")

    monkeypatch.setattr(bootstrap.instance, "list_processes", boom)
    logged = []
    monkeypatch.setattr(bootstrap.ndjson, "log", lambda msg, *a, **k: logged.append(msg))
    bootstrap._stop_stale_processes()
    assert logged, "a failed sweep should at least say so"
