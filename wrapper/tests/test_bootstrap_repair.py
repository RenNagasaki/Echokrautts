"""A finished install is not the same as a working one.

Two live failure reports drive these tests, both from installs that reported
success and then died later:

* an old ``datasets`` (pre-2.16) subclasses ``pyarrow.PyExtensionType``, which
  pyarrow removed — the venv installs cleanly and the first ``import
  f5_tts.api`` raises an AttributeError naming neither package;
* the same install had ``.state/deps.done`` from an earlier wrapper version, so
  every rerun skipped step 4 entirely and never got the chance to repair it.

So the contract pinned here is: the verifications run against an EXISTING venv
too, and a failing one drops the marker instead of being stepped over.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from src.config import Config

WRAPPER_ROOT = Path(__file__).resolve().parent.parent


def _load_bootstrap():
    spec = importlib.util.spec_from_file_location(
        "bootstrap_repair_under_test", WRAPPER_ROOT / "bootstrap" / "bootstrap.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bootstrap():
    return _load_bootstrap()


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _cpu_detection(bootstrap):
    return bootstrap.gpu_detect.Detection(
        backend="cpu",
        device="cpu",
        torch_index_url="https://example.invalid/cpu",
        detail="test",
    )


# --------------------------------------------------------------------------
# the datasets floor
# --------------------------------------------------------------------------

def test_datasets_constraint_defaults_past_pyextensiontype():
    """Anything below 2.16 is the reported crash; the default must exclude it."""
    constraint = Config().datasets_constraint
    assert constraint.startswith("datasets>=")
    floor = tuple(int(p) for p in constraint.split(">=")[1].split("."))
    assert floor >= (2, 16)


def test_engine_install_carries_the_datasets_constraint(bootstrap, monkeypatch):
    """It must ride in the SAME resolution as the engines, not a later install.

    Installed afterwards it would be a second resolution, which can no longer
    influence the datasets version f5-tts already pulled in.
    """
    calls = []
    monkeypatch.setattr(bootstrap, "_run_uv", lambda args, *_: calls.append(args))
    monkeypatch.setattr(bootstrap, "_verify_venv", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap, "_mark_done", lambda *_: None)
    monkeypatch.setattr(bootstrap, "_is_done", lambda *_: False)
    monkeypatch.setattr(bootstrap.procutil, "run", lambda *a, **k: _Proc())
    monkeypatch.setattr(bootstrap, "_uv_path", lambda: Path("uv"))
    monkeypatch.setattr(bootstrap, "_venv_python", lambda: Path("py"))

    config = Config()
    bootstrap.step_deps(config, _cpu_detection(bootstrap))

    engine_install = [
        [str(a) for a in call]
        for call in calls
        if str(WRAPPER_ROOT) in [str(a) for a in call]
    ]
    assert engine_install, "engine deps were never installed"
    assert config.datasets_constraint in engine_install[0]
    assert config.transformers_constraint in engine_install[0]


# --------------------------------------------------------------------------
# the stale marker
# --------------------------------------------------------------------------

def test_healthy_existing_venv_is_skipped_not_rebuilt(bootstrap, monkeypatch):
    """The repair must not turn every rerun into a reinstall."""
    monkeypatch.setattr(bootstrap, "_is_done", lambda *_: True)
    monkeypatch.setattr(bootstrap, "_existing_venv_problem", lambda *a: (None, False))
    cleared, ran = [], []
    monkeypatch.setattr(bootstrap, "_clear_done", lambda name: cleared.append(name))
    monkeypatch.setattr(bootstrap, "_run_uv", lambda args, *_: ran.append(args))

    bootstrap.step_deps(Config(), _cpu_detection(bootstrap))

    assert cleared == []
    assert ran == []


def test_broken_existing_venv_drops_the_marker_and_reinstalls(bootstrap, monkeypatch):
    """Exactly the reported case: marker present, venv unusable."""
    monkeypatch.setattr(bootstrap, "_is_done", lambda *_: True)
    monkeypatch.setattr(bootstrap, "_existing_venv_problem", lambda *a: ("torch kaputt", False))
    cleared, ran = [], []
    monkeypatch.setattr(bootstrap, "_clear_done", lambda name: cleared.append(name))
    monkeypatch.setattr(bootstrap, "_run_uv", lambda args, *_: ran.append(args))
    monkeypatch.setattr(bootstrap, "_verify_venv", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap, "_mark_done", lambda *_: None)
    monkeypatch.setattr(bootstrap.procutil, "run", lambda *a, **k: _Proc())
    monkeypatch.setattr(bootstrap, "_uv_path", lambda: Path("uv"))
    monkeypatch.setattr(bootstrap, "_venv_python", lambda: Path("py"))

    bootstrap.step_deps(Config(), _cpu_detection(bootstrap))

    assert cleared == ["deps"], "the stale marker must be removed"
    assert any("venv" in [str(a) for a in call] for call in ran), "no rebuild happened"


def test_venv_problem_reports_instead_of_raising(bootstrap, monkeypatch, tmp_path):
    """A failing probe means 'reinstall', never an aborted bootstrap."""
    py = tmp_path / "python.exe"
    py.write_text("", encoding="utf-8")
    monkeypatch.setattr(bootstrap, "_venv_python", lambda: py)

    def boom(*_a, **_k):
        raise bootstrap.FatalError("f5-tts verification failed: pyarrow. Delete .venv")

    monkeypatch.setattr(bootstrap, "_verify_torch", boom)
    problem, repairable = bootstrap._existing_venv_problem(Config(), "2.7.0")
    assert problem is not None
    assert "f5-tts" in problem


def test_missing_venv_is_a_problem_even_with_the_marker(bootstrap, monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap, "_venv_python", lambda: tmp_path / "gone" / "python.exe")
    assert bootstrap._existing_venv_problem(Config(), "2.7.0") == ("venv fehlt", False)


# --------------------------------------------------------------------------
# the F5 import check itself
# --------------------------------------------------------------------------

def test_verify_f5_accepts_a_working_import(bootstrap, monkeypatch):
    monkeypatch.setattr(bootstrap.procutil, "run", lambda *a, **k: _Proc())
    bootstrap._verify_f5("py")  # must not raise


def test_verify_f5_reports_the_pyarrow_crash(bootstrap, monkeypatch):
    """The message must carry the original error — that is the whole point."""
    monkeypatch.setattr(
        bootstrap.procutil,
        "run",
        lambda *a, **k: _Proc(
            1,
            stderr="AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'",
        ),
    )
    with pytest.raises(bootstrap.FatalError) as err:
        bootstrap._verify_f5("py")
    assert "PyExtensionType" in str(err.value)


def test_verify_venv_runs_the_f5_check_too(bootstrap, monkeypatch):
    """Guards the composition: a verify that is not called protects nothing."""
    called = []
    monkeypatch.setattr(bootstrap, "_verify_torch", lambda *a, **k: called.append("torch"))
    monkeypatch.setattr(bootstrap, "_verify_transformers", lambda *a: called.append("transformers"))
    monkeypatch.setattr(bootstrap, "_verify_f5", lambda *a: called.append("f5"))
    monkeypatch.setattr(bootstrap, "_verify_moss", lambda *a: called.append("moss"))

    bootstrap._verify_venv("py", Config(), "2.7.0")
    assert called == ["torch", "transformers", "f5", "moss"]


# --------------------------------------------------------------------------
# MOSS-TTS-Nano install shape
# --------------------------------------------------------------------------

def test_moss_is_pinned_to_a_commit_not_a_branch(bootstrap):
    """A moving `main` would change what users get with nothing changing here."""
    package = Config().moss_install["package"]
    assert package.startswith("https://github.com/OpenMOSS/MOSS-TTS-Nano/archive/")
    sha = package.rsplit("/", 1)[-1].removesuffix(".tar.gz")
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha), sha


def test_moss_uses_an_archive_not_git(bootstrap):
    """`git+https://` would need git on the user's machine; a one-click
    installer on Windows cannot assume that."""
    assert not Config().moss_install["package"].startswith("git+")


def test_moss_is_installed_no_deps_and_never_pins_torch(bootstrap, monkeypatch):
    """MOSS declares torch==2.7.0; with deps, pip would fetch it from PyPI and
    replace the CUDA build with a CPU one."""
    calls = []
    monkeypatch.setattr(bootstrap, "_run_uv", lambda args, *_: calls.append([str(a) for a in args]))
    config = Config()
    bootstrap._install_moss(config, "py", 4, "deps")

    package_cmd = [c for c in calls if "--no-deps" in c][0]
    assert config.moss_install["package"] in package_cmd
    assert not any(a.startswith("torch") for c in calls for a in c)
    assert not any("WeTextProcessing" in a for c in calls for a in c), "pynini has no Windows wheels"


def test_moss_install_can_be_switched_off(bootstrap, monkeypatch):
    monkeypatch.setattr(bootstrap, "_run_uv", lambda *a, **k: pytest.fail("must not install"))
    bootstrap._install_moss(Config(moss_install={}), "py", 4, "deps")


def test_moss_runs_before_the_torch_re_pin(bootstrap, monkeypatch):
    """Its own torch pin can move torch; the re-pin is what puts it back."""
    order = []
    monkeypatch.setattr(bootstrap, "_is_done", lambda *_: False)
    monkeypatch.setattr(bootstrap, "_mark_done", lambda *_: None)
    monkeypatch.setattr(bootstrap, "_verify_venv", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap, "_venv_python", lambda: Path("py"))
    monkeypatch.setattr(bootstrap, "_uv_path", lambda: Path("uv"))
    monkeypatch.setattr(bootstrap.procutil, "run", lambda *a, **k: _Proc())
    monkeypatch.setattr(bootstrap, "_install_moss", lambda *a, **k: order.append("moss"))

    def record(args, *_):
        if any(str(a).startswith("torch==") for a in args):
            order.append("torch-pin")

    monkeypatch.setattr(bootstrap, "_run_uv", record)
    bootstrap.step_deps(Config(), _cpu_detection(bootstrap))

    assert "moss" in order, "moss was never installed"
    assert "torch-pin" in order[order.index("moss") + 1:], "no re-pin after the moss install"


def test_verify_venv_checks_moss_too(bootstrap, monkeypatch):
    called = []
    monkeypatch.setattr(bootstrap, "_verify_torch", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap, "_verify_transformers", lambda *a: None)
    monkeypatch.setattr(bootstrap, "_verify_f5", lambda *a: None)
    monkeypatch.setattr(bootstrap, "_verify_moss", lambda *a: called.append("moss"))

    bootstrap._verify_venv("py", Config(), "2.7.0")
    assert called == ["moss"]


def test_verify_moss_imports_the_runtime_module(bootstrap, monkeypatch):
    """MOSS ships its runtime as a TOP-LEVEL module, not inside its package —
    importing the package alone would prove nothing."""
    seen = {}

    def run(cmd, *a, **k):
        seen["code"] = cmd[-1]
        return _Proc()

    monkeypatch.setattr(bootstrap.procutil, "run", run)
    bootstrap._verify_moss("py", Config())
    assert "moss_tts_nano_runtime" in seen["code"]


def test_verify_moss_reports_a_missing_package(bootstrap, monkeypatch):
    monkeypatch.setattr(
        bootstrap.procutil, "run",
        lambda *a, **k: _Proc(1, stderr="ModuleNotFoundError: No module named 'moss_tts_nano_runtime'"),
    )
    with pytest.raises(bootstrap.FatalError) as err:
        bootstrap._verify_moss("py", Config())
    assert "moss_tts_nano_runtime" in str(err.value)


# --------------------------------------------------------------------------
# Upgrading an existing install
#
# All of this comes from one real 0.0.1.0 upgrade that failed: the installed
# venv simply lacked the newly added engine, the marker repair decided to
# rebuild, and `uv venv --clear` died with "Zugriff verweigert (os error 5)" on
# .venv\Scripts — leaving the install broken rather than merely un-upgraded.
# --------------------------------------------------------------------------

def test_a_missing_engine_is_repaired_in_place(bootstrap, monkeypatch):
    """The ordinary upgrade must not cost a multi-gigabyte torch download.

    torch and transformers verified fine; only an engine package was absent.
    Rebuilding would throw away a working foundation to add 14 MB.
    """
    monkeypatch.setattr(bootstrap, "_verify_torch", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap, "_verify_transformers", lambda *a: None)
    monkeypatch.setattr(bootstrap, "_verify_f5", lambda *a: None)

    def missing_moss(*_a, **_k):
        raise bootstrap.FatalError("moss verification failed: No module named 'moss_tts_nano_runtime'")

    monkeypatch.setattr(bootstrap, "_verify_moss", missing_moss)
    monkeypatch.setattr(bootstrap, "_venv_python", lambda: Path(__file__))  # exists

    problem, repairable = bootstrap._existing_venv_problem(Config(), "2.7.0")
    assert problem and repairable is True


def test_a_wrong_torch_still_forces_a_rebuild(bootstrap, monkeypatch):
    """The foundation cannot be patched over: no amount of installing fixes a
    venv whose torch is the wrong build."""
    def bad_torch(*_a, **_k):
        raise bootstrap.FatalError("torch verification failed: installed 2.4.1")

    monkeypatch.setattr(bootstrap, "_verify_torch", bad_torch)
    monkeypatch.setattr(bootstrap, "_venv_python", lambda: Path(__file__))

    problem, repairable = bootstrap._existing_venv_problem(Config(), "2.7.0")
    assert problem and repairable is False


def test_repair_installs_without_recreating_the_venv(bootstrap, monkeypatch):
    """The venv must NOT be deleted — that is what collided with the server."""
    monkeypatch.setattr(bootstrap, "_is_done", lambda *_: True)
    monkeypatch.setattr(bootstrap, "_existing_venv_problem", lambda *a: ("moss fehlt", True))
    monkeypatch.setattr(bootstrap, "_venv_python", lambda: Path("py"))
    monkeypatch.setattr(bootstrap, "_verify_venv", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap, "_mark_done", lambda *_: None)
    created, installed = [], []
    monkeypatch.setattr(bootstrap, "_create_venv", lambda *a, **k: created.append(a))
    monkeypatch.setattr(bootstrap, "_install_engines", lambda *a, **k: installed.append(a))
    monkeypatch.setattr(bootstrap, "_clear_done", lambda *_: pytest.fail("marker must be kept"))

    bootstrap.step_deps(Config(), _cpu_detection(bootstrap))

    assert created == [], "the venv must not be recreated for a missing engine"
    assert installed, "the engines were never installed"


def test_a_locked_venv_directory_is_retried(bootstrap, monkeypatch):
    """Windows cannot delete a directory whose files are open, and the likely
    holder is the server that stopped a second ago."""
    calls = []

    def flaky(args, *_a, **_k):
        calls.append(args)
        if len(calls) < 3:
            raise bootstrap.FatalError(
                "uv venv failed: failed to remove directory `.venv/Scripts`: "
                "Zugriff verweigert (os error 5)"
            )

    monkeypatch.setattr(bootstrap, "_run_uv", flaky)
    monkeypatch.setattr(bootstrap.time, "sleep", lambda _s: None)
    monkeypatch.setattr(bootstrap.ndjson, "progress", lambda *a, **k: None)

    bootstrap._create_venv("3.11", 4, "deps")
    assert len(calls) == 3, "it must keep trying while the handle is released"


def test_a_permanently_locked_venv_says_what_to_do(bootstrap, monkeypatch):
    """After the retries, the message has to be actionable — the original just
    quoted an errno, which tells a user nothing."""
    def always_locked(*_a, **_k):
        raise bootstrap.FatalError("failed to remove directory: Zugriff verweigert (os error 5)")

    monkeypatch.setattr(bootstrap, "_run_uv", always_locked)
    monkeypatch.setattr(bootstrap.time, "sleep", lambda _s: None)
    monkeypatch.setattr(bootstrap.ndjson, "progress", lambda *a, **k: None)

    with pytest.raises(bootstrap.FatalError) as err:
        bootstrap._create_venv("3.11", 4, "deps", attempts=2)
    message = str(err.value)
    assert "Server" in message, "must name the likely cause"
    assert "os error 5" in message, "must keep the original error"


def test_an_unrelated_venv_failure_is_not_retried(bootstrap, monkeypatch):
    """Retrying something that is not a lock just delays the real error."""
    calls = []

    def broken(args, *_a, **_k):
        calls.append(args)
        raise bootstrap.FatalError("uv venv failed: no such python version")

    monkeypatch.setattr(bootstrap, "_run_uv", broken)
    monkeypatch.setattr(bootstrap.time, "sleep", lambda _s: pytest.fail("must not wait"))

    with pytest.raises(bootstrap.FatalError):
        bootstrap._create_venv("3.11", 4, "deps")
    assert len(calls) == 1


def test_the_retry_budget_outlasts_a_shutting_down_server(bootstrap, monkeypatch):
    """Measured, not guessed: five tries over ~16 s did NOT outlast the holder in
    a real failure, while the same command succeeded once more time had passed.
    The budget has to be in the tens of seconds, not a handful."""
    waits = []
    monkeypatch.setattr(bootstrap.time, "sleep", lambda s: waits.append(s))
    monkeypatch.setattr(bootstrap.ndjson, "progress", lambda *a, **k: None)
    monkeypatch.setattr(
        bootstrap, "_run_uv",
        lambda *a, **k: (_ for _ in ()).throw(
            bootstrap.FatalError("failed to remove directory: Zugriff verweigert (os error 5)")
        ),
    )

    with pytest.raises(bootstrap.FatalError):
        bootstrap._create_venv("3.11", 4, "deps")

    assert sum(waits) >= 45, f"only waited {sum(waits):.0f}s in total"


def test_a_free_directory_costs_no_waiting(bootstrap, monkeypatch):
    """The budget must not slow down the normal case."""
    monkeypatch.setattr(bootstrap, "_run_uv", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap.time, "sleep", lambda _s: pytest.fail("must not wait"))
    bootstrap._create_venv("3.11", 4, "deps")
