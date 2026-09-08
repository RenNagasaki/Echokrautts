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
    monkeypatch.setattr(bootstrap, "_existing_venv_problem", lambda *a: None)
    cleared, ran = [], []
    monkeypatch.setattr(bootstrap, "_clear_done", lambda name: cleared.append(name))
    monkeypatch.setattr(bootstrap, "_run_uv", lambda args, *_: ran.append(args))

    bootstrap.step_deps(Config(), _cpu_detection(bootstrap))

    assert cleared == []
    assert ran == []


def test_broken_existing_venv_drops_the_marker_and_reinstalls(bootstrap, monkeypatch):
    """Exactly the reported case: marker present, venv unusable."""
    monkeypatch.setattr(bootstrap, "_is_done", lambda *_: True)
    monkeypatch.setattr(bootstrap, "_existing_venv_problem", lambda *a: "f5-tts kaputt")
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

    monkeypatch.setattr(bootstrap, "_verify_venv", boom)
    problem = bootstrap._existing_venv_problem(Config(), "2.7.0")
    assert problem is not None
    assert "f5-tts" in problem


def test_missing_venv_is_a_problem_even_with_the_marker(bootstrap, monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap, "_venv_python", lambda: tmp_path / "gone" / "python.exe")
    assert bootstrap._existing_venv_problem(Config(), "2.7.0") == "venv fehlt"


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
