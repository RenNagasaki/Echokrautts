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


@pytest.fixture(autouse=True)
def _no_real_process_sweep(bootstrap, monkeypatch):
    """The retry loop sweeps leftover processes from the REAL wrapper venv —
    a test must never stop a server the developer has running."""
    monkeypatch.setattr(bootstrap, "_stop_stale_processes", lambda *a, **k: None)


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def state(bootstrap, monkeypatch, tmp_path):
    """Real marker files in a temp `.state` — the markers are the behaviour."""
    monkeypatch.setattr(bootstrap, "STATE_DIR", tmp_path / ".state")
    return tmp_path / ".state"


def _markers(state_dir):
    return sorted(p.stem for p in state_dir.glob("*.done")) if state_dir.exists() else []


def _quiet_install(bootstrap, monkeypatch, calls=None):
    """Stub everything that would touch uv, a venv or a subprocess."""
    calls = [] if calls is None else calls
    monkeypatch.setattr(bootstrap, "_run_uv", lambda args, *_: calls.append([str(a) for a in args]))
    monkeypatch.setattr(bootstrap, "_verify_group", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap.procutil, "run", lambda *a, **k: _Proc())
    monkeypatch.setattr(bootstrap, "_uv_path", lambda: Path("uv"))
    monkeypatch.setattr(bootstrap, "_venv_python", lambda *_a: Path("py"))
    monkeypatch.setattr(bootstrap.ndjson, "progress", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap.ndjson, "log", lambda *a, **k: None)
    return calls


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


def test_engine_install_carries_the_datasets_constraint(bootstrap, monkeypatch, state):
    """It must ride in the SAME resolution as f5-tts, not a later install.

    Installed afterwards it would be a second resolution, which can no longer
    influence the datasets version f5-tts already pulled in.
    """
    calls = _quiet_install(bootstrap, monkeypatch)
    config = Config(tts_backend="f5")
    bootstrap.step_deps(config, _cpu_detection(bootstrap))

    f5_install = [c for c in calls if "f5-tts" in c]
    assert f5_install, "f5-tts was never installed"
    assert config.datasets_constraint in f5_install[0]
    assert config.transformers_constraint in f5_install[0]


# --------------------------------------------------------------------------
# the stale marker
# --------------------------------------------------------------------------

def test_healthy_existing_venv_is_skipped_not_rebuilt(bootstrap, monkeypatch, state):
    """The repair must not turn every rerun into a reinstall."""
    calls = _quiet_install(bootstrap, monkeypatch)
    for name in ("deps-classic", "engine-f5"):
        bootstrap._mark_done(name)
    monkeypatch.setattr(bootstrap, "_foundation_problem", lambda *a: None)
    monkeypatch.setattr(bootstrap, "_engine_problem", lambda *a: None)

    bootstrap.step_deps(Config(tts_backend="f5"), _cpu_detection(bootstrap))

    assert calls == []
    assert _markers(state) == ["deps-classic", "engine-f5"]


def test_broken_existing_venv_drops_the_marker_and_reinstalls(bootstrap, monkeypatch, state):
    """Exactly the reported case: marker present, venv unusable."""
    calls = _quiet_install(bootstrap, monkeypatch)
    for name in ("deps-classic", "engine-f5", "engine-xtts"):
        bootstrap._mark_done(name)
    monkeypatch.setattr(bootstrap, "_foundation_problem", lambda *a: "torch kaputt")

    bootstrap.step_deps(Config(tts_backend="f5"), _cpu_detection(bootstrap))

    assert any(c[0] == "venv" for c in calls), "no rebuild happened"
    # The rebuilt venv holds only the active engine — XTTS's marker must not
    # survive into a venv that no longer contains it.
    assert _markers(state) == ["deps-classic", "engine-f5"]


def test_venv_problem_reports_instead_of_raising(bootstrap, monkeypatch, tmp_path):
    """A failing probe means 'reinstall', never an aborted bootstrap."""
    py = tmp_path / "python.exe"
    py.write_text("", encoding="utf-8")
    monkeypatch.setattr(bootstrap, "_venv_python", lambda *_a: py)

    def boom(*_a, **_k):
        raise bootstrap.FatalError("torch verification failed: installed 2.4.1. Delete .venv")

    monkeypatch.setattr(bootstrap, "_verify_torch", boom)
    problem = bootstrap._foundation_problem("classic", Config(), "2.7.0")
    assert problem and "torch" in problem


def test_missing_venv_is_a_problem_even_with_the_marker(bootstrap, monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap, "_venv_python", lambda *_a: tmp_path / "gone" / "python.exe")
    assert bootstrap._foundation_problem("classic", Config(), "2.7.0") == "venv missing"


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


def test_verify_group_runs_foundation_and_each_engine(bootstrap, monkeypatch):
    """Guards the composition: a verify that is not called protects nothing."""
    called = []
    monkeypatch.setattr(bootstrap, "_verify_torch", lambda *a, **k: called.append("torch"))
    monkeypatch.setattr(bootstrap, "_verify_transformers", lambda *a: called.append("transformers"))
    monkeypatch.setitem(bootstrap.ENGINE_VERIFIERS, "f5", lambda *a: called.append("f5"))
    monkeypatch.setitem(bootstrap.ENGINE_VERIFIERS, "moss", lambda *a: called.append("moss"))

    bootstrap._verify_group("py", Config(), "2.7.0", ["f5", "moss"])
    assert called == ["torch", "transformers", "f5", "moss"]


def test_every_engine_has_an_installer_and_a_verifier(bootstrap):
    """A new engine in src/engines.py without these would crash mid-install."""
    from src import engines

    assert set(bootstrap.ENGINE_INSTALLERS) == set(engines.KNOWN_BACKENDS)
    assert set(bootstrap.ENGINE_VERIFIERS) == set(engines.KNOWN_BACKENDS)


def test_moss_verifier_checks_both_runtimes(bootstrap, monkeypatch):
    called = []
    monkeypatch.setattr(bootstrap, "_verify_moss", lambda *a: called.append("moss"))
    monkeypatch.setattr(bootstrap, "_verify_moss_onnx", lambda *a: called.append("onnx"))
    bootstrap.ENGINE_VERIFIERS["moss"]("py", Config())
    assert called == ["moss", "onnx"]


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


def test_moss_runs_before_the_torch_re_pin(bootstrap, monkeypatch, state):
    """Its own torch pin can move torch; the re-pin is what puts it back."""
    order = []
    _quiet_install(bootstrap, monkeypatch)
    monkeypatch.setitem(bootstrap.ENGINE_INSTALLERS, "moss", lambda *a, **k: order.append("moss"))

    def record(args, *_):
        if any(str(a).startswith("torch==") for a in args):
            order.append("torch-pin")

    monkeypatch.setattr(bootstrap, "_run_uv", record)
    bootstrap.step_deps(Config(tts_backend="moss"), _cpu_detection(bootstrap))

    assert "moss" in order, "moss was never installed"
    assert "torch-pin" in order[order.index("moss") + 1:], "no re-pin after the moss install"




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

def test_a_missing_engine_is_reported_by_its_own_probe(bootstrap, monkeypatch):
    """An absent engine is an ENGINE problem (repair in place), not a
    foundation problem (rebuild)."""
    def missing_moss(*_a, **_k):
        raise bootstrap.FatalError("moss verification failed: No module named 'moss_tts_nano_runtime'")

    monkeypatch.setitem(bootstrap.ENGINE_VERIFIERS, "moss", missing_moss)
    assert "moss" in bootstrap._engine_problem("moss", "py", Config())


def test_a_wrong_torch_still_forces_a_rebuild(bootstrap, monkeypatch, tmp_path):
    """The foundation cannot be patched over: no amount of installing fixes a
    venv whose torch is the wrong build."""
    def bad_torch(*_a, **_k):
        raise bootstrap.FatalError("torch verification failed: installed 2.4.1")

    monkeypatch.setattr(bootstrap, "_verify_torch", bad_torch)
    monkeypatch.setattr(bootstrap, "_venv_python", lambda *_a: Path(__file__))  # exists
    assert bootstrap._foundation_problem("classic", Config(), "2.7.0")


def test_repair_installs_without_recreating_the_venv(bootstrap, monkeypatch, state):
    """The venv must NOT be deleted — that is what collided with the server."""
    _quiet_install(bootstrap, monkeypatch)
    for name in ("deps-classic", "engine-f5", "engine-moss"):
        bootstrap._mark_done(name)
    monkeypatch.setattr(bootstrap, "_foundation_problem", lambda *a: None)
    monkeypatch.setattr(bootstrap, "_engine_problem", lambda e, *a: "moss fehlt" if e == "moss" else None)
    created, installed = [], []
    monkeypatch.setattr(bootstrap, "_create_venv", lambda *a, **k: created.append(a))
    monkeypatch.setattr(bootstrap, "_install_engines", lambda *a, **k: installed.append(a[3]))

    bootstrap.step_deps(Config(tts_backend="f5"), _cpu_detection(bootstrap))

    assert created == [], "the venv must not be recreated for a broken engine"
    assert installed == [["moss"]], "only the broken engine is reinstalled"
    assert "deps-classic" in _markers(state), "the venv marker must be kept — the venv is fine"


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


def test_adding_an_engine_also_re_runs_its_model_step(bootstrap, monkeypatch, state):
    """A (re)installed engine may have no weights on disk yet.

    An install from before the engine existed has a `model.done` that the
    legacy migration turns into "weights present" for it. Without clearing, the
    download step is skipped and the server starts an engine with no model —
    which surfaces as transformers reinterpreting the missing path as a repo id
    ("Repo id must use alphanumeric chars"), naming neither cause nor fix.
    """
    _quiet_install(bootstrap, monkeypatch)
    for name in ("deps-classic", "engine-f5", "model-f5-de", "model-moss", "model-xtts"):
        bootstrap._mark_done(name)
    monkeypatch.setattr(bootstrap, "_foundation_problem", lambda *a: None)
    monkeypatch.setattr(bootstrap, "_engine_problem", lambda *a: None)
    monkeypatch.setattr(bootstrap, "_install_engines", lambda *a, **k: None)

    bootstrap.step_deps(Config(tts_backend="moss"), _cpu_detection(bootstrap))

    markers = _markers(state)
    assert "engine-moss" in markers
    assert "model-moss" not in markers, "the new engine's download must run again"
    assert {"model-f5-de", "model-xtts"} <= set(markers), "other engines' weights are untouched"


def test_a_healthy_install_does_not_re_download_models(bootstrap, monkeypatch, state):
    """The re-check is for upgrades, not for every start."""
    _quiet_install(bootstrap, monkeypatch)
    for name in ("deps-classic", "engine-xtts", "model-xtts"):
        bootstrap._mark_done(name)
    monkeypatch.setattr(bootstrap, "_foundation_problem", lambda *a: None)
    monkeypatch.setattr(bootstrap, "_engine_problem", lambda *a: None)

    bootstrap.step_deps(Config(tts_backend="xtts"), _cpu_detection(bootstrap))

    assert _markers(state) == ["deps-classic", "engine-xtts", "model-xtts"]


# ------------------------------------------------ voices are not our business

def test_the_native_install_does_not_fetch_voices(bootstrap, tmp_path, monkeypatch, state):
    """`step_model` downloads WEIGHTS, never the voice pack (decided 2026-09-10).

    The voices belong to the Echokraut plugin, which installs them itself, and a
    native install is always driven by that plugin — fetching them here would be
    a second party writing into the same folder. The container is the exception
    (no plugin, and an empty samples volume makes every /tts a 404), so
    `src/voicepack.py` stays and `docker/entrypoint.sh` calls it.

    Asserted on the commands actually issued, because the module still exists:
    nothing about its presence would catch the bootstrap calling it again.
    """
    called = _model_step(bootstrap, monkeypatch, tmp_path)
    bootstrap.step_model(Config(tts_backend="f5", models_dir=str(tmp_path)))

    modules = [c[c.index("-m") + 1] for c in called if "-m" in c]
    assert "src.voicepack" not in modules, "the native install must not fetch voices"
    assert "src.models" in modules, "…while the weights it IS responsible for still arrive"


def _model_step(bootstrap, monkeypatch, tmp_path):
    called: list[list[str]] = []
    monkeypatch.setattr(bootstrap, "_popen_forward", lambda cmd, env: called.append(list(cmd)) or 0)
    monkeypatch.setattr(bootstrap, "_venv_python", lambda *_a: tmp_path / "python.exe")
    monkeypatch.setattr(bootstrap.ndjson, "progress", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap.ndjson, "log", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap, "_server_env", lambda config: {})
    return called


@pytest.mark.parametrize("backend, modules", [
    ("f5", ["src.models"]),
    ("xtts", ["src.xtts_backend"]),
    ("moss", ["src.moss_backend", "src.moss_onnx_backend"]),
])
def test_only_the_active_engines_weights_are_downloaded(bootstrap, monkeypatch, tmp_path, state, backend, modules):
    """Lazy install: the install-everything era cost ~12 GB of weights per
    installation, whatever engine was used (user complaint, 2026-11-12)."""
    called = _model_step(bootstrap, monkeypatch, tmp_path)
    bootstrap.step_model(Config(tts_backend=backend, models_dir=str(tmp_path)))
    assert [c[c.index("-m") + 1] for c in called] == modules


def test_f5_downloads_only_the_active_language(bootstrap, monkeypatch, tmp_path, state):
    called = _model_step(bootstrap, monkeypatch, tmp_path)
    bootstrap.step_model(Config(tts_backend="f5", language="fr", models_dir=str(tmp_path)))
    assert called[0][-1] == "--active-language"
    assert _markers(state) == ["model-f5-fr"]


def test_a_language_switch_downloads_that_language(bootstrap, monkeypatch, tmp_path, state):
    bootstrap._mark_done("model-f5-de")
    called = _model_step(bootstrap, monkeypatch, tmp_path)
    bootstrap.step_model(Config(tts_backend="f5", language="de", models_dir=str(tmp_path)))
    assert called == [], "the active language is already there"
    bootstrap.step_model(Config(tts_backend="f5", language="en", models_dir=str(tmp_path)))
    assert len(called) == 1, "a new language is fetched on the switch"


def test_a_failed_optional_download_is_not_fatal(bootstrap, monkeypatch, tmp_path, state):
    """MOSS without its ONNX graphs still works (PyTorch fallback)."""
    called = _model_step(bootstrap, monkeypatch, tmp_path)
    monkeypatch.setattr(
        bootstrap, "_popen_forward",
        lambda cmd, env: called.append(cmd) or (1 if "src.moss_onnx_backend" in cmd else 0),
    )
    bootstrap.step_model(Config(tts_backend="moss", models_dir=str(tmp_path)))
    assert _markers(state) == ["model-moss"]


def test_a_failed_engine_download_is_fatal_and_leaves_no_marker(bootstrap, monkeypatch, tmp_path, state):
    _model_step(bootstrap, monkeypatch, tmp_path)
    monkeypatch.setattr(bootstrap, "_popen_forward", lambda cmd, env: 1)
    with pytest.raises(bootstrap.FatalError):
        bootstrap.step_model(Config(tts_backend="xtts", models_dir=str(tmp_path)))
    assert _markers(state) == []


# ------------------------------------------------------------ lazy install

def test_a_fresh_install_installs_only_the_active_engine(bootstrap, monkeypatch, state):
    calls = _quiet_install(bootstrap, monkeypatch)
    bootstrap.step_deps(Config(tts_backend="xtts"), _cpu_detection(bootstrap))

    flat = [a for c in calls for a in c]
    assert "coqui-tts" in flat
    assert "f5-tts" not in flat, "F5 must not be installed for an XTTS user"
    assert not any("MOSS-TTS-Nano" in a for a in flat), "MOSS must not be installed either"
    assert _markers(state) == ["deps-classic", "engine-xtts"]


def test_switching_engine_adds_it_to_the_existing_venv(bootstrap, monkeypatch, state):
    """A switch is a restart that installs what is missing — no new venv."""
    calls = _quiet_install(bootstrap, monkeypatch)
    for name in ("deps-classic", "engine-xtts"):
        bootstrap._mark_done(name)
    monkeypatch.setattr(bootstrap, "_foundation_problem", lambda *a: None)
    monkeypatch.setattr(bootstrap, "_engine_problem", lambda *a: None)
    verified = []
    monkeypatch.setattr(bootstrap, "_verify_group", lambda py, c, tv, ids, *_group: verified.append(ids))

    bootstrap.step_deps(Config(tts_backend="f5"), _cpu_detection(bootstrap))

    assert not any(c[0] == "venv" for c in calls), "the venv must be reused"
    assert any("f5-tts" in c for c in calls)
    assert not any("coqui-tts" in c for c in calls), "the healthy engine is left alone"
    # Re-verified as a whole: the new resolution may have moved a package
    # the already installed engine needed.
    assert verified == [["f5", "xtts"]]
    assert _markers(state) == ["deps-classic", "engine-f5", "engine-xtts"]


def test_legacy_markers_become_per_engine_markers(bootstrap, state):
    """An install from before lazy install neither rebuilds nor re-downloads."""
    bootstrap._mark_done("deps")
    bootstrap._mark_done("model")
    config = Config()

    bootstrap._migrate_legacy_markers(config)

    # Only the engines that existed then — Pocket was never on such a machine.
    expected = {"deps-classic", "engine-f5", "engine-xtts", "engine-moss", "model-xtts", "model-moss"}
    expected |= {f"model-f5-{lang}" for lang in config.languages}
    assert set(_markers(state)) == expected


def test_legacy_migration_is_a_no_op_without_old_markers(bootstrap, state):
    bootstrap._mark_done("engine-xtts")
    bootstrap._migrate_legacy_markers(Config())
    assert _markers(state) == ["engine-xtts"]


def test_the_server_runs_from_the_active_engines_venv(bootstrap, monkeypatch):
    seen = []
    monkeypatch.setattr(bootstrap, "_popen_forward", lambda cmd, env: seen.append(cmd) or 0)
    monkeypatch.setattr(bootstrap, "_server_env", lambda config: {})
    monkeypatch.setattr(bootstrap, "_venv_python", lambda group="classic": Path(f"venv-{group}/python"))
    monkeypatch.setattr(bootstrap.ndjson, "progress", lambda *a, **k: None)
    bootstrap.step_serve(Config(tts_backend="moss"))
    assert seen[0][0] == str(Path("venv-classic/python"))


def test_the_classic_group_keeps_the_historical_venv(bootstrap):
    """Every existing install's `.venv` stays valid — no migration, no download."""
    assert bootstrap._env_dir("classic") == bootstrap.VENV_DIR
    assert bootstrap._env_dir("qwen").name == ".venv-qwen"


# ------------------------------------------------- upgrading onto a new runtime

def test_a_venv_without_onnxruntime_is_repairable(bootstrap, monkeypatch):
    """The reported 0.0.1.4 upgrade: MOSS fell back to PyTorch on every start.

    An install predating the ONNX runtime carries valid markers, so without a
    probe the dependency step is skipped and onnxruntime never arrives. The
    engine then degrades to PyTorch, correctly and 2.5x slower, and only the log
    says why. It is an ENGINE problem → repaired in place, no torch rebuild.
    """
    monkeypatch.setattr(bootstrap, "_verify_moss", lambda *a: None)

    def missing_onnx(*_a, **_k):
        raise bootstrap.FatalError("moss onnx runtime missing: No module named 'onnxruntime'")

    monkeypatch.setattr(bootstrap, "_verify_moss_onnx", missing_onnx)
    assert bootstrap._engine_problem("moss", "py", Config())


def test_onnx_probe_is_skipped_when_switched_off(bootstrap, monkeypatch):
    """`moss_onnx_install.deps` empty means the user does not want it.

    Then its absence is a choice, not a defect, and must not make every start
    reinstall something the config just said to leave out.
    """
    called = []
    monkeypatch.setattr(bootstrap.procutil, "run", lambda *a, **k: called.append(a) or _Proc())
    bootstrap._verify_moss_onnx("py", Config(moss_onnx_install={"deps": []}))
    assert called == []


def test_onnx_probe_checks_the_vendor_module_too(bootstrap, monkeypatch):
    """The wheel alone proves nothing — the vendor's ONNX module has to import."""
    seen = []
    monkeypatch.setattr(bootstrap.procutil, "run", lambda cmd, *a, **k: seen.append(cmd) or _Proc())
    bootstrap._verify_moss_onnx("py", Config())
    assert seen, "the probe has to actually run"
    source = seen[0][-1]
    assert "onnxruntime" in source and "onnx_tts_runtime" in source


def test_a_sound_venv_without_marker_is_reused_not_rebuilt(bootstrap, monkeypatch, state):
    """Live, 2026-11-12: a first install whose ENGINE step failed left a venv
    with a good torch but no marker. The next start rebuilt it, re-downloading
    torch, and died on the Windows directory lock (os error 5)."""
    calls = _quiet_install(bootstrap, monkeypatch)
    monkeypatch.setattr(bootstrap, "_venv_python", lambda *_a: Path(__file__))  # exists
    monkeypatch.setattr(bootstrap, "_foundation_problem", lambda *a: None)

    bootstrap.step_deps(Config(tts_backend="f5"), _cpu_detection(bootstrap))

    assert not any(c[0] == "venv" for c in calls), "the sound venv must be reused"
    assert any("f5-tts" in c for c in calls)
    assert _markers(state) == ["deps-classic", "engine-f5"]


def test_a_broken_venv_without_marker_is_rebuilt(bootstrap, monkeypatch, state):
    calls = _quiet_install(bootstrap, monkeypatch)
    monkeypatch.setattr(bootstrap, "_venv_python", lambda *_a: Path(__file__))  # exists
    monkeypatch.setattr(bootstrap, "_foundation_problem", lambda *a: "torch kaputt")

    bootstrap.step_deps(Config(tts_backend="f5"), _cpu_detection(bootstrap))

    assert any(c[0] == "venv" for c in calls)


def test_xtts_brings_six_which_coqui_imports_but_does_not_declare(bootstrap, monkeypatch, state):
    """Found by the Windows install test: without f5-tts beside it, coqui's
    `import six` failed in an XTTS-only venv."""
    calls = _quiet_install(bootstrap, monkeypatch)
    bootstrap.step_deps(Config(tts_backend="xtts"), _cpu_detection(bootstrap))
    assert any("coqui-tts" in c and "six" in c for c in calls)
