"""Plug-and-play bootstrap for the F5-TTS wrapper (SPEC §3).

Idempotent, fixed 6-step sequence. Each step announces itself and its completion
as NDJSON ``progress`` events on stdout so the host can render "Step X/6 …".
Steps already done (markers in ``<wrapper>/.state``) are reported as ``skipped``.

Steps:
  1. Obtain ``uv``         → <wrapper>/.uv
  2. Pin Python            → uv python install <version>
  3. Detect GPU backend    → cache to .state/detection.json
  4. venv + dependencies   → torch from backend index, then project deps
  5. Preload F5-TTS model  → into <wrapper>/models (HF cache)
  6. Start the server      → uvicorn on host:port, emits the ``ready`` event

This script must run on the *initial* interpreter (system or uv-managed) and
therefore imports only the standard library plus the wrapper's light, pure
modules (``ndjson``, ``config``, ``gpu_detect``, ``procutil``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import tarfile
import urllib.request
import zipfile
from pathlib import Path

# Make the wrapper's ``src`` package importable regardless of cwd.
WRAPPER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WRAPPER_ROOT))

from src import engines, gpu_detect, i18n, instance, ndjson, procutil  # noqa: E402
from src.config import load_config  # noqa: E402

TOTAL_STEPS = 6
STATE_DIR = WRAPPER_ROOT / ".state"
UV_DIR = WRAPPER_ROOT / ".uv"
VENV_DIR = WRAPPER_ROOT / ".venv"


def _env_dir(group: str = engines.CLASSIC_GROUP) -> Path:
    """The venv of a group. ``classic`` keeps the historical ``.venv`` so every
    existing install stays valid; any other group gets ``.venv-<group>``."""
    return VENV_DIR if group == engines.CLASSIC_GROUP else WRAPPER_ROOT / f".venv-{group}"

UV_RELEASE = "https://github.com/astral-sh/uv/releases/latest/download"
UV_ASSETS = {
    "win": ("uv-x86_64-pc-windows-msvc.zip", "uv.exe"),
    "linux": ("uv-x86_64-unknown-linux-gnu.tar.gz", "uv"),
}


class FatalError(Exception):
    """Unrecoverable bootstrap failure → NDJSON error(fatal) + non-zero exit."""


# Keep the job-object handle alive for the whole bootstrap lifetime; if it is
# garbage-collected the job closes early and kills everything prematurely.
_JOB_HANDLE = None


def _install_kill_on_close_job() -> None:
    """Bind this process (and all descendants) to a Windows job that is killed
    when the bootstrap dies — for ANY reason: Ctrl+C, console window closed,
    taskkill, crash. This is the robust way to guarantee the server (and its
    VRAM) can never be orphaned; the cross-platform parent-PID watchdog (in the
    server) proved unreliable on Windows due to the python launcher stub and PID
    reuse. No-op on non-Windows (the watchdog covers Linux)."""
    global _JOB_HANDLE
    if not procutil.IS_WINDOWS:
        return
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    JobObjectExtendedLimitInformation = 9

    # CRITICAL on x64: declare HANDLE returns/args, else ctypes assumes 32-bit
    # int and truncates the job handle → every later job call fails silently.
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD
    ]
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.GetCurrentProcess.restype = wintypes.HANDLE

    class BASIC(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IO(ctypes.Structure):
        _fields_ = [(f"c{i}", ctypes.c_ulonglong) for i in range(6)]

    class EXT(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BASIC),
            ("IoInfo", IO),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    try:
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return
        info = EXT()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(
            job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
        ):
            return
        if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
            # Already in a non-nestable job, or access denied → rely on watchdog.
            ndjson.log("orphan protection: job assignment failed, using watchdog", "warning")
            return
        _JOB_HANDLE = job  # keep handle open for the process lifetime
        ndjson.log("orphan protection: kill-on-close job active")
    except OSError as exc:
        ndjson.log(f"orphan protection: job setup failed ({exc}), using watchdog", "warning")
        return


# ----------------------------------------------------------------- ownership
# Held for the bootstrap's whole life, serving included (module global so it is
# never garbage-collected and released early). See src/instance.py for the
# tester log that made this necessary.
_INSTANCE_LOCK = instance.InstanceLock(STATE_DIR)


def _claim_install_folder() -> None:
    """Become the only bootstrap in this folder, then clear out what an earlier
    one left running. A second bootstrap used to go straight to ``uv venv
    --clear`` on a venv the first was still installing into, and wrecked it."""
    try:
        stopped = instance.take_over(_INSTANCE_LOCK, instance.terminate)
    except TimeoutError as exc:
        raise FatalError(i18n.t("boot.takeover_failed", pid=exc.args[0])) from exc
    if stopped is not None:
        ndjson.log(
            i18n.t("boot.took_over", pid=stopped),
            level="warning",
        )
    _stop_stale_processes()


def _stop_stale_processes(wait: float = 10.0) -> None:
    """Stop everything still running from our venv or our uv.

    Safe only because we hold the instance lock: no other bootstrap owns this
    folder, so whatever runs from it is a leftover — an orphaned server, the
    ``uv`` of a killed install. On Windows those are exactly what keeps
    ``.venv\\Scripts`` undeletable. Never raises: a failed sweep leaves the
    retry loop in ``_create_venv`` to do what it did before.
    """
    try:
        pids = instance.stale_processes(
            instance.list_processes(),
            [*(_env_dir(g) for g in engines.GROUPS), UV_DIR], os.getpid()
        )
        if not pids:
            return
        ndjson.log(
            i18n.t("boot.stale_stopped", count=len(pids), pids=", ".join(map(str, pids))),
            level="warning",
        )
        for pid in pids:
            instance.terminate(pid)
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            alive = {p.pid for p in instance.list_processes()}
            if not alive.intersection(pids):
                return
            time.sleep(0.25)
    except Exception as exc:  # noqa: BLE001 — best effort, see docstring
        ndjson.log(i18n.t("boot.stale_failed", error=exc), level="warning")


# --------------------------------------------------------------------- state
def _marker(name: str) -> Path:
    return STATE_DIR / f"{name}.done"


def _is_done(name: str) -> bool:
    return _marker(name).exists()


def _clear_done(name: str) -> None:
    _marker(name).unlink(missing_ok=True)


def _deps_marker(group: str) -> str:
    """venv of ``group`` built and its foundation (torch, transformers) verified."""
    return f"deps-{group}"


def _engine_marker(engine_id: str) -> str:
    """Engine package installed into its group's venv and verified importable."""
    return f"engine-{engine_id}"


def _model_marker(spec: "engines.EngineSpec", language: str, config=None) -> str:
    """Weights on disk — per language for engines with one model per language,
    and per variant (``spec.model_variant``, e.g. Pocket's layers + revision;
    long values such as a commit are cut to 12 characters)."""
    marker = f"model-{spec.id}-{language}" if spec.per_language else f"model-{spec.id}"
    if config is not None:
        for field in spec.model_variant:
            marker += f"-{str(getattr(config, field))[:12]}"
    return marker


def _clear_model_markers(spec: "engines.EngineSpec") -> None:
    if spec.per_language:
        for marker in STATE_DIR.glob(f"model-{spec.id}-*.done"):
            marker.unlink(missing_ok=True)
    else:
        _clear_done(_model_marker(spec, ""))


LEGACY_ENGINES = ("f5", "xtts", "moss")


def _migrate_legacy_markers(config) -> None:
    """Translate the markers of the install-everything era (before 2026-11-12).

    ``deps.done`` meant "the one venv holds every engine", ``model.done`` "every
    engine's weights and every F5 language are on disk". Both are rewritten into
    the per-group/per-engine markers they imply, so an existing install neither
    rebuilds nor re-downloads. They are only evidence of a FINISHED install, as
    always: ``step_deps`` still verifies every engine it now believes installed,
    and repairs (and re-checks the weights of) whichever does not import — which
    covers an old install from before an engine existed.
    """
    # Exactly the engines the install-everything era installed — NOT whatever
    # the classic group holds today: an engine added since (Pocket) was never on
    # those machines, and claiming its weights would start it without a model.
    classic = [engines.spec_for(e) for e in LEGACY_ENGINES]
    if _is_done("deps"):
        _mark_done(_deps_marker(engines.CLASSIC_GROUP))
        for spec in classic:
            _mark_done(_engine_marker(spec.id))
        _clear_done("deps")
    if _is_done("model"):
        for spec in classic:
            if spec.per_language:
                for language in config.languages:
                    _mark_done(_model_marker(spec, language))
            else:
                _mark_done(_model_marker(spec, ""))
        _clear_done("model")


def _foundation_problem(group: str, config, torch_version: str) -> str | None:
    """What is wrong with the venv of ``group`` itself, or None.

    The foundation is what no engine install can fix — a wrong torch (or one
    that drags FFmpeg back in via torchcodec), a transformers outside the
    group's pin. Such a venv has to be rebuilt. Never raises: every failure here
    means "fix it", and a surprise from the probe must not abort the bootstrap
    that would have done the fixing.
    """
    py = _venv_python(group)
    if not py.exists():
        return "venv missing"
    return _problem_of(_verify_foundation, str(py), config, torch_version, group)


def _engine_problem(engine_id: str, py: str, config) -> str | None:
    """What is wrong with one installed engine, or None. Never raises.

    A missing or broken engine is repaired IN PLACE — installing a package costs
    megabytes where a rebuild costs a multi-gigabyte torch download and has to
    delete a directory a just-stopped server may still hold open (the first
    0.0.1.0 upgrade died exactly there).
    """
    return _problem_of(ENGINE_VERIFIERS[engine_id], py, config)


def _problem_of(check, *args) -> str | None:
    """Run a raising ``_verify_*`` check and turn its failure into a short reason."""
    try:
        check(*args)
    except FatalError as exc:
        return str(exc).split(".")[0]
    except OSError as exc:  # venv present but unusable (permissions, half-deleted)
        return f"venv not executable: {exc}"
    return None


def _mark_done(name: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _marker(name).write_text("ok", encoding="utf-8")


# ----------------------------------------------------------------- download
def _download(url: str, dest: Path, index: int, step: str, message: str) -> None:
    """Download ``url`` to ``dest`` emitting periodic percent progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_pct = -5

    def hook(blocks: int, block_size: int, total: int) -> None:
        nonlocal last_pct
        if total > 0:
            pct = min(100, int(blocks * block_size * 100 / total))
            if pct - last_pct >= 5:
                last_pct = pct
                ndjson.progress(index, TOTAL_STEPS, step, message, percent=pct)

    try:
        urllib.request.urlretrieve(url, dest, reporthook=hook)  # noqa: S310 (HTTPS)
    except OSError as exc:
        raise FatalError(i18n.t("boot.download_failed", url=url, error=exc)) from exc


# ------------------------------------------------------------- platform bits
def _platform_key() -> str:
    if sys.platform.startswith("win"):
        return "win"
    if sys.platform.startswith("linux"):
        return "linux"
    raise FatalError(i18n.t("boot.unsupported_platform", platform=sys.platform))


def _uv_path() -> Path:
    exe = "uv.exe" if _platform_key() == "win" else "uv"
    return UV_DIR / exe


def _venv_python(group: str = engines.CLASSIC_GROUP) -> Path:
    venv = _env_dir(group)
    if _platform_key() == "win":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _run_uv(args: list[str], index: int, step: str) -> None:
    cmd = [str(_uv_path()), *args]
    proc = procutil.run(cmd, timeout=None)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        raise FatalError(f"uv {' '.join(args[:2])} failed: {' | '.join(tail)}")


# --------------------------------------------------------------------- steps
def step_uv() -> None:
    index, step = 1, "uv"
    if _uv_path().exists() or _is_done(step):
        ndjson.progress(index, TOTAL_STEPS, step, i18n.t("step.uv_present"), skipped=True)
        return
    # Use a uv already on PATH if present (still record so reinstall is fast).
    on_path = shutil.which("uv")
    if on_path:
        UV_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(on_path, _uv_path())
        _mark_done(step)
        ndjson.progress(index, TOTAL_STEPS, step, i18n.t("step.uv_from_path"), done=True)
        return

    ndjson.progress(index, TOTAL_STEPS, step, i18n.t("step.uv_fetching"))
    key = _platform_key()
    asset, inner = UV_ASSETS[key]
    archive = STATE_DIR / asset
    _download(f"{UV_RELEASE}/{asset}", archive, index, step, i18n.t("step.uv_downloading"))

    UV_DIR.mkdir(parents=True, exist_ok=True)
    if asset.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(UV_DIR)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(UV_DIR)  # noqa: S202 (trusted release tarball)
        # Linux tarball nests the binary in a sub-directory.
        for cand in UV_DIR.rglob(inner):
            if cand != _uv_path():
                shutil.move(str(cand), str(_uv_path()))
            break
    if key != "win":
        os.chmod(_uv_path(), 0o755)
    archive.unlink(missing_ok=True)
    _mark_done(step)
    ndjson.progress(index, TOTAL_STEPS, step, i18n.t("step.uv_installed"), done=True)


def step_python(version: str) -> None:
    index, step = 2, "python"
    if _is_done(step):
        ndjson.progress(index, TOTAL_STEPS, step, i18n.t("step.python_present", version=version), skipped=True)
        return
    ndjson.progress(index, TOTAL_STEPS, step, i18n.t("step.python_installing", version=version))
    _run_uv(["python", "install", version], index, step)
    _mark_done(step)
    ndjson.progress(index, TOTAL_STEPS, step, i18n.t("step.python_ready", version=version), done=True)


def step_detect(config) -> gpu_detect.Detection:
    index, step = 3, "detect"
    cache = STATE_DIR / "detection.json"
    ndjson.progress(index, TOTAL_STEPS, step, i18n.t("step.detecting"))
    det = gpu_detect.detect_backend(config)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {
                "backend": det.backend,
                "device": det.device,
                "torch_index_url": det.torch_index_url,
                "extra_packages": det.extra_packages,
                "max_workers_hint": det.max_workers_hint,
                "detail": det.detail,
            }
        ),
        encoding="utf-8",
    )
    ndjson.progress(index, TOTAL_STEPS, step, det.detail, done=True)
    return det


def _create_venv(
    python_version: str, index: int, step: str, attempts: int = 8, target: Path = VENV_DIR
) -> None:
    r"""Create (or replace) the venv, tolerating a directory Windows still holds.

    ``--clear`` replaces any pre-existing venv — e.g. one an earlier ``uv run``
    accidentally synced from pyproject — instead of failing "already exists".

    The retries are not defensive padding. On Windows a directory cannot be
    removed while any file in it is open, and the two things most likely to hold
    ``.venv/Scripts/python.exe`` are the server that was running a second ago and
    the verification subprocesses this very step just ran. A real 0.0.1.0 upgrade
    died exactly here: ``failed to remove directory .venv\Scripts: Zugriff
    verweigert (os error 5)`` — and it left the install broken, because the venv
    was already partly gone. Waiting a few seconds costs nothing when the handle
    is about to be released, and when it is not, the final message says what to
    do instead of quoting an errno.
    """
    # MEASURED, not guessed: a first attempt failed 50 ms in and five tries over
    # ~16 s did not outlast the holder, while the very same command succeeded in
    # 0.145 s once more time had passed. Whatever holds it — a server unloading
    # CUDA, an antivirus scanner walking the freshly written files — takes longer
    # than a handful of seconds. Eight attempts stretch to roughly a minute,
    # which costs nothing when the handle is already free (the first try
    # succeeds) and saves a broken installation when it is not.
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            _run_uv(["venv", str(target), "--clear", "--python", python_version], index, step)
            return
        except FatalError as exc:
            locked = "os error 5" in str(exc) or "Zugriff verweigert" in str(exc)                 or "Access is denied" in str(exc) or "being used by another process" in str(exc)
            if not locked or attempt == attempts:
                if locked:
                    raise FatalError(i18n.t("boot.venv_locked", error=exc)) from exc
                raise
            # Says what is happening NOW (waiting), not what is about to
            # happen — the previous wording announced the next attempt before
            # sleeping, which read like it had already started and made the
            # gaps in the log look like hangs.
            ndjson.progress(
                index, TOTAL_STEPS, step,
                i18n.t("step.venv_locked_wait", delay=f"{delay:.0f}", attempt=attempt, attempts=attempts),
            )
            time.sleep(delay)
            # Something may have started from the venv since the claim (a
            # verification subprocess of this very step, a server a host just
            # launched) — we own the folder, so it goes.
            _stop_stale_processes()
            delay *= 1.5


def _install_engines(config, det, py: str, engine_ids: list[str], index: int, step: str) -> None:
    """Install torch + the given engines into an EXISTING venv.

    One sequence for the fresh install AND the in-place addition of an engine —
    the order is load-bearing (engines before the torch re-pin), and a second
    copy of it would drift the first time somebody touched one of them. Only the
    engines asked for are installed (lazy install, 2026-11-12): a switch to
    another engine is a restart that installs that one.
    """
    torch_version = det.torch_version or config.torch_version
    torch_pin = f"torch=={torch_version}"
    audio_pin = f"torchaudio=={det.torchaudio_version or config.torchaudio_version}"

    if det.wheel_urls:
        # ROCm on Windows needs its runtime SDK in the venv before torch, and it
        # is installed from wheel URLs too (AMD publishes no index for Windows).
        ndjson.progress(
            index, TOTAL_STEPS, step, i18n.t("step.rocm_runtime"), percent=5
        )
        _run_uv(["pip", "install", "--python", py, *det.wheel_urls], index, step)

    def install_torch(pct: int) -> None:
        ndjson.progress(
            index, TOTAL_STEPS, step, i18n.t("step.torch", backend=det.backend), percent=pct
        )
        if det.torch_wheel_urls:
            _run_uv(["pip", "install", "--python", py, *det.torch_wheel_urls], index, step)
            return
        _run_uv(
            ["pip", "install", "--python", py, torch_pin, audio_pin,
             "--index-url", det.torch_index_url],
            index, step,
        )

    install_torch(10)

    # The wrapper itself (fastapi, uvicorn, soundfile, …) plus transformers inside
    # the group's pin. transformers is installed explicitly, not left to an
    # engine: MOSS needs it but does not declare it (installed --no-deps), and
    # /asr-check needs it under every engine. The pin rides along with EVERY
    # engine install below, so a later addition cannot move it either.
    ndjson.progress(index, TOTAL_STEPS, step, i18n.t("step.wrapper_deps"), percent=30)
    group = engines.spec_for(engine_ids[0]).group
    _run_uv(
        ["pip", "install", "--python", py, str(WRAPPER_ROOT), _group_transformers(config, group)],
        index, step,
    )

    # Every engine goes in BEFORE the re-pin below, on purpose: their deps can
    # move torch (MOSS declares torch==2.7.0, f5-tts drags torchcodec), and the
    # re-pin is what puts it back. Installed after, any drift would survive.
    for engine_id in engine_ids:
        name = engines.spec_for(engine_id).name
        ndjson.progress(index, TOTAL_STEPS, step, i18n.t("step.installing", name=name), percent=50)
        ENGINE_INSTALLERS[engine_id](config, py, index, step)

    # Re-pin torch to undo any bump from the engine install, then drop the
    # unused torchcodec (audio loads via torchaudio.load → soundfile on the
    # pinned 2.7.x; torchcodec would re-introduce the FFmpeg requirement).
    install_torch(75)
    ndjson.progress(index, TOTAL_STEPS, step, i18n.t("step.torchcodec_remove"), percent=85)
    procutil.run([str(_uv_path()), "pip", "uninstall", "--python", py, "torchcodec"])

    for extra in det.extra_packages:
        ndjson.progress(index, TOTAL_STEPS, step, i18n.t("step.installing", name=extra), percent=90)
        # The torch pin travels WITH the extra. `torch-directml` declares
        # `torch==2.4.1`, so installed on its own it just moved torch and the
        # verification below failed the install (every released version, every
        # AMD/Windows machine). With the pin in the same resolution the extra
        # either agrees with the venv or the install fails HERE, where the
        # message names the package that disagreed. The backend index comes
        # along as an EXTRA index: the extras themselves live on PyPI, which
        # `--index-url` would replace outright.
        _run_uv(
            ["pip", "install", "--python", py, extra, torch_pin, audio_pin,
             "--extra-index-url", det.torch_index_url],
            index, step,
        )


def _install_f5(config, py: str, index: int, step: str) -> None:
    # The datasets floor rides along in the SAME resolution as the transformers
    # pin: f5-tts leaves `datasets` unconstrained, and a pre-2.16 pick installs
    # fine and then dies on `import f5_tts.api` with a pyarrow AttributeError
    # (see config.datasets_constraint).
    _run_uv(
        ["pip", "install", "--python", py, "f5-tts",
         config.transformers_constraint, config.datasets_constraint],
        index, step,
    )


def _install_xtts(config, py: str, index: int, step: str) -> None:
    # The maintained coqui-tts fork. Its own `transformers>=4.57` has no upper
    # bound and would pull a 5.x that crashes XTTS at model-load
    # (`isin_mps_friendly`), hence the pin in the same resolution. `six` is
    # imported by coqui (TTS/tts/utils/text/punctuation.py) but not declared —
    # it used to arrive through f5-tts, so an XTTS-only venv lacked it (found by
    # the Windows install test, 2026-11-12).
    _run_uv(
        ["pip", "install", "--python", py, "coqui-tts", "six", config.transformers_constraint],
        index, step,
    )


def _install_qwen(config, py: str, index: int, step: str) -> None:
    # faster-qwen3-tts (CUDA graphs) with ITS transformers pin in the same
    # resolution — the group's reason to exist: >= 5.15.1 (qwen-tts-hf), and 5.22
    # breaks the tokenizer encoder, hence exact.
    spec = config.qwen_install or {}
    _run_uv(
        ["pip", "install", "--python", py,
         *[str(p) for p in spec.get("packages") or []], _group_transformers(config, engines.QWEN_GROUP)],
        index, step,
    )


def _install_pocket(config, py: str, index: int, step: str) -> None:
    # Kyutai's package; needs no transformers, and the classic pin rides along
    # so its resolution cannot move it.
    _run_uv(
        ["pip", "install", "--python", py,
         *[str(p) for p in (config.pocket_install or {}).get("packages") or []],
         config.transformers_constraint],
        index, step,
    )


def _group_transformers(config, group: str) -> str:
    """The transformers requirement every install into ``group`` carries."""
    if group == engines.QWEN_GROUP:
        return str(config.qwen_install["transformers"])
    return config.transformers_constraint


def step_deps(config, det: gpu_detect.Detection) -> None:
    """Make sure the ACTIVE engine is installed and works — and nothing more.

    Per venv group a marker for the venv (foundation) and per engine one for the
    package. Every run re-verifies what the markers claim (a marker means an
    install finished, not that it still works): a broken foundation rebuilds the
    venv, a broken or missing engine is (re)installed in place.
    """
    index, step = 4, "deps"
    spec = engines.spec_for(config.tts_backend)
    # Before a single byte is downloaded: Qwen on a machine without a usable
    # GPU would be gigabytes for an engine that cannot run there.
    reason = engines.unsupported_device_reason(spec, det.backend)
    if reason:
        raise FatalError(reason)
    group = spec.group
    torch_version = det.torch_version or config.torch_version
    py = str(_venv_python(group))
    group_engines = [s.id for s in engines.engines_in(group)]

    has_marker = _is_done(_deps_marker(group))
    # Without the marker the venv may still be there with a sound foundation —
    # typically a first install whose ENGINE step failed. Rebuilding it would
    # re-download torch and, on Windows, collide with whatever still holds
    # `.venv\Scripts` (live, 2026-11-12: the retry after a failed XTTS install
    # died with os error 5 after a minute of waiting). So: reuse it.
    problem = (
        _foundation_problem(group, config, torch_version)
        if has_marker or _venv_python(group).exists()
        else "venv fehlt"
    )
    if problem is None:
        installed = [e for e in group_engines if _is_done(_engine_marker(e))]
        todo = [e for e in installed if _engine_problem(e, py, config)]
        if spec.id not in installed:
            todo.append(spec.id)
        if not todo and has_marker:
            ndjson.progress(index, TOTAL_STEPS, step, i18n.t("step.deps_present"), skipped=True)
            return
        names = ", ".join(engines.spec_for(e).name for e in todo)
        ndjson.log(i18n.t("boot.install_extend", names=names))
        _install_engines(config, det, py, todo, index, step)
        # Everything this venv claims to hold, not just what was added: a
        # new engine's resolution can move a package an older one needed.
        _verify_group(py, config, torch_version, sorted(set(installed) | set(todo)), group)
        _mark_done(_deps_marker(group))
        _mark_engines_installed(todo)
        ndjson.progress(index, TOTAL_STEPS, step, i18n.t("step.deps_extended"), done=True)
        return
    if has_marker:
        # The foundation itself is wrong (torch), which no amount of installing
        # fixes. Leaving the markers would hand the broken venv to the model step,
        # which then fails somewhere that explains nothing.
        ndjson.log(
            i18n.t("boot.install_rebuild", problem=problem),
            level="warning",
        )
        _clear_done(_deps_marker(group))
        for engine_id in group_engines:
            _clear_done(_engine_marker(engine_id))

    ndjson.progress(index, TOTAL_STEPS, step, i18n.t("step.venv_create"))
    # A backend may demand its own interpreter: AMD's native-Windows ROCm wheels
    # are cp312-only, while everything else runs on the configured 3.11. uv
    # fetches a missing Python itself, so this needs no extra step.
    python_version = det.python_version or config.python_version
    _create_venv(python_version, index, step, target=_env_dir(group))
    _install_engines(config, det, py, [spec.id], index, step)
    # Verify BEFORE marking done: an engine's resolution can leave a PyPI CPU
    # torch + torchcodec behind, which loads fine and crashes on the first
    # ``torchaudio.load``. Fail loudly so no marker caches a broken venv.
    _verify_group(py, config, torch_version, [spec.id], group)
    _mark_done(_deps_marker(group))
    _mark_engines_installed([spec.id])
    ndjson.progress(index, TOTAL_STEPS, step, i18n.t("step.deps_installed"), done=True)


def _mark_engines_installed(engine_ids: list[str]) -> None:
    """Set the engine markers — and drop their model markers.

    A (re)installed engine may be one whose weights were never downloaded: an
    install from before the engine existed carries a ``model.done`` that the
    legacy migration turned into "weights present" for it. Re-running its
    download is idempotent and costs a few HTTP checks for what is there; not
    re-running it once started a server with no model ("Repo id must use
    alphanumeric chars … models\\moss_tts_nano").
    """
    for engine_id in engine_ids:
        _mark_done(_engine_marker(engine_id))
        _clear_model_markers(engines.spec_for(engine_id))


def _install_moss(config, py: str, index: int, step: str) -> None:
    """Install the MOSS-TTS-Nano engine from a pinned source archive.

    Three deliberate choices, each of which would otherwise bite:

    * **A GitHub archive URL, not ``git+https://``.** MOSS publishes no PyPI
      package, and the git form would need git installed on the user's machine —
      which a one-click installer cannot assume on Windows.
    * **A COMMIT, not a branch.** A moving ``main`` would change what users get
      without a single line changing in this repo.
    * **``--no-deps``.** MOSS declares ``torch==2.7.0``; installed with deps, pip
      would fetch that from PyPI and replace the CUDA/ROCm build with a CPU one.
      The re-pin that follows in :func:`step_deps` owns torch. Its other
      dependencies are already in this venv, so only the genuinely missing ones
      are listed in ``config.moss_install["deps"]``.

    ``WeTextProcessing`` from MOSS's requirements.txt is deliberately NOT
    installed: it needs ``pynini``, which has no Windows wheels, and it is
    optional — the import is lazy and there are only zh/en normalizers anyway.
    """
    spec = config.moss_install or {}
    package = spec.get("package")
    if not package:  # explicitly disabled by config → engine simply unavailable
        return
    ndjson.progress(index, TOTAL_STEPS, step, i18n.t("step.installing", name="MOSS-TTS-Nano"), percent=65)
    _run_uv(["pip", "install", "--python", py, "--no-deps", str(package)], index, step)
    deps = [str(d) for d in (spec.get("deps") or [])]
    if deps:
        _run_uv(["pip", "install", "--python", py, *deps], index, step)
    # onnxruntime is MOSS's faster runtime (same weights, ~2.5x on the CPU) and
    # is installed here rather than as a MOSS dep, because MOSS is installed
    # --no-deps and its own requirements never reach the venv.
    onnx_deps = [str(d) for d in ((config.moss_onnx_install or {}).get("deps") or [])]
    if onnx_deps:
        ndjson.progress(index, TOTAL_STEPS, step, i18n.t("step.installing", name="ONNX Runtime"), percent=70)
        _run_uv(["pip", "install", "--python", py, *onnx_deps], index, step)


def _verify_moss(py: str, config) -> None:
    """Assert the MOSS runtime imports — before any marker is written.

    Because the install is ``--no-deps``, a missing package does not fail the
    install; it fails the first time somebody selects the backend, long after the
    install "worked". The import checked here is the top-level module the worker
    actually loads, not the package name — MOSS ships its runtime as a top-level
    module rather than inside its package, so importing the package alone would
    prove nothing.
    """
    if not (config.moss_install or {}).get("package"):
        return
    proc = procutil.run([py, "-c", "from moss_tts_nano_runtime import NanoTTSService"])
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        raise FatalError(
            "moss verification failed: " + " | ".join(tail) + ". "
            "Check config.moss_install (it installs --no-deps on purpose); "
            "delete .venv and .state/deps-classic.done and rerun."
        )


def _verify_moss_onnx(py: str, config) -> None:
    """Assert MOSS's ONNX runtime is installed — the FASTER of its two runtimes.

    Missing, the engine still works: it falls back to PyTorch and says so. That
    is the right behaviour at request time and the wrong thing to rely on at
    install time, because the fallback is roughly 2.5x slower and silent to
    anyone not reading the log.

    This is what makes an UPGRADE repair itself. An install from before ONNX
    existed has a valid `deps.done`, so without a probe that notices the missing
    package the whole dependency step is skipped and the package never arrives —
    the same shape of bug as the 0.0.1.0 upgrade, where an older marker skipped
    the step that would have installed a newly added engine. Reported as
    REPAIRABLE: the venv is fine, it is one small package short.

    Both imports matter: the wheel alone is not enough, the vendor's ONNX module
    has to load too.
    """
    if not ((config.moss_onnx_install or {}).get("deps")):
        return  # switched off by config → the engine simply uses PyTorch
    proc = procutil.run([py, "-c", "import onnxruntime, onnx_tts_runtime"])
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        raise FatalError(
            "moss onnx runtime missing: " + " | ".join(tail) + ". "
            "It is reinstalled on the next start."
        )


def _verify_foundation(py: str, config, torch_version: str, group: str = engines.CLASSIC_GROUP) -> None:
    """The group-wide assertions: pinned torch without torchcodec, and the
    group's transformers — classic: still below 5 (XTTS's ``isin_mps_friendly``
    is the canary); qwen: exactly the pinned 5.x."""
    _verify_torch(py, config, expected_version=torch_version)
    if group == engines.QWEN_GROUP:
        _verify_transformers_version(py, _group_transformers(config, group))
    else:
        _verify_transformers(py)


def _verify_transformers_version(py: str, requirement: str) -> None:
    """Assert an exact ``transformers==X`` pin held (other forms are not checked)."""
    if "==" not in requirement:
        return
    expected = requirement.split("==", 1)[1].strip()
    proc = procutil.run([py, "-c", "import transformers; print(transformers.__version__)"])
    found = (proc.stdout or "").strip().splitlines()[-1:] if proc.returncode == 0 else []
    if found != [expected]:
        raise FatalError(
            f"transformers verification failed: expected {expected}, found "
            f"{found[0] if found else (proc.stderr or '').strip()[-200:]}. The venv is rebuilt."
        )


def _verify_group(
    py: str, config, torch_version: str, engine_ids: list[str], group: str = engines.CLASSIC_GROUP
) -> None:
    """Every install assertion for a venv and the engines it holds — raises.

    Runs before any marker is written, so a bad resolve is never cached.
    """
    _verify_foundation(py, config, torch_version, group)
    for engine_id in engine_ids:
        ENGINE_VERIFIERS[engine_id](py, config)


def _verify_import(py: str, label: str, statement: str) -> None:
    """Assert ``statement`` runs in the venv; the failure carries the real error."""
    proc = procutil.run([py, "-c", statement])
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        raise FatalError(
            f"{label} verification failed: " + " | ".join(tail) + ". "
            "It is reinstalled on the next start."
        )


def _verify_xtts(py: str, config) -> None:
    """Assert coqui's XTTS model code imports (what the worker loads first)."""
    _verify_import(py, "xtts", "from TTS.tts.models.xtts import Xtts")


def _verify_qwen(py: str, config) -> None:
    """Assert faster-qwen3-tts imports (it pulls qwen-tts-hf + transformers 5)."""
    _verify_import(py, "qwen", "from faster_qwen3_tts import FasterQwen3TTS")


def _verify_pocket(py: str, config) -> None:
    """Assert Kyutai's pocket-tts imports (CPU engine)."""
    _verify_import(py, "pocket", "from pocket_tts import TTSModel")


def _verify_moss_all(py: str, config) -> None:
    _verify_moss(py, config)
    _verify_moss_onnx(py, config)


def _verify_f5(py: str, config=None) -> None:
    """Assert the F5 engine imports — the resolution has more ways to rot than torch.

    Reported live: an old ``datasets`` (pre-2.16) subclasses
    ``pyarrow.PyExtensionType``, which pyarrow removed, so the install succeeds
    and the FIRST use dies with "module 'pyarrow' has no attribute
    'PyExtensionType'" from a traceback whose frames name neither the wrapper
    nor the pinned package. ``config.datasets_constraint`` prevents that
    particular resolution; this import catches whatever the next one is,
    including in a venv that predates the constraint.
    """
    proc = procutil.run([py, "-c", "import f5_tts.api"])
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        raise FatalError(
            "f5-tts verification failed: " + " | ".join(tail) + ". "
            "Delete .venv and .state/deps-classic.done and rerun."
        )


def _verify_torch(py: str, config, expected_version: str | None = None) -> None:
    """Assert the venv ended up with the pinned, torchcodec-free torch.

    ``expected_version`` lets a backend override the configured pin — AMD's
    native-Windows ROCm build only exists as torch 2.9.1, so comparing against
    ``config.torch_version`` there would fail a correct install. torchcodec must
    still be absent either way: ``audio_compat`` keeps ``torchaudio.load``
    decoding through soundfile, so torchcodec would only re-introduce the FFmpeg
    dependency the wrapper avoids.
    """
    expected = expected_version or config.torch_version
    code = (
        "import json, importlib.util as u, torch;"
        "print(json.dumps({'v': torch.__version__,"
        " 'codec': u.find_spec('torchcodec') is not None}))"
    )
    proc = procutil.run([py, "-c", code])
    if proc.returncode != 0:
        raise FatalError(f"torch verification failed to run: {proc.stderr.strip()}")
    try:
        info = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise FatalError(f"torch verification: unparseable output {proc.stdout!r}") from exc
    base = str(info.get("v", "")).split("+")[0]
    if base != expected or info.get("codec"):
        raise FatalError(
            f"torch verification failed: installed={info!r}; "
            f"expected torch=={expected} and no torchcodec. "
            "The project install re-introduced an incompatible torch/torchcodec; "
            "delete .venv and .state/deps-classic.done and rerun."
        )


def _verify_transformers(py: str) -> None:
    """Assert transformers still provides the symbol XTTS needs.

    coqui-tts imports ``transformers.pytorch_utils.isin_mps_friendly``, removed in
    transformers 5.x. If the resolve slipped a 5.x past the pin, XTTS would import
    fine but crash at model-load — so fail HERE (before any marker) so the next run
    rebuilds instead of freezing a venv that only breaks when you pick the XTTS
    backend (SPEC §3 idempotency must not cache a half-broken venv)."""
    proc = procutil.run([py, "-c", "from transformers.pytorch_utils import isin_mps_friendly"])
    if proc.returncode != 0:
        raise FatalError(
            "transformers verification failed: XTTS requires "
            "transformers.pytorch_utils.isin_mps_friendly (removed in transformers "
            "5.x). Constrain transformers<5 (config.transformers_constraint); "
            "delete .venv and .state/deps-classic.done and rerun."
        )


# Per engine: how it is installed into its group's venv, and how that install
# is proven to work. Keys must be exactly ``engines.KNOWN_BACKENDS`` (a test
# holds them to it).
ENGINE_INSTALLERS = {
    "f5": _install_f5, "xtts": _install_xtts, "moss": _install_moss,
    "pocket": _install_pocket,
    "qwen-0.6b": _install_qwen, "qwen-1.7b": _install_qwen,
}
ENGINE_VERIFIERS = {
    "f5": _verify_f5, "xtts": _verify_xtts, "moss": _verify_moss_all,
    "pocket": _verify_pocket,
    "qwen-0.6b": _verify_qwen, "qwen-1.7b": _verify_qwen,
}


def _popen_forward(cmd: list[str], env: dict) -> int:
    """Run a venv command, forwarding its stdout line-by-line to ours.

    Used for long child runs (model download, server) so their NDJSON reaches
    the host live and the pipe is continuously drained (SPEC §13.1). stderr is
    inherited so tracebacks flow straight through.
    """
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        cwd=str(WRAPPER_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        encoding="utf-8",
        bufsize=1,
        **procutil.NO_WINDOW_KWARGS,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
    except KeyboardInterrupt:
        # The child is detached (CREATE_NO_WINDOW) and won't receive the console
        # Ctrl+C, so terminate it explicitly instead of leaving it orphaned.
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise
    return proc.wait()


def step_model(config) -> None:
    """Download the weights of the ACTIVE engine (F5: of the active language).

    The engine is chosen at start, and a switch is a restart that downloads the
    next one (lazy install, 2026-11-12) — the install-everything era cost
    ~12 GB of weights per installation, whatever engine was used.
    """
    index, step = 5, "model"
    spec = engines.spec_for(config.tts_backend)
    marker = _model_marker(spec, config.language, config)
    if _is_done(marker):
        ndjson.progress(index, TOTAL_STEPS, step, i18n.t("step.models_present"), skipped=True)
        return
    py = str(_venv_python(spec.group))
    env = _server_env(config)
    # Let the download sub-processes emit their per-file progress bars onto THIS
    # same "Step 5/6 · model" bar (src.progress.ModelProgress reads these).
    env["F5W_STEP_INDEX"] = str(index)
    env["F5W_STEP_TOTAL"] = str(TOTAL_STEPS)
    ndjson.progress(index, TOTAL_STEPS, step, i18n.t("step.model_loading", name=spec.name), percent=0)
    cmd = [py, "-m", spec.model_module]
    if spec.per_language:
        cmd.append("--active-language")
    if _popen_forward(cmd, env) != 0:
        raise FatalError(i18n.t("boot.model_failed", name=spec.name))
    # NOT fatal (MOSS's ONNX graphs): the engine falls back to its slower
    # runtime and says so, and a failed optional download must not cost a user
    # an otherwise working install.
    for module in spec.optional_model_modules:
        ndjson.progress(index, TOTAL_STEPS, step, i18n.t("step.extra_model", module=module), percent=80)
        if _popen_forward([py, "-m", module], env) != 0:
            ndjson.log(
                i18n.t("boot.extra_model_failed", module=module, name=spec.name),
                level="warning",
            )
    # NO voice pack here, on purpose (2026-09-10). The voices belong to the
    # Echokraut plugin, which installs them itself — a native install is always
    # driven by that plugin, so fetching them here would be a second party
    # writing into the same folder. `src/voicepack.py` stays for the CONTAINER,
    # where there is no plugin and an empty samples volume means every /tts is
    # a 404; `docker/entrypoint.sh` calls it.
    _mark_done(marker)
    ndjson.progress(index, TOTAL_STEPS, step, i18n.t("step.models_loaded"), percent=100, done=True)


def _env_value(value) -> str:
    """Serialize one resolved config value into its ``F5W_*`` string form.

    The inverse of ``config._coerce``: bools as true/false, lists comma-joined,
    dicts as JSON, ``None`` as the empty string (which _coerce maps back to None
    for the nullable fields).
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def _config_env(config) -> dict:
    """Every resolved config field as an ``F5W_*`` variable.

    The server runs as a *subprocess* that re-runs ``load_config`` from scratch,
    so anything the host passed to bootstrap.py on the command line is invisible
    to it unless forwarded. This used to be a hand-written list of a few fields,
    which silently dropped every other flag: ``--xtts-fp16 true`` reached the
    bootstrap but not the server, which then read ``xtts_fp16: false`` from
    config.json and ran without fp16 while ``/health`` honestly reported it off.
    Forwarding the whole resolved config removes the class of bug — the server
    can no longer disagree with the config the bootstrap resolved.
    """
    from dataclasses import fields as _fields

    from src.config import ENV_PREFIX

    return {
        ENV_PREFIX + f.name.upper(): _env_value(getattr(config, f.name))
        for f in _fields(config)
    }


def _server_env(config) -> dict:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = str(WRAPPER_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    config.models_path.mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(config.models_path)
    env["HF_HUB_CACHE"] = str(config.models_path)
    if config.hf_endpoint:
        env["HF_ENDPOINT"] = config.hf_endpoint
    # Propagate the resolved config so the server subprocess honors every flag
    # the host passed to bootstrap.py, not just config.json defaults.
    env.update(_config_env(config))
    # The server watches this pid and self-exits when it dies, so it can never be
    # orphaned (closed console window, killed bootstrap, etc.). An explicit
    # --parent-pid (e.g. the C# host) wins; otherwise watch bootstrap itself.
    # Set AFTER the bulk forwarding, which would otherwise write a bare "".
    env["F5W_PARENT_PID"] = str(config.parent_pid or os.getpid())
    # Both engines are installed; the XTTS env is harmless for the F5 backend.
    # Accept the CPML non-interactively and keep XTTS weights under models/.
    env["COQUI_TOS_AGREED"] = "1"
    env["TTS_HOME"] = str(config.models_path)
    return env


def step_serve(config) -> int:
    index, step = 6, "serve"
    ndjson.progress(index, TOTAL_STEPS, step, i18n.t("step.server_start", host=config.host, port=config.port))
    env = _server_env(config)
    cmd = [
        str(_venv_python(engines.spec_for(config.tts_backend).group)), "-m", "uvicorn",
        "src.server:create_app", "--factory",
        "--host", config.host, "--port", str(config.port),
        "--log-level", "warning",
    ]
    # Forward the server's stdout line-by-line (NOT a bare inherited fd: a child
    # writing to an inherited, block-buffered pipe swallowed the ``ready`` event
    # the host blocks on). Keeps bootstrap.py as the stable parent the host's
    # Process handle tracks.
    return _popen_forward(cmd, env)


# ---------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ndjson.starting()
    # Guarantee no orphaned server/VRAM if we die for any reason (SPEC §13.2).
    _install_kill_on_close_job()
    try:
        config = load_config(argv)
        _claim_install_folder()
        step_uv()
        step_python(config.python_version)
        det = step_detect(config)
        _migrate_legacy_markers(config)
        step_deps(config, det)
        step_model(config)
        return step_serve(config)
    except FatalError as exc:
        ndjson.error(str(exc), fatal=True)
        return 1
    except KeyboardInterrupt:
        ndjson.shutdown()
        return 0
    except Exception as exc:  # noqa: BLE001 — top-level guard (SPEC §13.2)
        import traceback

        traceback.print_exc(file=sys.stderr)
        ndjson.error(i18n.t("boot.unexpected", error=exc), fatal=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
