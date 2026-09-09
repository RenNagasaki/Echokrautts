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

from src import gpu_detect, ndjson, procutil  # noqa: E402
from src.config import load_config  # noqa: E402

TOTAL_STEPS = 6
STATE_DIR = WRAPPER_ROOT / ".state"
UV_DIR = WRAPPER_ROOT / ".uv"
VENV_DIR = WRAPPER_ROOT / ".venv"

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


# --------------------------------------------------------------------- state
def _marker(name: str) -> Path:
    return STATE_DIR / f"{name}.done"


def _is_done(name: str) -> bool:
    return _marker(name).exists()


def _clear_done(name: str) -> None:
    _marker(name).unlink(missing_ok=True)


def _existing_venv_problem(config, torch_version: str) -> tuple[str | None, bool]:
    """(what is wrong with the installed venv, can it be repaired in place?).

    Returns ``(None, False)`` when it is fine. The second flag is the important
    part and was learned from a real upgrade: a venv whose **torch** is wrong has
    to be rebuilt, but one that merely LACKS AN ENGINE does not — installing the
    missing package costs a few megabytes where a rebuild costs a multi-gigabyte
    torch download, and it does not have to delete a directory that a
    just-stopped server may still hold open.

    Never raises: every failure mode here means "fix it", so a surprise from the
    probe itself must not abort the bootstrap that would have done the fixing.
    """
    py = _venv_python()
    if not py.exists():
        return "venv fehlt", False
    try:
        # The foundation. Wrong torch (or a torchcodec that drags FFmpeg back in)
        # cannot be patched over — that venv has to go.
        _verify_torch(str(py), config, expected_version=torch_version)
        _verify_transformers(str(py))
    except FatalError as exc:
        return str(exc).split(".")[0], False
    except OSError as exc:  # venv present but unusable (permissions, half-deleted)
        return f"venv nicht ausführbar: {exc}", False
    try:
        # The engines. Missing or broken ones are re-installable in place; this
        # is the ordinary "upgraded to a version that added an engine" case.
        _verify_f5(str(py))
        _verify_moss(str(py), config)
    except FatalError as exc:
        return str(exc).split(".")[0], True
    except OSError as exc:
        return f"venv nicht ausführbar: {exc}", False
    return None, False


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
        raise FatalError(f"download failed: {url} ({exc})") from exc


# ------------------------------------------------------------- platform bits
def _platform_key() -> str:
    if sys.platform.startswith("win"):
        return "win"
    if sys.platform.startswith("linux"):
        return "linux"
    raise FatalError(f"unsupported platform: {sys.platform} (Windows/Linux only)")


def _uv_path() -> Path:
    exe = "uv.exe" if _platform_key() == "win" else "uv"
    return UV_DIR / exe


def _venv_python() -> Path:
    if _platform_key() == "win":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


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
        ndjson.progress(index, TOTAL_STEPS, step, "uv vorhanden", skipped=True)
        return
    # Use a uv already on PATH if present (still record so reinstall is fast).
    on_path = shutil.which("uv")
    if on_path:
        UV_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(on_path, _uv_path())
        _mark_done(step)
        ndjson.progress(index, TOTAL_STEPS, step, "uv von PATH übernommen", done=True)
        return

    ndjson.progress(index, TOTAL_STEPS, step, "Beschaffe uv …")
    key = _platform_key()
    asset, inner = UV_ASSETS[key]
    archive = STATE_DIR / asset
    _download(f"{UV_RELEASE}/{asset}", archive, index, step, "Lade uv …")

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
    ndjson.progress(index, TOTAL_STEPS, step, "uv installiert", done=True)


def step_python(version: str) -> None:
    index, step = 2, "python"
    if _is_done(step):
        ndjson.progress(index, TOTAL_STEPS, step, f"Python {version}", skipped=True)
        return
    ndjson.progress(index, TOTAL_STEPS, step, f"Installiere Python {version} …")
    _run_uv(["python", "install", version], index, step)
    _mark_done(step)
    ndjson.progress(index, TOTAL_STEPS, step, f"Python {version} bereit", done=True)


def step_detect(config) -> gpu_detect.Detection:
    index, step = 3, "detect"
    cache = STATE_DIR / "detection.json"
    ndjson.progress(index, TOTAL_STEPS, step, "Erkenne GPU/Backend …")
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


def _create_venv(python_version: str, index: int, step: str, attempts: int = 5) -> None:
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
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            _run_uv(["venv", str(VENV_DIR), "--clear", "--python", python_version], index, step)
            return
        except FatalError as exc:
            locked = "os error 5" in str(exc) or "Zugriff verweigert" in str(exc)                 or "Access is denied" in str(exc) or "being used by another process" in str(exc)
            if not locked or attempt == attempts:
                if locked:
                    raise FatalError(
                        "Das venv-Verzeichnis ist gesperrt und konnte nicht ersetzt werden. "
                        "Meist läuft noch ein EchokrauTTS-Server oder ein Virenscanner hält die "
                        "Dateien. Beende den laufenden Server (oder starte den Rechner neu) und "
                        "versuche es erneut. Ursprungsfehler: " + str(exc)
                    ) from exc
                raise
            ndjson.progress(
                index, TOTAL_STEPS, step,
                f"venv-Verzeichnis noch gesperrt, neuer Versuch {attempt + 1}/{attempts} …",
            )
            time.sleep(delay)
            delay *= 1.5


def _install_engines(config, det, py: str, index: int, step: str) -> None:
    """Install torch + every engine into an EXISTING venv.

    Split out so the fresh install and the in-place repair run the SAME steps
    in the same order — the order is load-bearing (engines before the torch
    re-pin), and a second copy of it would drift the first time somebody
    touched one of them.
    """
    torch_version = det.torch_version or config.torch_version
    torch_pin = f"torch=={torch_version}"
    audio_pin = f"torchaudio=={det.torchaudio_version or config.torchaudio_version}"

    if det.wheel_urls:
        # ROCm on Windows needs its runtime SDK in the venv before torch, and it
        # is installed from wheel URLs too (AMD publishes no index for Windows).
        ndjson.progress(
            index, TOTAL_STEPS, step, "Installiere ROCm-Laufzeit (Windows) …", percent=5
        )
        _run_uv(["pip", "install", "--python", py, *det.wheel_urls], index, step)

    def install_torch(pct: int) -> None:
        ndjson.progress(
            index, TOTAL_STEPS, step, f"Installiere PyTorch ({det.backend}) …", percent=pct
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

    # Install ALL engines so the active one is chosen at start (--tts-backend),
    # never re-installed on switch: the wrapper project (→ f5-tts + fastapi etc.)
    # AND the maintained coqui-tts fork (→ XTTS-v2), in a SINGLE uv resolution so
    # uv finds one mutually-compatible dependency set (or fails loudly) rather
    # than two sequential installs stomping each other's shared deps.
    # The transformers constraint is part of the SAME resolution so uv picks a
    # transformers that BOTH engines accept (XTTS needs a <5 build for
    # `isin_mps_friendly`; coqui-tts's own `>=4.57` has no upper bound and would
    # otherwise drag in a 5.x that crashes XTTS at model-load).
    ndjson.progress(index, TOTAL_STEPS, step, "Installiere Engine-Abhängigkeiten (F5 + XTTS) …", percent=50)
    # The datasets floor rides along in the SAME resolution for the same reason
    # as the transformers pin: f5-tts leaves `datasets` unconstrained, and a
    # pre-2.16 pick installs fine and then dies on `import f5_tts.api` with a
    # pyarrow AttributeError (see config.datasets_constraint).
    _run_uv(
        ["pip", "install", "--python", py,
         str(WRAPPER_ROOT), "coqui-tts",
         config.transformers_constraint, config.datasets_constraint],
        index, step,
    )

    # MOSS goes in BEFORE the re-pin below, on purpose: it declares
    # `torch==2.7.0` itself and its curated deps can move torch, and the re-pin
    # is what puts it back. Installed after, any drift would survive.
    _install_moss(config, py, index, step)

    # The project deps (f5-tts, or coqui-tts for XTTS) can win the torch
    # resolution and pull a build that re-introduces the FFmpeg requirement via
    # `torchcodec` (unused — audio loads via torchaudio.load → soundfile on the
    # pinned 2.7.x). Re-pin torch to undo any bump from the deps install, then
    # drop the unused torchcodec.
    install_torch(75)
    ndjson.progress(index, TOTAL_STEPS, step, "Entferne ungenutztes torchcodec …", percent=85)
    procutil.run([str(_uv_path()), "pip", "uninstall", "--python", py, "torchcodec"])

    for extra in det.extra_packages:
        ndjson.progress(index, TOTAL_STEPS, step, f"Installiere {extra} …", percent=90)
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

    # Verify BEFORE marking done: the project install (f5-tts) can win the torch
    # resolution and leave behind a PyPI CPU build + torchcodec, which loads fine
    # at startup but crashes on the first ``torchaudio.load`` (→ "Could not load
    # libtorchcodec"). If that slipped past the re-pin/uninstall, fail loudly so
    # ``deps.done`` is NOT written and the next run rebuilds — never freeze a
    # broken venv behind the marker (SPEC §3 idempotency must not cache garbage).


def step_deps(config, det: gpu_detect.Detection) -> None:
    index, step = 4, "deps"
    torch_version = det.torch_version or config.torch_version
    if _is_done(step):
        problem, repairable = _existing_venv_problem(config, torch_version)
        if problem is None:
            ndjson.progress(index, TOTAL_STEPS, step, "Abhängigkeiten vorhanden", skipped=True)
            return
        if repairable:
            # The ordinary upgrade: this version added an engine the installed
            # venv does not have. Its torch is verified good, so the engines are
            # simply installed into it. Rebuilding here would cost a
            # multi-gigabyte torch download AND require deleting a directory a
            # just-stopped server can still be holding open — which is exactly
            # how the first 0.0.1.0 upgrade failed, with "Zugriff verweigert" on
            # .venv\Scripts.
            ndjson.log(f"Installation wird ergänzt ({problem})")
            _install_engines(config, det, str(_venv_python()), index, step)
            _verify_venv(str(_venv_python()), config, torch_version)
            _mark_done(step)
            ndjson.progress(index, TOTAL_STEPS, step, "Abhängigkeiten ergänzt", done=True)
            return
        # The foundation itself is wrong (torch), which no amount of installing
        # fixes. Leaving the marker would hand the broken venv to the model step,
        # which then fails somewhere that explains nothing.
        ndjson.log(
            f"Vorhandene Installation unbrauchbar ({problem}) — wird neu gebaut",
            level="warning",
        )
        _clear_done(step)
    ndjson.progress(index, TOTAL_STEPS, step, "Erstelle venv …")
    # A backend may demand its own interpreter: AMD's native-Windows ROCm wheels
    # are cp312-only, while everything else runs on the configured 3.11. uv
    # fetches a missing Python itself, so this needs no extra step.
    python_version = det.python_version or config.python_version
    _create_venv(python_version, index, step)

    py = str(_venv_python())
    _install_engines(config, det, py, index, step)

    _verify_venv(py, config, torch_version)

    _mark_done(step)
    ndjson.progress(index, TOTAL_STEPS, step, "Abhängigkeiten installiert", done=True)


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
    ndjson.progress(index, TOTAL_STEPS, step, "Installiere MOSS-TTS-Nano …", percent=65)
    _run_uv(["pip", "install", "--python", py, "--no-deps", str(package)], index, step)
    deps = [str(d) for d in (spec.get("deps") or [])]
    if deps:
        _run_uv(["pip", "install", "--python", py, *deps], index, step)


def _verify_moss(py: str, config) -> None:
    """Assert the MOSS runtime imports — before ``deps.done`` is written.

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
            "delete .venv and .state/deps.done and rerun."
        )


def _verify_venv(py: str, config, torch_version: str) -> None:
    """Run every install assertion against an existing venv.

    Called twice, and the second call is the point: once before ``deps.done`` is
    written (a bad fresh resolve must not be cached), and once when that marker
    already exists, because a marker is only ever evidence that an install
    *finished* — not that it still works. A venv can rot after the fact: a hand
    patch, a shared/relocated install, or simply an older wrapper whose install
    steps predate a fix. Re-checking costs a few imports; skipping it means the
    bootstrap steps straight over the broken part and the failure resurfaces
    later inside model download or first inference, where nothing points back
    here.
    """
    _verify_torch(py, config, expected_version=torch_version)
    _verify_transformers(py)
    _verify_f5(py)
    _verify_moss(py, config)


def _verify_f5(py: str) -> None:
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
            "Delete .venv and .state/deps.done and rerun."
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
            "delete .venv and .state/deps.done and rerun."
        )


def _verify_transformers(py: str) -> None:
    """Assert transformers still provides the symbol XTTS needs.

    coqui-tts imports ``transformers.pytorch_utils.isin_mps_friendly``, removed in
    transformers 5.x. If the resolve slipped a 5.x past the pin, XTTS would import
    fine but crash at model-load — so fail HERE (before deps.done) so the next run
    rebuilds instead of freezing a venv that only breaks when you pick the XTTS
    backend (SPEC §3 idempotency must not cache a half-broken venv)."""
    proc = procutil.run([py, "-c", "from transformers.pytorch_utils import isin_mps_friendly"])
    if proc.returncode != 0:
        raise FatalError(
            "transformers verification failed: XTTS requires "
            "transformers.pytorch_utils.isin_mps_friendly (removed in transformers "
            "5.x). Constrain transformers<5 (config.transformers_constraint); "
            "delete .venv and .state/deps.done and rerun."
        )


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
    index, step = 5, "model"
    if _is_done(step):
        ndjson.progress(index, TOTAL_STEPS, step, "Sprachmodelle vorhanden", skipped=True)
        return
    # Download the weights for ALL engines (chosen at start, not install):
    #   F5   → every language's checkpoint (en/de/fr/ja), SPEC §14.3.
    #   XTTS → the one multilingual XTTS-v2 model.
    #   MOSS → the tiny multilingual model + its separate audio tokenizer.
    env = _server_env(config)
    # Let the download sub-processes emit their per-file progress bars onto THIS
    # same "Step 5/6 · model" bar (src.progress.ModelProgress reads these).
    env["F5W_STEP_INDEX"] = str(index)
    env["F5W_STEP_TOTAL"] = str(TOTAL_STEPS)
    ndjson.progress(index, TOTAL_STEPS, step, "Lade F5-Sprachmodelle …", percent=0)
    rc = _popen_forward([str(_venv_python()), "-m", "src.models"], env)
    if rc != 0:
        raise FatalError("F5-Modell-Download fehlgeschlagen (siehe stderr/Log)")
    ndjson.progress(index, TOTAL_STEPS, step, "Lade XTTS-v2-Modell …", percent=50)
    rc = _popen_forward([str(_venv_python()), "-m", "src.xtts_backend"], env)
    if rc != 0:
        raise FatalError("XTTS-Modell-Download fehlgeschlagen (siehe stderr/Log)")
    ndjson.progress(index, TOTAL_STEPS, step, "Lade MOSS-TTS-Nano-Modell …", percent=70)
    rc = _popen_forward([str(_venv_python()), "-m", "src.moss_backend"], env)
    if rc != 0:
        raise FatalError("MOSS-Modell-Download fehlgeschlagen (siehe stderr/Log)")
    # Voices, not weights — and deliberately NOT fatal: a failed voice pack means
    # "no voices yet", which the user can fix by dropping in a wav. It must never
    # keep an otherwise working install from starting, so a non-zero exit only
    # gets logged. The module itself skips the download when samples exist.
    ndjson.progress(index, TOTAL_STEPS, step, "Prüfe Sprachproben …", percent=90)
    if _popen_forward([str(_venv_python()), "-m", "src.voicepack"], env) != 0:
        ndjson.log("Voice-Pack-Download fehlgeschlagen — Installation läuft weiter", level="warning")
    _mark_done(step)
    ndjson.progress(index, TOTAL_STEPS, step, "Sprachmodelle geladen", percent=100, done=True)


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
    ndjson.progress(index, TOTAL_STEPS, step, f"Starte Server auf {config.host}:{config.port} …")
    env = _server_env(config)
    cmd = [
        str(_venv_python()), "-m", "uvicorn",
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
        step_uv()
        step_python(config.python_version)
        det = step_detect(config)
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
        ndjson.error(f"unexpected bootstrap error: {exc}", fatal=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
