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


def step_deps(config, det: gpu_detect.Detection) -> None:
    index, step = 4, "deps"
    if _is_done(step):
        ndjson.progress(index, TOTAL_STEPS, step, "Abhängigkeiten vorhanden", skipped=True)
        return
    ndjson.progress(index, TOTAL_STEPS, step, "Erstelle venv …")
    # --clear: replace any pre-existing .venv (e.g. one an earlier `uv run`
    # accidentally synced from pyproject) instead of failing "already exists".
    # We own this venv and install the pinned torch into it below.
    _run_uv(["venv", str(VENV_DIR), "--clear", "--python", config.python_version], index, step)

    py = str(_venv_python())
    torch_pin = f"torch=={config.torch_version}"
    audio_pin = f"torchaudio=={config.torchaudio_version}"

    def install_torch(pct: int) -> None:
        ndjson.progress(
            index, TOTAL_STEPS, step, f"Installiere PyTorch ({det.backend}) …", percent=pct
        )
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
    _run_uv(
        ["pip", "install", "--python", py,
         str(WRAPPER_ROOT), "coqui-tts", config.transformers_constraint],
        index, step,
    )

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
        _run_uv(["pip", "install", "--python", py, extra], index, step)

    # Verify BEFORE marking done: the project install (f5-tts) can win the torch
    # resolution and leave behind a PyPI CPU build + torchcodec, which loads fine
    # at startup but crashes on the first ``torchaudio.load`` (→ "Could not load
    # libtorchcodec"). If that slipped past the re-pin/uninstall, fail loudly so
    # ``deps.done`` is NOT written and the next run rebuilds — never freeze a
    # broken venv behind the marker (SPEC §3 idempotency must not cache garbage).
    _verify_torch(py, config)
    _verify_transformers(py)

    _mark_done(step)
    ndjson.progress(index, TOTAL_STEPS, step, "Abhängigkeiten installiert", done=True)


def _verify_torch(py: str, config) -> None:
    """Assert the venv ended up with the pinned, torchcodec-free torch."""
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
    if base != config.torch_version or info.get("codec"):
        raise FatalError(
            f"torch verification failed: installed={info!r}; "
            f"expected torch=={config.torch_version} and no torchcodec. "
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
    env = _server_env(config)
    # Let the download sub-processes emit their per-file progress bars onto THIS
    # same "Step 5/6 · model" bar (src.progress.ModelProgress reads these).
    env["F5W_STEP_INDEX"] = str(index)
    env["F5W_STEP_TOTAL"] = str(TOTAL_STEPS)
    ndjson.progress(index, TOTAL_STEPS, step, "Lade F5-Sprachmodelle …", percent=0)
    rc = _popen_forward([str(_venv_python()), "-m", "src.models"], env)
    if rc != 0:
        raise FatalError("F5-Modell-Download fehlgeschlagen (siehe stderr/Log)")
    ndjson.progress(index, TOTAL_STEPS, step, "Lade XTTS-v2-Modell …", percent=60)
    rc = _popen_forward([str(_venv_python()), "-m", "src.xtts_backend"], env)
    if rc != 0:
        raise FatalError("XTTS-Modell-Download fehlgeschlagen (siehe stderr/Log)")
    _mark_done(step)
    ndjson.progress(index, TOTAL_STEPS, step, "Sprachmodelle geladen", percent=100, done=True)


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
    # The server watches this pid and self-exits when it dies, so it can never be
    # orphaned (closed console window, killed bootstrap, etc.). An explicit
    # --parent-pid (e.g. the C# host) wins; otherwise watch bootstrap itself.
    env["F5W_PARENT_PID"] = str(config.parent_pid or os.getpid())
    # Propagate resolved overrides so the server subprocess (which re-runs
    # load_config) honors flags the host passed to bootstrap.py, e.g.
    # --language / --api-key, not just config.json defaults.
    env["F5W_LANGUAGE"] = config.language
    env["F5W_TTS_BACKEND"] = config.tts_backend
    if config.api_key:
        env["F5W_API_KEY"] = config.api_key
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
