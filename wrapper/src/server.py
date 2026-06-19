"""FastAPI application: endpoints, lifespan, parent watchdog (SPEC §5, §13).

The app is built by :func:`create_app`. In production the lifespan detects the
GPU backend, builds and starts the :class:`Engine`, emits the ``ready`` NDJSON
event, and launches the parent-PID watchdog. Tests inject a pre-built engine
(with fake workers) so no torch/F5-TTS is required.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import ndjson
from .config import Config, load_config
from .engine import Engine, QueueFull, TtsParams
from .gpu_detect import detect_backend
from .jobs import JobRegistry
from .samples import InvalidSample, SampleNotFound


class TtsRequest(BaseModel):
    sample: str = Field(..., description="Basename in samples/ (required).")
    text: str
    language: Optional[str] = None
    ref_text: Optional[str] = None
    speed: float = 1.0
    nfe_step: int = 32


def _require_api_key(config: Config):
    """Build a dependency enforcing a Bearer key iff one is configured (SPEC §5)."""

    async def dep(authorization: Optional[str] = Header(default=None)) -> None:
        if not config.api_key:
            return
        expected = f"Bearer {config.api_key}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="missing or invalid API key")

    return dep


async def _parent_watchdog(parent_pid: int, app: FastAPI) -> None:
    """Self-exit when the parent (game) process disappears (SPEC §13.2)."""
    while True:
        await asyncio.sleep(2.0)
        if not _pid_alive(parent_pid):
            ndjson.log(f"parent pid {parent_pid} gone; shutting down", level="warning")
            await _trigger_shutdown(app)
            return


def _port_in_use(host: str, port: int) -> bool:
    """True if (host, port) can't be bound — i.e. something already listens."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # No SO_REUSEADDR: on Windows it would let two servers bind the same port.
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def _pid_alive(pid: int) -> bool:
    try:
        if os.name == "nt":
            import ctypes

            PROCESS_QUERY = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY, False, pid)
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError, ValueError):
        return False


async def _trigger_shutdown(app: FastAPI) -> None:
    engine: Optional[Engine] = getattr(app.state, "engine", None)
    if engine is not None:
        await engine.aclose()
    ndjson.shutdown()
    # Ask uvicorn to stop; fall back to SIGINT to self.
    with contextlib.suppress(Exception):
        os.kill(os.getpid(), signal.SIGINT)


def create_app(
    config: Optional[Config] = None,
    engine: Optional[Engine] = None,
) -> FastAPI:
    config = config or load_config()
    jobs = JobRegistry()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        eng = engine
        if eng is None:
            # Fail fast on a busy port: uvicorn binds AFTER lifespan startup, so
            # without this we'd load every model (~minute on GPU) and only then
            # die on the bind. Emit a clear fatal event instead. (Skipped when an
            # engine is injected, i.e. in tests, to avoid touching a real port.)
            if _port_in_use(config.host, config.port):
                ndjson.error(
                    f"port {config.host}:{config.port} already in use — "
                    f"is another wrapper instance still running?",
                    fatal=True,
                )
                raise RuntimeError("port already in use")
            detection = detect_backend(config)
            ndjson.log(f"backend selected: {detection.detail}")
            eng = Engine(config, detection, jobs)
        await eng.start()
        app.state.engine = eng
        app.state.jobs = jobs
        ndjson.ready(config.host, config.port, eng.backend, eng.device, eng.max_workers)

        watchdog: Optional[asyncio.Task] = None
        if config.parent_pid:
            watchdog = asyncio.create_task(_parent_watchdog(config.parent_pid, app))
        try:
            yield
        finally:
            if watchdog:
                watchdog.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watchdog
            await eng.aclose()

    app = FastAPI(title="F5-TTS Wrapper", lifespan=lifespan)
    api_key_dep = _require_api_key(config)

    def _engine(request: Request) -> Engine:
        return request.app.state.engine

    @app.post("/tts", dependencies=[Depends(api_key_dep)])
    async def tts(req: TtsRequest, request: Request):
        eng: Engine = _engine(request)
        # The wrapper loads one language at startup; reject mismatched requests
        # rather than silently synthesizing in the wrong voice/accent.
        if req.language and req.language != config.language:
            raise HTTPException(
                status_code=400,
                detail=f"wrapper loaded for language '{config.language}', "
                f"request asked for '{req.language}'",
            )
        # Validate + admit BEFORE the 200 stream starts, so error codes are real.
        try:
            audio_path = eng.samples.resolve_path(req.sample)
        except InvalidSample as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except SampleNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        try:
            eng.admit()
        except QueueFull as exc:
            raise HTTPException(status_code=503, detail=str(exc))

        job = jobs.create()
        params = TtsParams(
            sample=req.sample,
            text=req.text,
            language=req.language,
            ref_text=req.ref_text,
            speed=req.speed,
            nfe_step=req.nfe_step,
        )
        headers = {
            "X-Job-Id": job.job_id,
            "X-Sample-Rate": str(eng.sample_rate),
            "X-Channels": "1",
            "X-Sample-Format": "pcm_s16le",
        }
        return StreamingResponse(
            eng.stream(job, params, audio_path),
            media_type="audio/pcm",
            headers=headers,
        )

    @app.get("/samples", dependencies=[Depends(api_key_dep)])
    async def samples(request: Request, details: bool = False):
        return _engine(request).samples.list_samples(details=details)

    @app.post("/cancel/{job_id}", dependencies=[Depends(api_key_dep)])
    async def cancel(job_id: str):
        if not jobs.cancel(job_id):
            raise HTTPException(status_code=404, detail="unknown job")
        return {"cancelled": True}

    @app.get("/jobs/{job_id}", dependencies=[Depends(api_key_dep)])
    async def job_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return job.snapshot()

    @app.get("/health")
    async def health(request: Request):
        return _engine(request).health()

    @app.post("/shutdown", dependencies=[Depends(api_key_dep)])
    async def shutdown(request: Request):
        asyncio.create_task(_trigger_shutdown(request.app))
        return {"shutdown": True}

    return app
