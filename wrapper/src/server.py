"""FastAPI application: endpoints, lifespan, parent watchdog (SPEC §5, §13).

The app is built by :func:`create_app`. In production the lifespan detects the
GPU backend, builds and starts the :class:`Engine`, emits the ``ready`` NDJSON
event, and launches the parent-PID watchdog. Tests inject a pre-built engine
(with fake workers) so no torch/F5-TTS is required.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import signal
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel, Field

from pathlib import Path

from . import engines, i18n, ndjson, wav
from .config import Config, load_config
from .engine import Engine, EngineFailed, QueueFull, Reference, TtsParams
from .gpu_detect import detect_backend
from .inline_ref import (
    INLINE_SAMPLE_RATE,
    InlineReference,
    InvalidInlineReference,
    build_reference,
    decode_pcm16,
)
from .jobs import JobRegistry
from .response_cache import ResponseCache
from .ratelimit import RateLimitExceeded, RateLimiter, client_address
from .samples import InvalidSample, SampleNotFound


# The built-in test page. Single self-contained file, no build step and no CDN,
# so it works in the container and on an offline machine alike.
UI_INDEX = Path(__file__).resolve().parent / "static" / "index.html"


# Longest clip /asr-check accepts: one Whisper window. The plugin only checks
# candidate references (3-15 s), so anything longer is a client bug.
ASR_MAX_SECONDS = 30.0


class TtsRequest(BaseModel):
    # Exactly one of ``sample`` and ``ref_audio`` (checked in the handler, so
    # the answer is a plain 400 like every other validation error here).
    sample: Optional[str] = Field(None, description="Basename in samples/.")
    text: str
    language: Optional[str] = None
    # With ``sample``: fallback transcript (a sidecar wins). With ``ref_audio``:
    # REQUIRED, the game text of the clip — no ASR fills it in.
    ref_text: Optional[str] = None
    # Inline reference (plan game-voice-reference §2.4): base64 PCM s16le,
    # 24 kHz mono. Kept in RAM for this request only — never written to
    # samples/, never a temp file.
    ref_audio: Optional[str] = Field(None, description="base64 PCM s16le 24 kHz mono.")
    # Ids of the clips ``ref_audio`` was built from (the plugin sends the game
    # clip ids). Their hash keys the conditioning cache; without them the key
    # is the hash of audio + text.
    ref_ids: Optional[list[str]] = None
    # Bounded because both reach the engine unchecked otherwise, and both are
    # now editable in the built-in UI. speed <= 0 is not slow, it is undefined;
    # a huge nfe_step holds a worker for minutes (the pool takes exactly one per
    # request), which is a cheap way to stall a shared instance. The ranges are
    # deliberately wider than anything useful — this is a guard rail, not taste.
    speed: float = Field(1.0, gt=0.0, le=3.0)
    nfe_step: int = Field(32, ge=4, le=128)
    # "pcm" (default) streams raw s16le as it is generated — what the plugin
    # consumes. "wav" buffers the whole clip and prepends a RIFF header, because
    # a browser cannot play headerless PCM; it is what the built-in web UI uses.
    # The streaming property is deliberately given up in that mode: a correct
    # RIFF header needs the total length, and the alternative (a header claiming
    # an unknown size) is honoured inconsistently across browsers.
    format: str = "pcm"
    # Generate anew even if the response cache holds this line (and replace
    # the entry). The plugin sends it when the user explicitly (re)generates.
    no_cache: bool = False


class AsrCheckRequest(BaseModel):
    """``POST /asr-check``: audio in, text out, nothing stored (plan §4)."""

    audio: str = Field(..., description="base64 PCM s16le 24 kHz mono.")
    language: Optional[str] = None


class SpeakerEmbeddingRequest(BaseModel):
    """``POST /speaker-embedding``: audio in, vector out, nothing stored (plan §11.3)."""

    audio: str = Field(..., description="base64 PCM s16le 24 kHz mono.")


# Shortest clip /speaker-embedding accepts: below this a speaker vector is noise
# (the plugin only sends its 3-15 s reference candidates anyway).
EMBED_MIN_SECONDS = 1.0


# Display names of the backends for user-facing messages (owned by engines.py).
BACKEND_NAMES = {spec.id: spec.name for spec in engines.ENGINES.values()}


def _request_languages(config: Config) -> Optional[frozenset]:
    """Languages the ACTIVE backend accepts per request, or ``None`` when it is
    locked to the model loaded at startup.

    XTTS and MOSS are multilingual in ONE model, so a per-request language just
    selects the target — no reload, no cost. (MOSS goes further and infers the
    language from the text itself; the set is still validated, so a request for
    something it was never trained on fails cleanly instead of returning
    confident nonsense.) F5 loads one finetune per process and can only voice
    that language. The imports are lazy so this module stays importable without
    the engines installed.

    Deliberately a lookup and not an ``if`` chain per backend: a further
    multilingual engine is one entry here, not a second branch in
    ``_resolve_language`` and ``/languages`` as well.
    """
    if config.tts_backend == "xtts":
        from .xtts_backend import XTTS_LANGUAGES

        return XTTS_LANGUAGES
    if config.tts_backend == "moss":
        from .moss_backend import MOSS_LANGUAGES

        return MOSS_LANGUAGES
    spec = engines.ENGINES.get(config.tts_backend)
    if spec is not None and spec.group == engines.QWEN_GROUP:
        from .qwen_backend import QWEN_LANGUAGES

        return QWEN_LANGUAGES
    return None


def _resolve_language(config: Config, requested: Optional[str]) -> str:
    """Pick the effective synthesis language, backend-aware (SPEC §5).

    For a multilingual backend an unsupported code is a clean 400 rather than a
    deep 500 from inside the engine. For a locked one (F5) a per-request
    ``language`` is ignored — the loaded model is used regardless — never
    rejected. Omitting ``language`` always falls back to the active/startup
    language.
    """
    options = _request_languages(config)
    if options is not None and requested:
        if requested not in options:
            name = BACKEND_NAMES.get(config.tts_backend, config.tts_backend)
            raise HTTPException(
                status_code=400,
                detail=f"language '{requested}' is not supported by {name} "
                f"(supported: {', '.join(sorted(options))})",
            )
        return requested
    # Locked backend, or no request language: use the loaded/startup language.
    return config.language


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
    cache = ResponseCache(config.response_cache_minutes, config.response_cache_max_mb, jobs)

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
                    i18n.t("server.port_in_use", host=config.host, port=config.port),
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
    # One limiter per app: its window is process state, like the engine's queue.
    limiter = RateLimiter(
        per_hour=config.rate_limit_per_hour,
        per_ip_per_hour=config.rate_limit_per_ip_per_hour,
    )
    app.state.limiter = limiter

    def _engine(request: Request) -> Engine:
        return request.app.state.engine

    # Whisper inside F5 is one process-wide pipeline and not thread-safe, so
    # /asr-check runs one at a time. An asyncio lock (not a thread lock) so
    # waiting callers do not each hold an executor thread that the TTS workers
    # need; bounded by max_queue like the engine, answering 503 beyond it.
    asr_lock = asyncio.Lock()
    asr_waiting = {"n": 0}
    # /speaker-embedding: one at a time as well (a batch over every voiced NPC
    # sends thousands), bounded the same way. Separate from the Whisper lock —
    # the two do not share a model.
    embed_lock = asyncio.Lock()
    embed_waiting = {"n": 0}

    def _decode_bounded(b64: str, min_s: float, max_s: float):
        """base64 PCM → (float audio, seconds), 400 outside [min_s, max_s]."""
        try:
            audio = decode_pcm16(b64)
        except InvalidInlineReference as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        seconds = audio.shape[-1] / float(INLINE_SAMPLE_RATE)
        if seconds < min_s or seconds > max_s:
            raise HTTPException(
                status_code=400, detail=f"audio is {seconds:.2f}s, allowed {min_s:g}-{max_s:g}s"
            )
        return audio, seconds

    @contextlib.asynccontextmanager
    async def _serialized(lock: asyncio.Lock, waiting: dict, name: str):
        """One call at a time behind ``lock``; 503 once ``max_queue`` callers wait."""
        if waiting["n"] >= max(1, config.max_queue):
            raise HTTPException(status_code=503, detail=f"{name} queue is full")
        waiting["n"] += 1
        try:
            async with lock:
                yield
        finally:
            waiting["n"] -= 1

    def _resolve_reference(eng: Engine, req: TtsRequest) -> Reference:
        """The request's reference as a sample path or an in-memory voice.

        Raises the HTTP error itself so /tts reads top to bottom.
        """
        if (req.sample is None) == (req.ref_audio is None):
            raise HTTPException(
                status_code=400, detail="exactly one of 'sample' or 'ref_audio' is required"
            )
        if req.ref_audio is not None:
            if not eng.supports_inline_reference:
                name = BACKEND_NAMES.get(config.tts_backend, config.tts_backend)
                raise HTTPException(
                    status_code=501,
                    detail=f"inline reference audio is not supported by {name} "
                    f"on this runtime (see /health inline_reference)",
                )
            try:
                return build_reference(
                    req.ref_audio, req.ref_text, req.ref_ids, eng.inline_ref_max_seconds
                )
            except InvalidInlineReference as exc:
                raise HTTPException(status_code=400, detail=str(exc))
        try:
            return eng.samples.resolve_path(req.sample)
        except InvalidSample as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except SampleNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    def _cache_key(eng: Engine, req: TtsRequest, reference: Reference, language) -> tuple:
        """Everything a request varies that changes the sound. Process-level
        options (engine, pocket_*, qwen_*, …) are constant for the cache's
        lifetime, which is this process."""
        if isinstance(reference, InlineReference):
            text_hash = hashlib.sha256(reference.text.encode("utf-8")).hexdigest()
            voice: tuple = ("inline", reference.voice_id, text_hash)
        else:
            voice_key = getattr(eng.samples, "voice_key", None)
            name = voice_key(reference) if voice_key else str(reference)
            voice = ("sample", name, req.ref_text or "")
        return (config.tts_backend, language, req.text, voice, float(req.speed), int(req.nfe_step))

    @app.post("/tts", dependencies=[Depends(api_key_dep)])
    async def tts(req: TtsRequest, request: Request):
        eng: Engine = _engine(request)
        # Resolve the effective language. XTTS is multilingual in a single model,
        # so a per-request language just selects the target — no reload. F5 loads
        # one finetune per process, so it ignores the request language and uses
        # its loaded model.
        language = _resolve_language(config, req.language)
        if req.format not in ("pcm", "wav"):
            raise HTTPException(
                status_code=400, detail=f"format must be 'pcm' or 'wav', not {req.format!r}"
            )
        # Validate + admit BEFORE the 200 stream starts, so error codes are real.
        reference = _resolve_reference(eng, req)
        # Rate limit BEFORE admission: a rejected caller must not touch the
        # queue at all. The two are different answers — 429 "you have had your
        # share this hour" vs 503 "busy right now, retry shortly".
        try:
            limiter.check(client_address(request, config.trust_forwarded_for))
        except RateLimitExceeded as exc:
            raise HTTPException(
                status_code=429,
                detail=str(exc),
                headers={"Retry-After": str(exc.retry_after)},
            )
        key = _cache_key(eng, req, reference, language)
        gen, how = (None, "") if req.no_cache else cache.lookup(key)
        if gen is None:
            # Only a NEW generation takes a queue slot; joining one or a cache
            # hit costs no worker.
            try:
                eng.admit()
            except (QueueFull, EngineFailed) as exc:
                raise HTTPException(status_code=503, detail=str(exc))
            params = TtsParams(
                sample=req.sample or "",
                text=req.text,
                language=language,
                ref_text=req.ref_text,
                speed=req.speed,
                nfe_step=req.nfe_step,
            )
            producer = jobs.create()
            gen, how = cache.start(
                key, producer, eng.stream(producer, params, reference), no_cache=req.no_cache
            )
        # The client's own job: /cancel and a dropped connection detach only
        # this client; the generation stops when its LAST client has left.
        job = jobs.create()
        cache.attach(gen)
        if how in ("hit", "shared"):
            ndjson.log(f"tts cache {how}: job={job.job_id} generation={gen.job.job_id}")
        headers = {
            "X-Job-Id": job.job_id,
            "X-Cache": how,
            "X-Sample-Rate": str(eng.sample_rate),
            "X-Channels": "1",
            "X-Sample-Format": "pcm_s16le",
        }
        if req.format == "wav":
            # Buffered on purpose — see TtsRequest.format. Everything else about
            # the request (validation, admission, job, cancel) is identical.
            pcm = bytearray()
            async for part in cache.subscribe(gen, job):
                pcm += part
            return Response(
                content=wav.wrap_pcm(bytes(pcm), sample_rate=eng.sample_rate),
                media_type="audio/wav",
                headers=headers,
            )
        return StreamingResponse(
            cache.subscribe(gen, job),
            media_type="audio/pcm",
            headers=headers,
        )

    @app.post("/asr-check", dependencies=[Depends(api_key_dep)])
    async def asr_check(req: AsrCheckRequest, request: Request):
        """Transcribe in-memory audio; nothing is stored (plan §4, Teil C).

        The plugin compares the result with the game text to rank candidate
        reference clips. Same wire format as an inline ``ref_audio``.
        """
        eng: Engine = _engine(request)
        audio, seconds = _decode_bounded(req.audio, 0.0, ASR_MAX_SECONDS)
        async with _serialized(asr_lock, asr_waiting, "asr"):
            loop = asyncio.get_running_loop()
            try:
                text = await loop.run_in_executor(
                    None, eng.transcribe_pcm, audio, INLINE_SAMPLE_RATE, req.language
                )
            except Exception as exc:  # noqa: BLE001 — surface as 500, keep serving
                ndjson.log(f"asr-check failed: {type(exc).__name__}: {exc}", level="warning")
                raise HTTPException(
                    status_code=500, detail=f"asr failed: {type(exc).__name__}: {exc}"
                )
        return {"text": text, "seconds": round(seconds, 3)}

    @app.post("/speaker-embedding", dependencies=[Depends(api_key_dep)])
    async def speaker_embedding(req: SpeakerEmbeddingRequest, request: Request):
        """Speaker vector of in-memory audio; nothing is stored (plan §11.3).

        The plugin groups the clips of one speaker by these vectors to find
        voice-actor changes. Same wire format as ``/asr-check``. The answer
        names the encoder: vectors of different encoders are not comparable.
        """
        eng: Engine = _engine(request)
        encoder = eng.speaker_encoder
        if encoder is None:
            raise HTTPException(
                status_code=501,
                detail=f"the {config.tts_backend} engine has no speaker encoder",
            )
        audio, seconds = _decode_bounded(req.audio, EMBED_MIN_SECONDS, ASR_MAX_SECONDS)
        async with _serialized(embed_lock, embed_waiting, "speaker-embedding"):
            try:
                vec = await eng.speaker_embedding(audio, INLINE_SAMPLE_RATE)
            except EngineFailed as exc:
                raise HTTPException(status_code=503, detail=str(exc))
            except Exception as exc:  # noqa: BLE001 — InferenceError & co: 500, keep serving
                ndjson.log(f"speaker-embedding failed: {type(exc).__name__}: {exc}", level="warning")
                raise HTTPException(
                    status_code=500,
                    detail=f"speaker embedding failed: {type(exc).__name__}: {exc}",
                )
        return {
            "embedding": [round(float(v), 6) for v in vec],
            "encoder": encoder,
            "dim": int(vec.shape[0]),
            "seconds": round(seconds, 3),
        }

    @app.get("/languages", dependencies=[Depends(api_key_dep)])
    async def languages(request: Request):
        """What the caller may put in a request's ``language`` field.

        Exists for the web UI, which has to grey the selector out for F5: that
        backend loads ONE finetune per process, so a per-request language is
        ignored (never rejected). Deriving that from /health alone would mean
        teaching the page the backend rules — better answered where the rules
        already live.
        """
        options = _request_languages(config)
        if options is not None:
            name = BACKEND_NAMES.get(config.tts_backend, config.tts_backend)
            return {
                "active": config.language,
                "options": sorted(options),
                "locked": False,
                "reason": f"{name} is multilingual in one model — pick any language per request",
            }
        return {
            "active": config.language,
            "options": [config.language],
            "locked": True,
            "reason": (
                f"F5 loads one language model per process; this one serves "
                f"'{config.language}'. Restart with --language to change it."
            ),
        }

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
        return {
            **_engine(request).health(),
            "rate_limit": limiter.snapshot(),
            "response_cache": cache.snapshot(),
        }

    @app.get("/", include_in_schema=False)
    async def ui():
        """The built-in test page.

        Deliberately NOT behind the API key: you have to be able to load the
        page in order to type the key into it. Every call the page then makes
        (/languages, /samples, /tts) is protected as usual.
        """
        if not UI_INDEX.exists():
            return JSONResponse(
                {"detail": "web UI not installed (src/static/index.html missing)"},
                status_code=404,
            )
        return FileResponse(UI_INDEX, media_type="text/html")

    @app.post("/shutdown", dependencies=[Depends(api_key_dep)])
    async def shutdown(request: Request):
        asyncio.create_task(_trigger_shutdown(request.app))
        return {"shutdown": True}

    return app
