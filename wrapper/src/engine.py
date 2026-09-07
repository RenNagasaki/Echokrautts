"""F5-TTS worker pool and inference (SPEC §4.2, §6, §9, §13.2).

The engine owns a small pool of F5-TTS model instances (loaded once at startup,
never per request). Each ``/tts`` job acquires one instance exclusively, runs the
blocking ``infer()`` per sentence chunk in a thread executor, and streams the
resulting PCM. Backpressure is provided by an asyncio queue of free workers plus
a ``max_queue`` cap (→ HTTP 503).

Crash isolation: a failed inference becomes an :class:`InferenceError` (HTTP 500)
and the offending worker is re-instantiated; the server process keeps running.
DirectML/XPU workers run a self-test at startup and silently fall back to CPU if
their op-coverage is insufficient.

All F5-TTS / torch imports are lazy (inside the worker) so the rest of the
package — and the unit test suite — import without those heavy deps installed.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Callable, Iterator, Optional, Protocol

import numpy as np

from . import audio_compat, ndjson, selftest
from .config import Config
from .gpu_detect import Detection
from .jobs import CANCELLED, DONE, ERROR, Job, JobRegistry
from .samples import SampleService
from .streaming import chunk_text

DEFAULT_SAMPLE_RATE = 24000  # F5TTS_v1_Base output rate (SPEC §5.1, verified §14.2)


class QueueFull(Exception):
    """Too many requests are queued (HTTP 503)."""


class InferenceError(Exception):
    """A worker failed to synthesize a chunk (HTTP 500)."""


@dataclass
class TtsParams:
    sample: str
    text: str
    language: Optional[str] = None
    ref_text: Optional[str] = None
    speed: float = 1.0
    nfe_step: int = 32


class WorkerProtocol(Protocol):
    sample_rate: int
    device: str
    # Whether the worker can stream a chunk incrementally via ``infer_stream``
    # (token streaming, e.g. XTTS). False → the engine uses the one-shot
    # ``infer`` path (e.g. F5, which returns a whole clip). Accessed via getattr
    # so workers/fakes that omit it default to False.
    supports_streaming: bool

    def infer(self, ref_file: str, ref_text: str, gen_text: str, nfe_step: int, speed: float, language: Optional[str] = None) -> np.ndarray: ...
    def transcribe(self, audio_path: Path) -> str: ...
    def self_test(self) -> bool: ...
    # Optional (only when supports_streaming): yield float32 audio chunks as they
    # are generated for a single text chunk.
    def infer_stream(self, ref_file: str, ref_text: str, gen_text: str, speed: float, language: Optional[str] = None) -> Iterator[np.ndarray]: ...


# --------------------------------------------------------------- real worker
class F5TTSWorker:
    """Wraps a single ``f5_tts.api.F5TTS`` instance bound to one device."""

    supports_streaming = False  # F5's infer() returns a whole clip, not a stream

    def __init__(self, config: Config, device: str):
        from f5_tts.api import F5TTS  # lazy: heavy import

        from .models import resolve_model

        self.device = device
        self.sample_rate = DEFAULT_SAMPLE_RATE
        # Load the checkpoint for the wrapper's active language. For finetunes
        # this returns local ckpt/vocab paths (already cached from install).
        spec = resolve_model(config, config.language)
        self._model = F5TTS(
            model=spec.arch,
            ckpt_file=spec.ckpt_file or "",
            vocab_file=spec.vocab_file or "",
            device=None if device == "cuda" else device,
            hf_cache_dir=str(config.models_path),
        )
        # F5TTS exposes target_sample_rate via its mel config when available.
        sr = getattr(self._model, "target_sample_rate", None)
        if isinstance(sr, int) and sr > 0:
            self.sample_rate = sr

    def infer(self, ref_file: str, ref_text: str, gen_text: str, nfe_step: int, speed: float, language: Optional[str] = None) -> np.ndarray:
        # ``language`` is ignored: an F5 worker is bound to one finetune at load
        # time (one language per process), and the server already rejects a
        # mismatched request before it reaches here. Accepted for protocol parity.
        wav, sr, _spec = self._model.infer(
            ref_file=ref_file,
            ref_text=ref_text or "",
            gen_text=gen_text,
            nfe_step=nfe_step,
            speed=speed,
            remove_silence=False,
            show_info=lambda *a, **k: None,
            progress=None,
        )
        if isinstance(sr, int) and sr > 0:
            self.sample_rate = sr
        return np.asarray(wav, dtype=np.float32)

    def transcribe(self, audio_path: Path) -> str:
        # F5-TTS ships a Whisper-based preprocessor that transcribes when the
        # ref_text is empty; reuse it so we don't pull a second ASR stack.
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text

        _audio, ref_text = preprocess_ref_audio_text(str(audio_path), "")
        return ref_text or ""

    def self_test(self) -> bool:
        """Tiny inference to confirm the backend's op-coverage (SPEC §4.2)."""
        return selftest.run(self)


WorkerFactory = Callable[[int, str], WorkerProtocol]


def _default_factory(config: Config) -> WorkerFactory:
    # Pick the backend worker for this process (one backend per process, like
    # language). XTTS is imported lazily so its heavy deps (coqui-tts) are only
    # required when actually selected.
    if config.tts_backend == "xtts":
        def make_xtts(worker_id: int, device: str) -> WorkerProtocol:
            from .xtts_backend import XTTSWorker  # lazy: coqui-tts/torch

            return XTTSWorker(config, device)

        return make_xtts

    def make(worker_id: int, device: str) -> WorkerProtocol:
        return F5TTSWorker(config, device)

    return make


# ------------------------------------------------------------------- helpers
def float_to_pcm16(wav: np.ndarray) -> bytes:
    """Convert a float32 mono waveform in [-1, 1] to little-endian PCM s16."""
    arr = np.asarray(wav, dtype=np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype("<i2").tobytes()


# Sentinel returned by _pump_next when a streaming worker's generator is done.
_STREAM_DONE = object()


def _pump_next(gen: Iterator[np.ndarray]) -> object:
    """Pull one item from a sync generator, returning ``_STREAM_DONE`` at the end.

    Runs in the thread executor (like ``infer``) so a blocking chunk synthesis
    never stalls the event loop, and so token streaming crosses the sync→async
    boundary one chunk at a time without a queue/threadsafe dance."""
    try:
        return next(gen)
    except StopIteration:
        return _STREAM_DONE


# -------------------------------------------------------------------- engine
class Engine:
    def __init__(
        self,
        config: Config,
        detection: Detection,
        jobs: JobRegistry,
        samples: Optional[SampleService] = None,
        worker_factory: Optional[WorkerFactory] = None,
    ):
        self._config = config
        self._detection = detection
        self._jobs = jobs
        self._factory = worker_factory or _default_factory(config)
        self.device = detection.device
        self.backend = detection.backend
        self._workers: list[WorkerProtocol] = []
        self._free: asyncio.Queue[WorkerProtocol] = asyncio.Queue()
        self._pending = 0  # queued + running
        self._started = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Sample service shares this engine's transcriber so ASR reuses a worker.
        self.samples = samples or SampleService(config, transcriber=self._transcribe)

    # -------------------------------------------------------------- lifecycle
    @property
    def max_workers(self) -> int:
        return len(self._workers)

    @property
    def queue_depth(self) -> int:
        return max(0, self._pending - self.max_workers)

    @property
    def sample_rate(self) -> int:
        return self._workers[0].sample_rate if self._workers else DEFAULT_SAMPLE_RATE

    async def start(self) -> None:
        """Build the worker pool, self-test fragile backends, fill the queue."""
        if self._started:
            return
        self._started = True
        self._loop = asyncio.get_running_loop()
        # Before any worker (and therefore any engine library) is imported: make
        # sure torchaudio.load decodes without TorchCodec/FFmpeg. Only matters on
        # torchaudio >= 2.9, i.e. AMD's native-Windows ROCm build.
        audio_compat.ensure_native_audio_loading()
        count = max(1, self._detection.max_workers_hint)
        device = self._detection.device
        for i in range(count):
            # Backends whose op coverage cannot be assumed fall back to CPU
            # instead of failing startup. DirectML/XPU are incomplete by nature;
            # native-Windows ROCm is included because the GPU was matched by
            # ADAPTER NAME (see gpu_detect) — the card may not really be on
            # AMD's supported list, and a CPU worker beats a dead server.
            fragile = device in ("dml", "xpu") or self.backend == "rocm_win"
            if not fragile:
                worker = await self._build_worker(i, device)
            else:
                reason = ""
                try:
                    worker = await self._build_worker(i, device)
                    # A worker that builds can still lack the ops to run.
                    if not await self._loop.run_in_executor(None, worker.self_test):
                        reason = "self-test failed"
                except Exception as exc:  # noqa: BLE001 — any init failure means "no GPU here"
                    # e.g. HIP/DirectML refusing the device outright.
                    reason = f"worker init failed: {exc}"
                if reason:
                    ndjson.log(
                        f"{device}/{self.backend} {reason}; worker {i} falling back to CPU",
                        level="warning",
                    )
                    device = "cpu"
                    self.device = "cpu"
                    self.backend = "cpu"
                    worker = await self._build_worker(i, "cpu")
            self._workers.append(worker)
            self._free.put_nowait(worker)

    async def _build_worker(self, worker_id: int, device: str) -> WorkerProtocol:
        assert self._loop is not None
        return await self._loop.run_in_executor(
            None, self._factory, worker_id, device
        )

    async def aclose(self) -> None:
        self._jobs.cancel_all()

    def health(self) -> dict:
        return {
            "status": "ok",
            "language": self._config.language,
            "tts_backend": self._config.tts_backend,
            "backend": self.backend,
            "device": self.device,
            # Effective XTTS fp16 state: opt-in flag AND xtts backend AND CUDA.
            "xtts_fp16": (
                self._config.tts_backend == "xtts"
                and bool(self._config.xtts_fp16)
                and self.device == "cuda"
            ),
            "workers": self.max_workers,
            "queue": self.queue_depth,
        }

    # ----------------------------------------------------------------- ASR
    def _transcribe(self, audio_path: Path) -> str:
        """Synchronous transcription using any worker (for ref-text fallback)."""
        if not self._workers:
            return ""
        return self._workers[0].transcribe(audio_path)

    # ------------------------------------------------------------- synthesis
    def admit(self) -> None:
        """Admission control (SPEC §9). Raises :class:`QueueFull` (503) or counts
        the request as pending. Must be paired with a single :meth:`stream` call,
        whose ``finally`` releases the pending slot."""
        if self._pending >= self.max_workers + self._config.max_queue:
            raise QueueFull("inference queue is full")
        self._pending += 1

    async def stream(
        self, job: Job, params: TtsParams, audio_path: Path
    ) -> AsyncIterator[bytes]:
        """Async generator yielding PCM per sentence chunk (SPEC §6).

        Call :meth:`admit` and resolve ``audio_path`` (so 503/400/404 surface
        before the HTTP 200) *before* iterating this. Resolves ref-text, chunks
        the text, acquires a worker with backpressure, then streams — honoring
        the cancel token and updating progress between chunks.
        """
        assert self._loop is not None
        ref_text = await self._loop.run_in_executor(
            None, self.samples.resolve_ref_text, audio_path, params.ref_text
        )

        chunks = chunk_text(params.text, self._config.max_chars_per_chunk)
        self._jobs.set_total(job, len(chunks))

        ndjson.log(
            f"tts request start: job={job.job_id} sample={params.sample} "
            f"file={audio_path.name} lang={params.language} "
            f"chars={len(params.text)} chunks={len(chunks)}"
        )
        # Wall clock for the whole synthesis + total PCM emitted, so we can report
        # generation time, audio length, and the real-time factor (efficiency) at
        # the end of a completed request.
        started = time.monotonic()
        total_bytes = 0
        # Time of the FIRST yielded PCM and the number of parts emitted. This is
        # the streaming metric: with token streaming (XTTS) audio starts flowing
        # long before the request finishes, so a `first` far below `generated`
        # proves the wrapper streams — and pins a late playback on the consumer
        # instead. Without it the only observable is the end of the request.
        first_audio: Optional[float] = None
        parts = 0

        worker: Optional[WorkerProtocol] = None
        try:
            worker = await self._free.get()  # waits if all workers busy
            self._jobs.mark_running(job, id(worker))

            streaming = getattr(worker, "supports_streaming", False)
            for chunk in chunks:
                if job.cancelled:
                    self._jobs.finish(job, CANCELLED)
                    return
                # Token streaming (XTTS) emits PCM parts as they are produced;
                # one-shot (F5) emits a single part per sentence. Both are
                # consumed as one async iterator so cancel checks and the byte /
                # part / first-audio bookkeeping exist exactly once.
                source = (
                    self._stream_chunk(worker, audio_path, ref_text, chunk, params)
                    if streaming
                    else self._one_shot_chunk(worker, audio_path, ref_text, chunk, params)
                )
                try:
                    async for pcm in source:
                        if job.cancelled:
                            self._jobs.finish(job, CANCELLED)
                            return
                        if first_audio is None:
                            first_audio = time.monotonic() - started
                        total_bytes += len(pcm)
                        parts += 1
                        yield pcm
                except InferenceError as exc:
                    # Surface the *real* cause on stdout — otherwise the only
                    # signal is the generic "rebuilt crashed worker" warning and
                    # the underlying error (which lives in the HTTP 500 / job
                    # state) is invisible to whoever watches the process.
                    ndjson.log(f"inference failed: {exc}", level="warning")
                    # Isolate the crash: swap the dead worker for a fresh one so
                    # `finally` returns a healthy worker to the pool (SPEC §9).
                    worker = await self._rebuild_worker(worker)
                    self._jobs.finish(job, ERROR, str(exc))
                    raise
                self._jobs.advance(job)

            self._jobs.finish(job, DONE)
            self._log_timing(job, worker, started, total_bytes, first_audio, parts)
        finally:
            if worker is not None:
                self._free.put_nowait(worker)
            self._pending -= 1

    def _log_timing(
        self,
        job: Job,
        worker: WorkerProtocol,
        started: float,
        total_bytes: int,
        first_audio: Optional[float],
        parts: int,
    ) -> None:
        """Emit a one-line efficiency summary for a completed request (SPEC §8.1).

        Reports how long generation took, how much audio was produced, and the
        real-time factor (generation ÷ audio; < 1.0 means faster than real time)
        so the user can gauge how efficiently the engine runs. PCM is 16-bit mono,
        so 2 bytes per sample.

        ``first`` (seconds until the first PCM left the engine) and ``parts``
        (how many pieces the response was delivered in) make the *streaming*
        behaviour observable: ``first`` ≪ ``generated`` with ``parts`` > 1 means
        audio flowed while synthesis was still running, so a consumer that only
        starts playing at the end is buffering on its own side."""
        gen_s = time.monotonic() - started
        sr = getattr(worker, "sample_rate", 0) or DEFAULT_SAMPLE_RATE
        audio_s = (total_bytes / 2) / sr if sr else 0.0
        rtf = gen_s / audio_s if audio_s > 0 else 0.0
        first_s = f"{first_audio:.2f}s" if first_audio is not None else "n/a"
        ndjson.log(
            f"tts request done: job={job.job_id} generated={gen_s:.2f}s "
            f"audio={audio_s:.2f}s rtf={rtf:.2f} first={first_s} parts={parts}"
        )

    async def _infer_chunk(
        self,
        worker: WorkerProtocol,
        audio_path: Path,
        ref_text: Optional[str],
        chunk: str,
        params: TtsParams,
    ) -> bytes:
        assert self._loop is not None
        try:
            wav = await self._loop.run_in_executor(
                None,
                worker.infer,
                str(audio_path),
                ref_text or "",
                chunk,
                params.nfe_step,
                params.speed,
                params.language,
            )
        except Exception as exc:  # noqa: BLE001 — isolate and surface as HTTP 500
            # Keep the exception *type* in the message — some errors (e.g. bare
            # ``RuntimeError()``) stringify to empty, leaving no clue what failed.
            raise InferenceError(f"{type(exc).__name__}: {exc}") from exc
        return float_to_pcm16(wav)

    async def _one_shot_chunk(
        self,
        worker: WorkerProtocol,
        audio_path: Path,
        ref_text: Optional[str],
        chunk: str,
        params: TtsParams,
    ) -> AsyncIterator[bytes]:
        """Adapt the one-shot path (F5) to the same async-iterator shape as
        :meth:`_stream_chunk`, so :meth:`stream` has a single consume loop."""
        yield await self._infer_chunk(worker, audio_path, ref_text, chunk, params)

    async def _stream_chunk(
        self,
        worker: WorkerProtocol,
        audio_path: Path,
        ref_text: Optional[str],
        chunk: str,
        params: TtsParams,
    ) -> AsyncIterator[bytes]:
        """Yield PCM parts as a streaming worker generates a single text chunk.

        The worker's ``infer_stream`` is a *sync* generator run on a compute
        device; each item is pulled in the thread executor (:func:`_pump_next`)
        so the event loop never blocks. Any failure — building the generator or
        producing a part — is surfaced as :class:`InferenceError` so the caller
        rebuilds the worker exactly like the one-shot path."""
        assert self._loop is not None
        try:
            gen = worker.infer_stream(str(audio_path), ref_text or "", chunk, params.speed, params.language)
        except Exception as exc:  # noqa: BLE001 — isolate and surface as HTTP 500
            raise InferenceError(f"{type(exc).__name__}: {exc}") from exc
        while True:
            try:
                part = await self._loop.run_in_executor(None, _pump_next, gen)
            except Exception as exc:  # noqa: BLE001
                raise InferenceError(f"{type(exc).__name__}: {exc}") from exc
            if part is _STREAM_DONE:
                return
            yield float_to_pcm16(part)  # type: ignore[arg-type]

    async def _rebuild_worker(self, dead: WorkerProtocol) -> WorkerProtocol:
        """Replace a crashed worker instance in place, returning the fresh one.

        On rebuild failure, returns the dead worker unchanged — the pool shrinks
        in practice only if every rebuild fails, but the process stays alive.
        """
        idx = self._workers.index(dead) if dead in self._workers else -1
        try:
            fresh = await self._build_worker(max(idx, 0), self.device)
        except Exception as exc:  # noqa: BLE001
            ndjson.log(f"failed to rebuild worker: {exc}", level="error")
            return dead
        if idx >= 0:
            self._workers[idx] = fresh
        ndjson.log("rebuilt crashed worker", level="warning")
        return fresh
