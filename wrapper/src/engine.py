"""F5-TTS worker pool and inference (SPEC §4.2, §6, §9, §13.2).

The engine owns a small pool of F5-TTS model instances (loaded once at startup,
never per request). Each ``/tts`` job acquires one instance exclusively, runs the
blocking ``infer()`` per sentence chunk in a thread executor, and streams the
resulting PCM. Backpressure is provided by an asyncio queue of free workers plus
a ``max_queue`` cap (→ HTTP 503).

Crash isolation: a failed inference becomes an :class:`InferenceError` (HTTP 500)
and the offending worker is re-instantiated; the server process keeps running.
Exception: a fatal device error (CUDA "device-side assert", illegal memory
access, …) poisons the whole process's CUDA context, so no in-process rebuild can
work. The engine then refuses new work (:class:`EngineFailed`, HTTP 503) and the
process exits with :data:`EXIT_RESTART_REQUIRED` so the host starts a fresh one.
DirectML/XPU workers run a self-test at startup and silently fall back to CPU if
their op-coverage is insufficient.

All F5-TTS / torch imports are lazy (inside the worker) so the rest of the
package — and the unit test suite — import without those heavy deps installed.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Callable, Iterator, Optional, Protocol, Union

import numpy as np

from . import audio_compat, ndjson, selftest
from .asr import PcmTranscriber, whisper_transcribe
from .config import Config
from .gpu_detect import Detection
from .inline_ref import ConditioningCache, InlineReference, ensure_sentence_end
from .jobs import CANCELLED, DONE, ERROR, Job, JobRegistry
from .samples import SampleService
from .streaming import chunk_text

DEFAULT_SAMPLE_RATE = 24000  # F5TTS_v1_Base output rate (SPEC §5.1, verified §14.2)


class QueueFull(Exception):
    """Too many requests are queued (HTTP 503)."""


class InferenceError(Exception):
    """A worker failed to synthesize a chunk (HTTP 500)."""


class EngineFailed(Exception):
    """The engine hit a fatal device error and is about to exit (HTTP 503)."""


# Exit code after a fatal device error: EX_TEMPFAIL ("temporary failure, retry").
# The host (EchokrautLocalInstaller) restarts the wrapper on exactly this code;
# keep both sides in sync (installer Constants.ECHOKRAUTTS_EXIT_RESTART_REQUIRED).
EXIT_RESTART_REQUIRED = 75

# Grace period between the fatal error and the hard exit, so the failing
# request's error and the NDJSON event still leave the process.
FATAL_EXIT_DELAY_S = 1.0

# Substrings (lower-case) of errors after which the CUDA context of THIS process
# is unusable for good: every later kernel launch fails with the same error, and
# building a new worker in-process fails too (User-Log 2026-11-08: 543 follow-up
# 500s after one "device-side assert triggered"). A plain out-of-memory is NOT on
# this list — it is recoverable and handled by the normal worker rebuild.
_FATAL_DEVICE_MARKERS = (
    "device-side assert",
    "illegal memory access",
    "unspecified launch failure",
    "cuda error:",
    "cudaerror",
    "hip error:",  # ROCm reports the same faults as "HIP error: …"
)
_RECOVERABLE_DEVICE_MARKERS = ("out of memory",)


def is_fatal_device_error(exc: BaseException) -> bool:
    """True when ``exc`` (or anything in its cause/context chain) is a device
    fault that leaves the process's CUDA/HIP context permanently broken.

    Walks ``__cause__``/``__context__`` because the engine wraps worker errors
    in :class:`InferenceError`. Out-of-memory errors are excluded even when they
    are phrased as "CUDA error: out of memory" — those are recoverable."""
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = f"{type(current).__name__}: {current}".lower()
        if any(m in text for m in _FATAL_DEVICE_MARKERS) and not any(
            m in text for m in _RECOVERABLE_DEVICE_MARKERS
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _hard_exit(code: int = EXIT_RESTART_REQUIRED) -> None:
    """Leave the process NOW, with ``code``.

    ``os._exit`` on purpose: a graceful uvicorn shutdown would exit with 0 (the
    host could not tell a crash from a stop), and tearing down a broken CUDA
    context can hang or raise in atexit/finalizers."""
    with contextlib.suppress(Exception):
        ndjson.shutdown()
    with contextlib.suppress(Exception):
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(code)


# What a request clones from: a file in samples/ or an in-memory inline
# reference (plan game-voice-reference §2.4). Workers receive ``str(path)`` for
# the former and the :class:`InlineReference` object itself for the latter —
# an object, not a path, precisely so it can never be mistaken for one.
Reference = Union[Path, InlineReference]


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
    # Whether ``ref_file`` may be an :class:`InlineReference` (conditioning from
    # memory, nothing on disk). Read via getattr, so omitting it means False and
    # the server answers 501 for inline requests instead of crashing a worker.
    supports_inline_reference: bool

    def infer(self, ref_file: "str | InlineReference", ref_text: str, gen_text: str, nfe_step: int, speed: float, language: Optional[str] = None) -> np.ndarray: ...
    def transcribe(self, audio_path: Path) -> str: ...
    def self_test(self) -> bool: ...
    # Optional (only when supports_streaming): yield float32 audio chunks as they
    # are generated for a single text chunk.
    def infer_stream(self, ref_file: str, ref_text: str, gen_text: str, speed: float, language: Optional[str] = None) -> Iterator[np.ndarray]: ...


# --------------------------------------------------------------- real worker
class F5TTSWorker:
    """Wraps a single ``f5_tts.api.F5TTS`` instance bound to one device."""

    supports_streaming = False  # F5's infer() returns a whole clip, not a stream
    supports_inline_reference = True

    def __init__(self, config: Config, device: str):
        from f5_tts.api import F5TTS  # lazy: heavy import

        from .models import resolve_model

        self.device = device
        self.sample_rate = DEFAULT_SAMPLE_RATE
        # Prepared inline reference tensors (mono, CPU), per voice. F5 computes
        # the reference mel inside ``sample()`` on every call, so this saves the
        # decode/convert step only — F5's conditioning is not separable from
        # generation the way XTTS latents are.
        self._inline_cache: ConditioningCache = ConditioningCache(config.inline_ref_cache_size)
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

    def infer(self, ref_file: "str | InlineReference", ref_text: str, gen_text: str, nfe_step: int, speed: float, language: Optional[str] = None) -> np.ndarray:
        # ``language`` is ignored: an F5 worker is bound to one finetune at load
        # time (one language per process), and the server already rejects a
        # mismatched request before it reaches here. Accepted for protocol parity.
        if isinstance(ref_file, InlineReference):
            return self._infer_inline(ref_file, gen_text, nfe_step, speed)
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

    def _infer_inline(self, ref: InlineReference, gen_text: str, nfe_step: int, speed: float) -> np.ndarray:
        """F5 inference from an in-memory reference — no file anywhere.

        ``F5TTS.infer`` cannot be used: its ``preprocess_ref_audio_text`` opens
        the reference by path and exports a trimmed copy to a
        ``NamedTemporaryFile`` (f5-tts 1.1.22, ``utils_infer.py``) for EVERY new
        reference. So this goes one level down to ``infer_batch_process``,
        which takes ``(tensor, sample_rate)``, and repeats the two things the
        skipped layers did that matter: the sentence end on ``ref_text`` and the
        per-batch text budget of ``infer_process``. The silence trimming is
        deliberately not repeated — the plugin picks and cuts the clips, and
        trimming here would desync audio from ``ref_text``.
        """
        import torch  # lazy
        from f5_tts.infer.utils_infer import chunk_text as f5_chunk_text
        from f5_tts.infer.utils_infer import infer_batch_process

        audio = self._inline_cache.get_or_compute(
            ref.voice_id, lambda: torch.from_numpy(np.ascontiguousarray(ref.audio)).unsqueeze(0)
        )
        ref_text = ensure_sentence_end(ref.text)
        seconds = audio.shape[-1] / ref.sample_rate
        # Same formula as f5_tts.infer.utils_infer.infer_process.
        max_chars = int(len(ref_text.encode("utf-8")) / seconds * (22 - seconds) * speed)
        batches = f5_chunk_text(gen_text, max_chars=max(1, max_chars))
        if not batches:
            return np.zeros(0, dtype=np.float32)
        model = self._model
        wav, sr, _spec = next(
            infer_batch_process(
                (audio, ref.sample_rate),
                ref_text,
                batches,
                model.ema_model,
                model.vocoder,
                mel_spec_type=model.mel_spec_type,
                progress=None,
                nfe_step=nfe_step,
                speed=speed,
                device=model.device,
            )
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


# Every backend this build can load. Kept next to the factory below so the two
# cannot drift: the list IS what the factory dispatches on.
KNOWN_BACKENDS = ("f5", "xtts", "moss")


def _default_factory(config: Config) -> WorkerFactory:
    # Pick the backend worker for this process (one backend per process, like
    # language). XTTS is imported lazily so its heavy deps (coqui-tts) are only
    # required when actually selected.
    #
    # The check exists because F5 is BOTH the default and the last branch, so
    # without it every unrecognised value quietly became F5: a typo in
    # `F5W_TTS_BACKEND` (documented, and read from the user's environment file
    # by all three compose files), or a backend id a plugin stored before it was
    # renamed. That failure is silent — nothing crashes, the wrong voice just
    # comes out, and `/health` keeps reporting the value that was asked for.
    # Failing at startup with the list of valid names costs one line and a
    # minute of someone's evening instead of an hour.
    if config.tts_backend not in KNOWN_BACKENDS:
        raise ValueError(
            f"unknown tts_backend {config.tts_backend!r} — "
            f"known backends: {', '.join(KNOWN_BACKENDS)}"
        )

    if config.tts_backend == "xtts":
        def make_xtts(worker_id: int, device: str) -> WorkerProtocol:
            from .xtts_backend import XTTSWorker  # lazy: coqui-tts/torch

            return XTTSWorker(config, device)

        return make_xtts

    if config.tts_backend == "moss":
        def make_moss(worker_id: int, device: str) -> WorkerProtocol:
            # Two runtimes, same weights: ONNX is ~2.5x faster on the CPU and
            # is the default. It needs a package and a separate download that an
            # older install does not have, so a machine without them falls back
            # to PyTorch rather than refusing to start -- MOSS still works, just
            # slower. Unlike an unknown BACKEND (silently wrong voice, hence the
            # hard failure above) this substitution is the same model, and it is
            # said out loud so nobody wonders why it got slow.
            if str(config.moss_runtime).strip().lower() == "onnx":
                from . import moss_onnx_backend  # lazy: onnxruntime

                usable, reason = moss_onnx_backend.is_available(config)
                if usable:
                    return moss_onnx_backend.MossOnnxWorker(config, device)
                ndjson.log_once(
                    f"MOSS: ONNX nicht verfuegbar ({reason}) — nutze PyTorch, "
                    "das ist etwa 2,5x langsamer",
                    level="warning",
                )
            from .moss_backend import MossWorker  # lazy: moss runtime/torch

            return MossWorker(config, device)

        return make_moss

    def make(worker_id: int, device: str) -> WorkerProtocol:
        return F5TTSWorker(config, device)

    return make


# ------------------------------------------------------------------- helpers
def float_to_pcm16(wav: np.ndarray) -> bytes:
    """Convert a float32 mono waveform in [-1, 1] to little-endian PCM s16."""
    arr = np.asarray(wav, dtype=np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype("<i2").tobytes()


def _worker_ref(reference: "Reference") -> "str | InlineReference":
    """What a worker's ``ref_file`` gets: the path as a string (unchanged
    contract for file samples) or the in-memory reference object itself."""
    return reference if isinstance(reference, InlineReference) else str(reference)


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
        asr: Optional[PcmTranscriber] = None,
        on_fatal: Optional[Callable[[], None]] = None,
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
        # In-memory ASR for /asr-check (Whisper via F5, lazy). Injectable so the
        # suite runs without torch. Serialised by the server, not here.
        self._asr: PcmTranscriber = asr or whisper_transcribe
        # Set once a fatal device error happened; the process is going down.
        self._fatal: Optional[str] = None
        # What to do after a fatal device error. Default: hard exit with
        # EXIT_RESTART_REQUIRED after a short grace period. Injectable for tests.
        self._on_fatal: Callable[[], None] = on_fatal or self._schedule_exit

    # -------------------------------------------------------------- lifecycle
    @property
    def max_workers(self) -> int:
        return len(self._workers)

    @property
    def queue_depth(self) -> int:
        return max(0, self._pending - self.max_workers)

    @property
    def supports_inline_reference(self) -> bool:
        """True only when EVERY worker can condition from memory.

        All-or-nothing because any worker may serve a request. ``False`` makes
        ``/tts`` answer 501 for inline requests and ``/health`` report it, which
        is what the plugin checks before it ever sends game audio (plan §5.3).
        """
        return bool(self._workers) and all(
            getattr(w, "supports_inline_reference", False) for w in self._workers
        )

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

        # A backend may run somewhere other than where it was told to: MOSS
        # pins itself to the CPU by default, because the GPU buys it nothing and
        # a game usually wants the card. Report what the workers ACTUALLY use —
        # `/health` and the `ready` event read this, and until this existed they
        # announced "device: cuda" while the model ran on the CPU.
        # `backend` deliberately keeps the detected value: the machine really
        # does have that GPU, this model just is not using it.
        effective = getattr(self._workers[0], "device", None) if self._workers else None
        if effective and effective != self.device:
            ndjson.log(
                f"{self._config.tts_backend} läuft auf {effective}, "
                f"obwohl {self.device} erkannt wurde",
            )
            self.device = effective

    async def _build_worker(self, worker_id: int, device: str) -> WorkerProtocol:
        assert self._loop is not None
        return await self._loop.run_in_executor(
            None, self._factory, worker_id, device
        )

    async def aclose(self) -> None:
        self._jobs.cancel_all()

    def health(self) -> dict:
        return {
            "status": "failed" if self._fatal else "ok",
            "fatal_error": self._fatal,
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
            # Plan game-voice-reference §2.4/§5.3: the plugin only sends game
            # audio to a server that says it keeps it in memory.
            "inline_reference": self.supports_inline_reference,
            "asr_check": True,
        }

    # ----------------------------------------------------------------- ASR
    def _transcribe(self, audio_path: Path) -> str:
        """Synchronous transcription using any worker (for ref-text fallback)."""
        if not self._workers:
            return ""
        return self._workers[0].transcribe(audio_path)

    def transcribe_pcm(self, audio: np.ndarray, sample_rate: int, language: Optional[str]) -> str:
        """In-memory ASR for ``/asr-check`` (blocking; run it in an executor)."""
        return self._asr(audio, sample_rate, language)

    # ------------------------------------------------------------- synthesis
    def admit(self) -> None:
        """Admission control (SPEC §9). Raises :class:`QueueFull` (503) or counts
        the request as pending. Must be paired with a single :meth:`stream` call,
        whose ``finally`` releases the pending slot.

        Raises :class:`EngineFailed` (503) once a fatal device error happened —
        the process is about to exit and the host restarts it."""
        if self._fatal is not None:
            raise EngineFailed(f"engine failed, restarting: {self._fatal}")
        if self._pending >= self.max_workers + self._config.max_queue:
            raise QueueFull("inference queue is full")
        self._pending += 1

    async def stream(
        self, job: Job, params: TtsParams, reference: Reference
    ) -> AsyncIterator[bytes]:
        """Async generator yielding PCM per sentence chunk (SPEC §6).

        Call :meth:`admit` and resolve the reference (so 503/400/404 surface
        before the HTTP 200) *before* iterating this. Resolves ref-text, chunks
        the text, acquires a worker with backpressure, then streams — honoring
        the cancel token and updating progress between chunks.

        ``reference`` is a sample path or an :class:`InlineReference`. The
        inline one brings its own text (the game text); the sample service —
        sidecar files, ASR fallback, its transcript cache — is never consulted
        for it, because that path writes ``.txt`` files next to the audio.
        """
        assert self._loop is not None
        if isinstance(reference, InlineReference):
            ref_text: Optional[str] = reference.text
            ref_label = reference.label
        else:
            ref_text = await self._loop.run_in_executor(
                None, self.samples.resolve_ref_text, reference, params.ref_text
            )
            ref_label = reference.name

        chunks = chunk_text(params.text, self._config.max_chars_per_chunk)
        self._jobs.set_total(job, len(chunks))

        ndjson.log(
            f"tts request start: job={job.job_id} sample={params.sample} "
            f"file={ref_label} lang={params.language} "
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
                    self._stream_chunk(worker, reference, ref_text, chunk, params)
                    if streaming
                    else self._one_shot_chunk(worker, reference, ref_text, chunk, params)
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
                    self._jobs.finish(job, ERROR, str(exc))
                    if is_fatal_device_error(exc):
                        # The CUDA context is dead for the whole process: a
                        # rebuild here always fails and every later request
                        # would 500. Exit so the host starts a fresh process.
                        self._enter_fatal(str(exc))
                        raise
                    # Isolate the crash: swap the dead worker for a fresh one so
                    # `finally` returns a healthy worker to the pool (SPEC §9).
                    worker = await self._rebuild_worker(worker)
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
        reference: Reference,
        ref_text: Optional[str],
        chunk: str,
        params: TtsParams,
    ) -> bytes:
        assert self._loop is not None
        try:
            wav = await self._loop.run_in_executor(
                None,
                worker.infer,
                _worker_ref(reference),
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
        reference: Reference,
        ref_text: Optional[str],
        chunk: str,
        params: TtsParams,
    ) -> AsyncIterator[bytes]:
        """Adapt the one-shot path (F5) to the same async-iterator shape as
        :meth:`_stream_chunk`, so :meth:`stream` has a single consume loop."""
        yield await self._infer_chunk(worker, reference, ref_text, chunk, params)

    async def _stream_chunk(
        self,
        worker: WorkerProtocol,
        reference: Reference,
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
            gen = worker.infer_stream(_worker_ref(reference), ref_text or "", chunk, params.speed, params.language)
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

    def _enter_fatal(self, message: str) -> None:
        """Mark the engine failed (idempotent), report it, and trigger the exit."""
        if self._fatal is not None:
            return
        self._fatal = message
        ndjson.error(
            f"fatal device error, the GPU context of this process is unusable — "
            f"exiting with code {EXIT_RESTART_REQUIRED} so the host restarts the "
            f"wrapper: {message}",
            fatal=True,
        )
        self._on_fatal()

    def _schedule_exit(self) -> None:
        if self._loop is not None:
            self._loop.call_later(FATAL_EXIT_DELAY_S, _hard_exit, EXIT_RESTART_REQUIRED)
        else:  # pragma: no cover — stream() always runs on a started engine
            _hard_exit(EXIT_RESTART_REQUIRED)

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
