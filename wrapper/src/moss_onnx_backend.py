"""MOSS-TTS-Nano on ONNX Runtime — the same model, roughly 2.5x faster.

Same weights as :mod:`moss_backend`, exported. That is the whole point: every
other way of speeding MOSS up that was measured (int8 quantization, fewer
codebooks, bfloat16) trades audio for time, and this one does not.

**Measured on a Ryzen Zen4, same sentence, same seed, ms per 80 ms frame — so
anything under 80 is faster than real time:**

| runtime      | 1 thread | 4 threads | 8 threads | 16 threads |
|--------------|----------|-----------|-----------|------------|
| PyTorch fp32 |    161.5 |     115.4 |     100.3 |       95.5 |
| ONNX         |     63.3 |      41.4 |      35.9 |       95.7 |

ONNX on ONE core beats PyTorch on sixteen. That matters here more than the
headline factor: MOSS exists for people running a game on the same machine, so
the cores it does *not* take are the point.

Note the 16-thread column. Past roughly 8 intra-op threads onnxruntime spends
more time coordinating than computing and gives the whole win back, which is
why ``moss_onnx_threads`` defaults low instead of following the core count.

Everything else matches :class:`moss_backend.MossWorker` — 24 kHz mono float32,
cloning from audio with no transcript, and the same language caveat: MOSS takes
no language argument, so the reference clip's language decides the accent.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from . import hfcache, ndjson
from .config import Config
from .inline_ref import ConditioningCache, InlineReference
from .moss_audio import ContractResampler
from .moss_common import DEFAULT_SAMPLE_RATE, MossWorkerBase

# The two Hugging Face repos holding the exported graphs. Separate from the
# PyTorch ones, and about 763 MB together against 312 MB for the checkpoints —
# the one real cost of this path.
ONNX_TTS_REPO = "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX"
ONNX_CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"

# Where they land under models/. Underscores like the PyTorch directories: not
# strictly needed here, but one convention is cheaper to keep than a rule about
# which directory may hold which character.
ONNX_DIRNAME = "moss_onnx"


def onnx_model_dir(config: Config) -> Path:
    return Path(config.models_path) / ONNX_DIRNAME


def download_model(config: Config) -> Path:
    """Fetch both ONNX repos into ``models/moss_onnx``.

    Delegates to the vendor's own downloader instead of reimplementing it: that
    function knows the file patterns AND normalises the directory layout
    afterwards, and a hand-rolled copy would rot silently when either changes.
    It is private, so a missing symbol is reported as what it is rather than
    surfacing as an AttributeError from inside a download.
    """
    hfcache.use_models_dir(config)
    import onnx_tts_runtime as runtime

    target = onnx_model_dir(config)
    target.mkdir(parents=True, exist_ok=True)
    fetch = getattr(runtime, "_download_default_browser_onnx_assets", None)
    if fetch is None:
        raise RuntimeError(
            "Die MOSS-ONNX-Laufzeit kennt keinen Download mehr "
            "(_download_default_browser_onnx_assets fehlt). Hole "
            f"{ONNX_TTS_REPO} und {ONNX_CODEC_REPO} von Hand nach {target}."
        )
    fetch(target)
    return target


def is_available(config: Config) -> tuple[bool, str]:
    """Can this machine run the ONNX path? Returns (usable, reason if not).

    Two independent things can be missing — the package and the weights — and a
    caller that wants to fall back has to be able to say which, or the user
    reads "not available" and has no idea what to install.
    """
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False, "onnxruntime ist nicht installiert"
    target = onnx_model_dir(config)
    try:
        import onnx_tts_runtime as runtime

        runtime.ensure_browser_onnx_model_dir(str(target))
    except Exception as exc:  # noqa: BLE001 — anything here means "not usable"
        return False, f"ONNX-Modelldateien fehlen in {target} ({type(exc).__name__})"
    return True, ""


def resolve_threads(config: Config) -> int:
    """Intra-op threads: the configured value, or a deliberately modest default.

    Measured, this is not "more is better": 16 threads were as slow as one and
    slower than four. Four is also the polite choice on a machine that is
    running a game, which is the situation this engine exists for.
    """
    configured = int(getattr(config, "moss_onnx_threads", 0) or 0)
    if configured > 0:
        return configured
    return max(1, min(4, os.cpu_count() or 1))


class MossOnnxWorker(MossWorkerBase):
    """One ONNX MOSS runtime, behind the same protocol as every other worker."""

    supports_streaming = True
    # The codec encoder is an ONNX session fed a waveform array, so a reference
    # can be encoded straight from memory (see _encode_inline).
    supports_inline_reference = True

    def __init__(self, config: Config, device: str):
        hfcache.use_models_dir(config)
        import onnx_tts_runtime as runtime

        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.supports_streaming = bool(config.moss_stream)
        self._max_new_frames = int(config.moss_max_new_frames)
        self._resampler = ContractResampler()
        # Prompt codes of inline references, per voice (RAM only, bounded).
        self._inline_cache: ConditioningCache = ConditioningCache(config.inline_ref_cache_size)
        # A CPU runtime by design. The GPU measured SLOWER than the CPU for this
        # model — the time goes into per-call overhead, not arithmetic — and
        # leaving the card to the game is the reason MOSS is offered at all.
        self.device = "cpu"
        threads = resolve_threads(config)
        self._runtime = runtime.OnnxTtsRuntime(
            model_dir=str(onnx_model_dir(config)),
            thread_count=threads,
            output_dir=str(Path(config.models_path) / "moss-scratch"),
        )
        ndjson.log_once(f"MOSS-TTS-Nano (ONNX) geladen, {threads} Threads")

    # ---------------------------------------------------------------- protocol

    def infer_stream(
        self,
        ref_file: str,
        ref_text: str,
        gen_text: str,
        speed: float,
        language: str | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield float32 chunks as the ONNX runtime produces them.

        The vendor decodes incrementally but hands the audio back only once the
        clip is finished, so the pieces exist and are simply not offered. This
        drives the generation on a thread and takes the pieces off its callback
        through a queue — the smallest bridge from "calls you back" to "is a
        generator" — while keeping the decode-budget policy the vendor chose.

        ``ref_text`` and ``language`` are unused (the model infers the language
        from the text); ``speed`` has no equivalent and is reported once rather
        than silently dropped, matching the PyTorch worker.
        """
        self.warn_unsupported_speed(speed)
        self._resampler.reset()
        for chunk, rate in self._raw_chunks(ref_file, gen_text):
            converted = self._resampler.to_contract(chunk, rate, self.sample_rate)
            if converted.size:
                yield converted


    # ------------------------------------------------------------------ inside

    def _raw_chunks(self, ref_file: str, gen_text: str):
        """(channels-first chunk, sample rate) as the runtime decodes them."""
        rt = self._runtime
        sample_rate = int(rt.codec_meta["codec_config"]["sample_rate"])
        # enable_wetext=False on purpose: WeTextProcessing needs pynini, which
        # has no Windows wheels, so this wrapper deliberately does not ship it.
        # It only normalises zh/en text anyway.
        prepared = rt.prepare_synthesis_text(
            text=gen_text, voice="", enable_wetext=False, enable_normalize_tts_text=True
        )
        if isinstance(ref_file, InlineReference):
            prompt_codes = self._inline_cache.get_or_compute(
                ref_file.voice_id, lambda: self._encode_inline(ref_file)
            )
        else:
            prompt_codes = rt.resolve_prompt_audio_codes(voice=None, prompt_audio_path=ref_file)
        rt.manifest["generation_defaults"]["max_new_frames"] = self._max_new_frames

        for text_chunk in rt.split_voice_clone_text(str(prepared["text"]), max_tokens=75):
            rows = rt.build_voice_clone_request_rows(prompt_codes, rt.encode_text(text_chunk))
            yield from self._decode_streaming(rows, sample_rate)

    def _encode_inline(self, ref: InlineReference) -> list[list[int]]:
        """Prompt codes for an in-memory reference — the vendor's
        ``encode_reference_audio`` minus its ``torchaudio.load``.

        The vendor method only takes a path; its body is a resample/channel
        conversion followed by one ``codec_encode`` session run, and both work
        on arrays. Repeated here (pinned MOSS commit, ``onnx_tts_runtime.py``
        ``_load_reference_audio`` + ``encode_reference_audio``) rather than
        written to a temp file, because the reference may be game audio.
        """
        import torch  # lazy
        import torchaudio  # lazy

        rt = self._runtime
        codec = rt.codec_meta["codec_config"]
        target_rate = int(codec["sample_rate"])
        target_channels = int(codec["channels"])
        waveform = torch.from_numpy(ref.audio.astype("float32", copy=False)).unsqueeze(0)
        if ref.sample_rate != target_rate:
            waveform = torchaudio.functional.resample(waveform, ref.sample_rate, target_rate)
        if target_channels > 1:
            waveform = waveform.repeat(target_channels, 1)
        batch = waveform.unsqueeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        session = rt.sessions["codec_encode"]
        outputs = session.run(
            None,
            {"waveform": batch, "input_lengths": np.asarray([batch.shape[-1]], dtype=np.int32)},
        )
        named = dict(zip([o.name for o in session.get_outputs()], outputs))
        codes = np.asarray(named["audio_codes"], dtype=np.int32)
        length = int(np.asarray(named["audio_code_lengths"]).reshape(-1)[0])
        quantizers = int(codec["num_quantizers"])
        return [[int(codes[0, f, q]) for q in range(quantizers)] for f in range(length)]

    def _decode_streaming(self, request_rows, sample_rate: int):
        import onnx_tts_runtime as runtime

        pending: list[list[int]] = []
        emitted = {"samples": 0, "first_at": None}
        out: queue.Queue = queue.Queue()
        session = self._runtime.codec_streaming_session
        session.reset()

        def flush(force: bool) -> None:
            while pending:
                budget = runtime._resolve_stream_decode_frame_budget(
                    emitted["samples"], sample_rate, emitted["first_at"]
                )
                budget = max(1, int(budget))
                if not force and len(pending) < budget:
                    return
                take = len(pending) if force else min(len(pending), budget)
                frames = pending[:take]
                del pending[:take]
                decoded = session.run_frames(frames)
                if decoded is not None:
                    audio, length = decoded
                    if length > 0:
                        if emitted["first_at"] is None:
                            emitted["first_at"] = time.perf_counter()
                        emitted["samples"] += int(length)
                        # (1, channels, samples) -> channels-first for the resampler
                        out.put(np.asarray(audio[0, :, :length], dtype=np.float32))
                if not force:
                    return

        def on_frame(_frames, _step, frame) -> None:
            pending.append(list(frame))
            flush(False)

        def produce() -> None:
            try:
                self._runtime.generate_audio_frames(request_rows, on_frame=on_frame)
                flush(True)
            except BaseException as exc:  # noqa: BLE001 — handed to the consumer
                out.put(exc)
            finally:
                out.put(None)

        thread = threading.Thread(target=produce, name="moss-onnx-decode", daemon=True)
        thread.start()
        try:
            while True:
                item = out.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item, sample_rate
        finally:
            thread.join(timeout=30)
            session.reset()


if __name__ == "__main__":
    # Entry point used by the bootstrap (run inside the venv): downloads the
    # exported graphs into models/moss_onnx.
    import sys
    import traceback

    from .config import load_config

    try:
        path = download_model(load_config())
        ndjson.log(f"MOSS-ONNX-Modell bereit ({path})")
    except Exception as exc:  # noqa: BLE001 — surface as a fatal NDJSON error
        traceback.print_exc(file=sys.stderr)
        ndjson.error(f"moss onnx model download failed: {exc}", fatal=True)
        raise SystemExit(1)
