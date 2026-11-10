"""Inline voice references and /asr-check (plan game-voice-reference §2.4).

The point of the feature is a guarantee — reference audio stays in RAM — so
most tests here check what must NOT happen: no file in samples/ or anywhere
else, no sidecar transcript, no ASR fallback, no path handed to a worker.
"""

from __future__ import annotations

import base64
import sys
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import make_engine
from src import inline_ref
from src.config import Config
from src.inline_ref import (
    ConditioningCache,
    InlineReference,
    InvalidInlineReference,
    build_reference,
    decode_pcm16,
    ensure_sentence_end,
    reference_key,
)
from src.server import create_app

SR = inline_ref.INLINE_SAMPLE_RATE


def pcm_b64(seconds: float, freq: float = 220.0) -> str:
    t = np.arange(int(seconds * SR)) / SR
    pcm = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    return base64.b64encode(pcm.tobytes()).decode("ascii")


def files_under(path) -> set:
    return {p for p in path.rglob("*")}


@pytest.fixture
def client(config):
    engine = make_engine(config)
    with TestClient(create_app(config=config, engine=engine)) as c:
        c.engine = engine
        yield c


# ------------------------------------------------------------------ decoding


def test_decode_pcm16_round_trips_to_float():
    pcm = np.array([0, 16384, -32768, 32767], dtype="<i2")
    out = decode_pcm16(base64.b64encode(pcm.tobytes()).decode())
    assert out.dtype == np.float32
    assert out.tolist() == pytest.approx([0.0, 0.5, -1.0, 32767 / 32768])


@pytest.mark.parametrize("bad", ["", "not base64!!", base64.b64encode(b"\x01\x02\x03").decode()])
def test_decode_pcm16_rejects_malformed_input(bad):
    with pytest.raises(InvalidInlineReference):
        decode_pcm16(bad)


def test_build_reference_requires_game_text():
    with pytest.raises(InvalidInlineReference, match="ref_text"):
        build_reference(pcm_b64(3), "  ")


def test_build_reference_bounds_the_length():
    with pytest.raises(InvalidInlineReference, match="at least"):
        build_reference(pcm_b64(0.5), "Hallo.")
    with pytest.raises(InvalidInlineReference, match="at most"):
        build_reference(pcm_b64(16), "Hallo.", max_seconds=15.0)
    ref = build_reference(pcm_b64(4), " Hallo. ", max_seconds=15.0)
    assert ref.seconds == pytest.approx(4.0)
    assert ref.text == "Hallo."


def test_key_is_the_hash_of_the_clip_ids():
    a = np.zeros(10, dtype=np.float32)
    b = np.ones(10, dtype=np.float32)
    # Same clip ids → same voice, whatever the bytes (the plan's cache key).
    assert reference_key(["c1", "c2"], a, "x") == reference_key(["c1", "c2"], b, "y")
    # Order and boundaries matter: ["ab"] is not ["a", "b"].
    assert reference_key(["a", "b"], a, "x") != reference_key(["ab"], a, "x")
    assert reference_key(["c1", "c2"], a, "x") != reference_key(["c2", "c1"], a, "x")


def test_key_without_ids_depends_on_audio_and_text():
    a = np.zeros(10, dtype=np.float32)
    b = np.ones(10, dtype=np.float32)
    assert reference_key(None, a, "x") == reference_key([], a, "x")
    assert reference_key(None, a, "x") != reference_key(None, b, "x")
    assert reference_key(None, a, "x") != reference_key(None, a, "y")


def test_label_never_contains_text():
    ref = build_reference(pcm_b64(2), "Geheimer Spieltext.")
    assert ref.label.startswith("inline:")
    assert "Spieltext" not in ref.label
    assert "Spieltext" not in repr(ref)


@pytest.mark.parametrize(
    "given, expected",
    [("Hallo", "Hallo. "), ("Hallo.", "Hallo. "), ("Hallo. ", "Hallo. "), ("こんにちは。", "こんにちは。")],
)
def test_ensure_sentence_end(given, expected):
    assert ensure_sentence_end(given) == expected


# --------------------------------------------------------------------- cache


def test_cache_computes_once_and_evicts_least_recently_used():
    cache: ConditioningCache = ConditioningCache(2)
    calls = []

    def make(k):
        return lambda: calls.append(k) or k.upper()

    assert cache.get_or_compute("a", make("a")) == "A"
    assert cache.get_or_compute("a", make("a")) == "A"
    cache.get_or_compute("b", make("b"))
    cache.get_or_compute("a", make("a"))  # a is now most recent
    cache.get_or_compute("c", make("c"))  # evicts b
    assert calls == ["a", "b", "c"]
    assert "a" in cache and "c" in cache and "b" not in cache


def test_cache_size_zero_disables_caching():
    cache: ConditioningCache = ConditioningCache(0)
    calls = []
    cache.get_or_compute("a", lambda: calls.append(1))
    cache.get_or_compute("a", lambda: calls.append(1))
    assert len(calls) == 2 and len(cache) == 0


# -------------------------------------------------------------------- /tts


def test_health_reports_inline_reference(client):
    body = client.get("/health").json()
    assert body["inline_reference"] is True
    assert body["asr_check"] is True


def test_inline_tts_hands_the_object_not_a_path_and_writes_nothing(client, config, tmp_path):
    before = files_under(tmp_path)
    r = client.post(
        "/tts",
        json={"text": "Guten Tag.", "ref_audio": pcm_b64(3), "ref_text": "Ein Spieltext", "ref_ids": ["c1"]},
    )
    assert r.status_code == 200
    assert len(r.content) > 0
    ref, text = client.engine._workers[0].refs[-1]
    assert isinstance(ref, InlineReference)
    assert text == "Ein Spieltext"
    # Nothing new on disk: not in samples/, not a sidecar, not anywhere under
    # the test root (models/ included).
    assert files_under(tmp_path) == before


def test_inline_tts_never_consults_the_sample_service(config, monkeypatch):
    config.asr_for_missing_ref_text = True
    engine = make_engine(config)

    def boom(*_a, **_k):
        raise AssertionError("sample service must not be used for inline references")

    monkeypatch.setattr(engine.samples, "resolve_ref_text", boom)
    monkeypatch.setattr(engine, "_transcribe", boom)
    with TestClient(create_app(config=config, engine=engine)) as c:
        r = c.post("/tts", json={"text": "Hallo.", "ref_audio": pcm_b64(2), "ref_text": "Text."})
    assert r.status_code == 200


def test_inline_tts_streaming_worker_gets_the_object(config):
    engine = make_engine(config, streaming=True)
    with TestClient(create_app(config=config, engine=engine)) as c:
        r = c.post("/tts", json={"text": "Hallo.", "ref_audio": pcm_b64(2), "ref_text": "Text."})
    assert r.status_code == 200
    assert isinstance(engine._workers[0].refs[-1][0], InlineReference)


def test_inline_tts_wav_format(client):
    r = client.post(
        "/tts", json={"text": "Hallo.", "ref_audio": pcm_b64(2), "ref_text": "Text.", "format": "wav"}
    )
    assert r.status_code == 200
    assert r.content[:4] == b"RIFF"


def test_sample_requests_still_pass_a_path(client):
    r = client.post("/tts", json={"sample": "anna_de.wav", "text": "Hallo."})
    assert r.status_code == 200
    ref, _ = client.engine._workers[0].refs[-1]
    assert isinstance(ref, str) and ref.endswith("anna_de.wav")


@pytest.mark.parametrize(
    "body",
    [
        {"text": "Hallo."},  # neither
        {"text": "Hallo.", "sample": "anna_de.wav", "ref_audio": pcm_b64(2), "ref_text": "T"},  # both
        {"text": "Hallo.", "ref_audio": pcm_b64(2)},  # no game text
        {"text": "Hallo.", "ref_audio": "%%%", "ref_text": "T"},  # not base64
        {"text": "Hallo.", "ref_audio": pcm_b64(20), "ref_text": "T"},  # too long
    ],
)
def test_inline_tts_validation_is_a_400(client, body):
    r = client.post("/tts", json=body)
    assert r.status_code == 400, r.text
    assert client.engine._workers[0].infer_calls == 0


def test_inline_max_seconds_is_configurable(config):
    config.inline_ref_max_seconds = 25.0
    engine = make_engine(config)
    with TestClient(create_app(config=config, engine=engine)) as c:
        r = c.post("/tts", json={"text": "Hallo.", "ref_audio": pcm_b64(20), "ref_text": "T"})
    assert r.status_code == 200


def test_backend_without_inline_support_answers_501(config):
    engine = make_engine(config, inline=False)
    with TestClient(create_app(config=config, engine=engine)) as c:
        assert c.get("/health").json()["inline_reference"] is False
        r = c.post("/tts", json={"text": "Hallo.", "ref_audio": pcm_b64(2), "ref_text": "T"})
        assert r.status_code == 501
        # File samples keep working on that backend.
        assert c.post("/tts", json={"sample": "anna_de.wav", "text": "Hallo."}).status_code == 200


async def test_inline_support_is_all_or_nothing(config):
    from conftest import FakeWorker, cpu_detection
    from src.engine import Engine
    from src.jobs import JobRegistry

    flags = iter([True, False])
    engine = Engine(
        config,
        cpu_detection(max_workers_hint=2),
        JobRegistry(),
        worker_factory=lambda i, d: FakeWorker(inline=next(flags)),
    )
    config.max_workers = 2
    await engine.start()
    assert engine.supports_inline_reference is False


def test_log_line_names_the_voice_not_the_text(client, capsys):
    capsys.readouterr()
    client.post("/tts", json={"text": "Hallo.", "ref_audio": pcm_b64(2), "ref_text": "Streng geheim."})
    out = capsys.readouterr().out
    assert "file=inline:" in out
    assert "Streng geheim" not in out


def test_inline_tts_requires_api_key(config):
    config.api_key = "s3cret"
    engine = make_engine(config)
    with TestClient(create_app(config=config, engine=engine)) as c:
        body = {"text": "Hallo.", "ref_audio": pcm_b64(2), "ref_text": "T"}
        assert c.post("/tts", json=body).status_code == 401
        assert c.post("/tts", json=body, headers={"Authorization": "Bearer s3cret"}).status_code == 200


# --------------------------------------------------------------- /asr-check


def test_asr_check_transcribes_from_memory(config, tmp_path):
    seen = []

    def fake_asr(audio, sr, lang):
        seen.append((audio.shape, audio.dtype, sr, lang))
        return "erkannter text"

    engine = make_engine(config, asr=fake_asr)
    before = files_under(tmp_path)
    with TestClient(create_app(config=config, engine=engine)) as c:
        r = c.post("/asr-check", json={"audio": pcm_b64(2), "language": "de"})
    assert r.status_code == 200
    assert r.json() == {"text": "erkannter text", "seconds": 2.0}
    assert seen == [((2 * SR,), np.float32, SR, "de")]
    assert files_under(tmp_path) == before


@pytest.mark.parametrize("audio", ["", "%%%", None])
def test_asr_check_rejects_bad_audio(client, audio):
    body = {} if audio is None else {"audio": audio}
    r = client.post("/asr-check", json=body)
    assert r.status_code in (400, 422)


def test_asr_check_bounds_the_length(client):
    assert client.post("/asr-check", json={"audio": pcm_b64(31)}).status_code == 400


def test_asr_failure_is_a_500_and_the_server_keeps_serving(config):
    def broken(audio, sr, lang):
        raise RuntimeError("whisper died")

    engine = make_engine(config, asr=broken)
    with TestClient(create_app(config=config, engine=engine)) as c:
        r = c.post("/asr-check", json={"audio": pcm_b64(2)})
        assert r.status_code == 500
        assert "whisper died" in r.json()["detail"]
        assert c.get("/health").status_code == 200


def test_asr_check_requires_api_key(config):
    config.api_key = "s3cret"
    engine = make_engine(config)
    with TestClient(create_app(config=config, engine=engine)) as c:
        assert c.post("/asr-check", json={"audio": pcm_b64(2)}).status_code == 401


# ------------------------------------------------------------ real workers
#
# torch / f5-tts / coqui / onnxruntime are not in the test venv, so these drive
# the workers' inline paths with stand-ins and check what this repo decides:
# which library entry point is used (never a path-taking one), what it is fed,
# and that conditioning is cached per voice.


class _Tensor:
    """Just enough of a torch tensor for the F5 inline path."""

    def __init__(self, arr):
        self.arr = np.asarray(arr)

    def unsqueeze(self, dim):
        return _Tensor(np.expand_dims(self.arr, dim))

    @property
    def shape(self):
        return self.arr.shape


def test_f5_inline_bypasses_the_temp_file_preprocessing(monkeypatch):
    from src.engine import F5TTSWorker

    calls = {}

    def infer_batch_process(ref_audio, ref_text, batches, model_obj, vocoder, **kw):
        calls["ref_audio"], calls["ref_text"], calls["batches"], calls["kw"] = (
            ref_audio, ref_text, batches, kw
        )
        yield np.ones(100, dtype=np.float32), 24000, None

    fake_torch = types.ModuleType("torch")
    fake_torch.from_numpy = _Tensor
    utils = types.ModuleType("f5_tts.infer.utils_infer")
    utils.infer_batch_process = infer_batch_process
    utils.chunk_text = lambda text, max_chars: calls.setdefault("max_chars", max_chars) and [text]

    def no_preprocess(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("preprocess_ref_audio_text writes a temp WAV")

    utils.preprocess_ref_audio_text = no_preprocess
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "f5_tts", types.ModuleType("f5_tts"))
    monkeypatch.setitem(sys.modules, "f5_tts.infer", types.ModuleType("f5_tts.infer"))
    monkeypatch.setitem(sys.modules, "f5_tts.infer.utils_infer", utils)

    worker = object.__new__(F5TTSWorker)
    worker.sample_rate = 24000
    worker._inline_cache = ConditioningCache(4)

    def no_path_api(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("F5TTS.infer takes a path")

    worker._model = types.SimpleNamespace(
        infer=no_path_api, ema_model="ema", vocoder="voc", mel_spec_type="vocos", device="cpu"
    )
    ref = build_reference(pcm_b64(4), "Ein Spieltext")

    out = worker.infer(ref, "", "Guten Tag.", 16, 1.0)

    assert out.shape == (100,)
    audio, sr = calls["ref_audio"]
    assert sr == SR and audio.shape == (1, 4 * SR)
    assert calls["ref_text"] == "Ein Spieltext. "
    assert calls["batches"] == ["Guten Tag."]
    assert calls["kw"]["nfe_step"] == 16 and calls["kw"]["progress"] is None
    # infer_process' budget: bytes(ref_text) / ref_s * (22 - ref_s) * speed
    assert calls["max_chars"] == int(len(b"Ein Spieltext. ") / 4 * 18)
    # The prepared tensor is cached per voice.
    assert ref.voice_id in worker._inline_cache


def test_xtts_inline_conditioning_is_cached_per_voice_and_never_by_path():
    from src import xtts_backend

    class Latent:
        def __init__(self, dtype="float32"):
            self.dtype = dtype

        def half(self):
            return Latent("float16")

    def no_path(audio_path):  # pragma: no cover - must never run
        raise AssertionError("get_conditioning_latents loads from a path")

    worker = object.__new__(xtts_backend.XTTSWorker)
    worker._model = types.SimpleNamespace(get_conditioning_latents=no_path)
    worker._fp16 = True
    worker._cond_cache = {}
    worker._inline_cache = ConditioningCache(4)
    computed = []
    worker._inline_latents = lambda ref: computed.append(ref.voice_id) or (Latent(), Latent())

    ref = build_reference(pcm_b64(3), "Text.", ref_ids=["c1"])
    first = worker._conditioning(ref)
    again = worker._conditioning(build_reference(pcm_b64(3, freq=440), "Text.", ref_ids=["c1"]))

    assert computed == [ref.voice_id]  # same clip ids → computed once
    assert first is again
    assert [t.dtype for t in first] == ["float16", "float16"]  # fp16 cast applies
    assert worker._cond_cache == {}  # the path cache stays untouched


def test_xtts_inline_uses_coqui_defaults():
    """The inline latents must match what the file path computes, so the
    reference length/rate constants have to equal coqui's defaults."""
    from src import xtts_backend

    assert xtts_backend.XTTS_LOAD_SR == 22050
    assert xtts_backend.XTTS_GPT_COND_LEN == 6
    assert xtts_backend.XTTS_GPT_COND_CHUNK_LEN == 6


def test_moss_onnx_inline_encodes_from_memory_and_caches(monkeypatch):
    from src import moss_onnx_backend

    def no_path(**kw):  # pragma: no cover - must never run for inline
        raise AssertionError("resolve_prompt_audio_codes loads from a path")

    rt = types.SimpleNamespace(
        codec_meta={"codec_config": {"sample_rate": 48000}},
        prepare_synthesis_text=lambda **kw: {"text": kw["text"]},
        resolve_prompt_audio_codes=no_path,
        manifest={"generation_defaults": {}},
        split_voice_clone_text=lambda text, max_tokens: [text],
        encode_text=lambda t: [1, 2],
        build_voice_clone_request_rows=lambda codes, text: ("rows", tuple(map(tuple, codes))),
    )
    worker = object.__new__(moss_onnx_backend.MossOnnxWorker)
    worker._runtime = rt
    worker._max_new_frames = 10
    worker._inline_cache = ConditioningCache(4)
    encoded = []
    worker._encode_inline = lambda ref: encoded.append(ref.voice_id) or [[7, 8]]
    rows_seen = []
    worker._decode_streaming = lambda rows, sr: rows_seen.append(rows) or iter(())

    ref = build_reference(pcm_b64(3), "Text.")
    list(worker._raw_chunks(ref, "Hallo."))
    list(worker._raw_chunks(ref, "Noch einmal."))

    assert encoded == [ref.voice_id]
    assert rows_seen == [("rows", ((7, 8),)), ("rows", ((7, 8),))]


def test_inline_support_per_runtime():
    from src import moss_backend, moss_onnx_backend, xtts_backend
    from src.engine import F5TTSWorker

    assert F5TTSWorker.supports_inline_reference is True
    assert xtts_backend.XTTSWorker.supports_inline_reference is True
    assert moss_onnx_backend.MossOnnxWorker.supports_inline_reference is True
    # The PyTorch MOSS runtime only takes a path deep inside the vendor service.
    assert moss_backend.MossWorker.supports_inline_reference is False


def test_config_inline_fields_coerce_from_env():
    from src.config import load_config

    cfg = load_config(
        argv=[],
        config_path=__import__("pathlib").Path("/nonexistent.json"),
        env={"F5W_INLINE_REF_MAX_SECONDS": "12.5", "F5W_INLINE_REF_CACHE_SIZE": "3"},
    )
    assert cfg.inline_ref_max_seconds == 12.5
    assert cfg.inline_ref_cache_size == 3
    assert Config().inline_ref_max_seconds == 15.0
