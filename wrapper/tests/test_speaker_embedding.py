"""POST /speaker-embedding (plan game-voice-reference §11.3).

Audio in, speaker vector out, nothing stored — the plugin groups the clips of
one speaker by these vectors to find voice-actor changes.
"""

from __future__ import annotations

import sys
import threading
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import make_engine
from src.server import create_app
from test_inline_ref import SR, files_under, pcm_b64


def _speaker(seen=None, vec=(3.0, 4.0)):
    def fn(audio, sr):
        if seen is not None:
            seen.append((audio.shape, audio.dtype, sr))
        return np.array(vec)

    return ("fake-encoder", fn)


def test_health_reports_the_encoder(config):
    engine = make_engine(config, speaker=_speaker())
    with TestClient(create_app(config=config, engine=engine)) as c:
        body = c.get("/health").json()
    assert body["speaker_embedding"] is True
    assert body["speaker_encoder"] == "fake-encoder"


def test_health_without_encoder(client_without_encoder):
    body = client_without_encoder.get("/health").json()
    assert body["speaker_embedding"] is False
    assert body["speaker_encoder"] is None


@pytest.fixture
def client_without_encoder(config):
    engine = make_engine(config)
    with TestClient(create_app(config=config, engine=engine)) as c:
        yield c


def test_embedding_is_normalised_and_names_the_encoder(config, tmp_path):
    seen = []
    engine = make_engine(config, speaker=_speaker(seen))
    before = files_under(tmp_path)
    with TestClient(create_app(config=config, engine=engine)) as c:
        r = c.post("/speaker-embedding", json={"audio": pcm_b64(2)})
    assert r.status_code == 200
    body = r.json()
    assert body["embedding"] == pytest.approx([0.6, 0.8])
    assert body["encoder"] == "fake-encoder"
    assert body["dim"] == 2
    assert body["seconds"] == 2.0
    assert seen == [((2 * SR,), np.float32, SR)]
    assert files_under(tmp_path) == before  # nothing written


def test_engine_without_encoder_answers_501(client_without_encoder):
    r = client_without_encoder.post("/speaker-embedding", json={"audio": pcm_b64(2)})
    assert r.status_code == 501


def test_workers_with_different_encoders_count_as_unsupported(config):
    engine = make_engine(config, workers=2, speaker=_speaker())
    with TestClient(create_app(config=config, engine=engine)) as c:
        engine._workers[1].speaker_encoder = "other"
        assert engine.speaker_encoder is None
        assert c.post("/speaker-embedding", json={"audio": pcm_b64(2)}).status_code == 501


@pytest.mark.parametrize("seconds", [0.5, 31])
def test_length_is_bounded(config, seconds):
    engine = make_engine(config, speaker=_speaker())
    with TestClient(create_app(config=config, engine=engine)) as c:
        assert c.post("/speaker-embedding", json={"audio": pcm_b64(seconds)}).status_code == 400


@pytest.mark.parametrize("audio", ["", "%%%"])
def test_bad_audio_is_a_400(config, audio):
    engine = make_engine(config, speaker=_speaker())
    with TestClient(create_app(config=config, engine=engine)) as c:
        assert c.post("/speaker-embedding", json={"audio": audio}).status_code == 400


def test_a_zero_vector_is_a_500_and_the_server_keeps_serving(config):
    engine = make_engine(config, speaker=_speaker(vec=(0.0, 0.0)))
    with TestClient(create_app(config=config, engine=engine)) as c:
        r = c.post("/speaker-embedding", json={"audio": pcm_b64(2)})
        assert r.status_code == 500
        assert c.get("/health").status_code == 200


def test_encoder_failure_is_a_500_and_frees_the_worker(config):
    def broken(audio, sr):
        raise RuntimeError("encoder died")

    engine = make_engine(config, speaker=("fake-encoder", broken))
    with TestClient(create_app(config=config, engine=engine)) as c:
        r = c.post("/speaker-embedding", json={"audio": pcm_b64(2)})
        assert r.status_code == 500
        assert "encoder died" in r.json()["detail"]
        # The borrowed worker went back: a TTS request still gets one.
        assert c.post("/tts", json={"text": "Hallo.", "ref_audio": pcm_b64(3), "ref_text": "Hi."}).status_code == 200


def test_requires_api_key(config):
    config.api_key = "s3cret"
    engine = make_engine(config, speaker=_speaker())
    with TestClient(create_app(config=config, engine=engine)) as c:
        assert c.post("/speaker-embedding", json={"audio": pcm_b64(2)}).status_code == 401
        ok = c.post(
            "/speaker-embedding", json={"audio": pcm_b64(2)}, headers={"Authorization": "Bearer s3cret"}
        )
        assert ok.status_code == 200


# ------------------------------------------------------------ real workers


def test_qwen_uses_the_models_own_encoder_and_names_the_model():
    from test_qwen_backend import _worker

    worker, model = _worker()
    calls = []

    def extract(audio, sr):
        calls.append((audio.dtype, audio.shape, sr))
        return np.array([1.0, 2.0])

    model.model.model = types.SimpleNamespace(extract_speaker_embedding=extract, speaker_encoder_sample_rate=24000)
    vec = worker.speaker_embedding(np.zeros(SR, dtype=np.float64), SR)
    assert vec.tolist() == [1.0, 2.0]
    assert calls == [(np.float32, (SR,), 24000)]
    assert worker.speaker_encoder == "qwen-0.6b-xvector"


def test_xtts_resamples_to_its_encoder_rate(monkeypatch):
    from src import xtts_backend

    seen = {}

    class _T:
        def __init__(self, a):
            self.a = a

        def unsqueeze(self, _):
            return self

        def clip(self, lo, hi):
            return self

        def to(self, device):
            return self

    def resample(t, src, dst):
        seen["rates"] = (src, dst)
        return t

    class _Vec:
        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.ones((1, 512, 1), dtype=np.float32)

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    fake_torch = types.SimpleNamespace(from_numpy=_T, inference_mode=_Ctx)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules, "torchaudio", types.SimpleNamespace(functional=types.SimpleNamespace(resample=resample))
    )
    worker = xtts_backend.XTTSWorker.__new__(xtts_backend.XTTSWorker)
    worker._busy = threading.Lock()  # set by __init__, skipped here
    worker._model = types.SimpleNamespace(
        device="cpu", get_speaker_embedding=lambda t, sr: seen.setdefault("sr", sr) and _Vec()
    )
    vec = worker.speaker_embedding(np.zeros(SR, dtype=np.float32), SR)
    assert seen["rates"] == (SR, xtts_backend.XTTS_LOAD_SR)
    assert seen["sr"] == xtts_backend.XTTS_LOAD_SR
    assert vec.shape == (512,)
