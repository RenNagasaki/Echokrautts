import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import make_engine
from src.server import create_app


@pytest.fixture
def client(config):
    engine = make_engine(config)
    app = create_app(config=config, engine=engine)
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["backend"] == "cpu"
    assert body["language"] == "de"
    assert body["tts_backend"] == "f5"
    assert body["workers"] >= 1


def test_tts_language_match_ok(client):
    r = client.post("/tts", json={"sample": "anna_de.wav", "text": "Hallo.", "language": "de"})
    assert r.status_code == 200


def test_tts_f5_ignores_request_language(config):
    # F5 loads one finetune per process → a mismatched request language is not
    # rejected; it's ignored and the loaded (startup) model is used regardless.
    engine = make_engine(config)  # default tts_backend="f5"
    app = create_app(config=config, engine=engine)
    with TestClient(app) as c:
        r = c.post("/tts", json={"sample": "anna_de.wav", "text": "Hallo.", "language": "en"})
        assert r.status_code == 200
    # The worker (bound to the loaded model) is invoked with the startup language.
    assert engine._workers[0].languages == ["de"]


def test_tts_xtts_accepts_per_request_language(config):
    # XTTS is multilingual in one model → a per-request language is honored
    # (no reload) and forwarded to the worker verbatim.
    config.tts_backend = "xtts"
    engine = make_engine(config)
    app = create_app(config=config, engine=engine)
    with TestClient(app) as c:
        r = c.post("/tts", json={"sample": "anna_de.wav", "text": "Hello.", "language": "en"})
        assert r.status_code == 200
    assert engine._workers[0].languages == ["en"]


def test_tts_xtts_omitted_language_uses_startup(config):
    config.tts_backend = "xtts"
    engine = make_engine(config)
    app = create_app(config=config, engine=engine)
    with TestClient(app) as c:
        r = c.post("/tts", json={"sample": "anna_de.wav", "text": "Hallo."})
        assert r.status_code == 200
    # Falls back to the active/startup language (de).
    assert engine._workers[0].languages == ["de"]


def test_tts_xtts_unsupported_language_rejected(config):
    config.tts_backend = "xtts"
    engine = make_engine(config)
    app = create_app(config=config, engine=engine)
    with TestClient(app) as c:
        r = c.post("/tts", json={"sample": "anna_de.wav", "text": "x", "language": "xx"})
        assert r.status_code == 400
        assert "not supported" in r.json()["detail"]


def test_samples_list(client):
    r = client.get("/samples")
    assert r.status_code == 200
    assert r.json() == {"samples": ["anna_de.wav"]}


def test_samples_details(client):
    r = client.get("/samples", params={"details": "true"})
    assert r.status_code == 200
    entry = r.json()["samples"][0]
    assert entry["name"] == "anna_de.wav"
    assert "has_ref_text" in entry and "bytes" in entry


def test_tts_streams_pcm_with_headers(client):
    r = client.post("/tts", json={"sample": "anna_de.wav", "text": "Hallo. Welt."})
    assert r.status_code == 200
    assert r.headers["x-sample-rate"] == "24000"
    assert r.headers["x-channels"] == "1"
    assert r.headers["x-sample-format"] == "pcm_s16le"
    job_id = r.headers["x-job-id"]
    assert len(r.content) > 0
    assert len(r.content) % 2 == 0  # 16-bit samples
    # Job is queryable and finished.
    js = client.get(f"/jobs/{job_id}")
    assert js.status_code == 200
    assert js.json()["state"] == "done"


def test_tts_invalid_sample(client):
    r = client.post("/tts", json={"sample": "../escape.wav", "text": "x"})
    assert r.status_code == 400


def test_tts_missing_sample(client):
    r = client.post("/tts", json={"sample": "ghost.wav", "text": "x"})
    assert r.status_code == 404


def test_cancel_unknown_job(client):
    r = client.post("/cancel/does-not-exist")
    assert r.status_code == 404


def test_jobs_unknown(client):
    r = client.get("/jobs/nope")
    assert r.status_code == 404


def test_api_key_enforced(config):
    config.api_key = "secret"
    engine = make_engine(config)
    app = create_app(config=config, engine=engine)
    with TestClient(app) as c:
        # Missing key → 401 on a protected route.
        assert c.get("/samples").status_code == 401
        # Health stays open (no auth dependency).
        assert c.get("/health").status_code == 200
        # Correct key → ok.
        ok = c.get("/samples", headers={"Authorization": "Bearer secret"})
        assert ok.status_code == 200
