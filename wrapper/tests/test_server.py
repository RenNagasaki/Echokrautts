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
    assert body["xtts_fp16"] is False  # default off, f5 backend, cpu device
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


# ------------------------------------------------------------- wav + web UI
def test_tts_wav_format_returns_a_playable_file(client):
    import io
    import wave

    r = client.post(
        "/tts", json={"sample": "anna_de.wav", "text": "Hallo.", "format": "wav"}
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    with wave.open(io.BytesIO(r.content)) as fh:
        assert fh.getnchannels() == 1
        assert fh.getsampwidth() == 2
        assert fh.getnframes() > 0


def test_tts_wav_carries_the_same_metadata_headers(client):
    r = client.post(
        "/tts", json={"sample": "anna_de.wav", "text": "Hallo.", "format": "wav"}
    )
    assert r.headers["X-Job-Id"]
    assert r.headers["X-Sample-Rate"] == "24000"


def test_tts_defaults_to_raw_pcm(client):
    r = client.post("/tts", json={"sample": "anna_de.wav", "text": "Hallo."})
    assert r.headers["content-type"].startswith("audio/pcm")
    assert not r.content.startswith(b"RIFF")


def test_tts_wav_holds_the_same_audio_as_the_pcm_stream(client):
    body = {"sample": "anna_de.wav", "text": "Hallo."}
    pcm = client.post("/tts", json=body).content
    blob = client.post("/tts", json=dict(body, format="wav")).content
    assert blob[44:] == pcm  # header prepended, payload untouched


def test_unknown_format_is_a_400(client):
    r = client.post(
        "/tts", json={"sample": "anna_de.wav", "text": "Hallo.", "format": "mp3"}
    )
    assert r.status_code == 400
    assert "format" in r.json()["detail"]


def test_languages_locked_for_f5(client):
    # F5 loads one finetune per process → the UI must grey the selector out.
    body = client.get("/languages").json()
    assert body["locked"] is True
    assert body["options"] == ["de"]
    assert body["active"] == "de"


def test_languages_open_for_xtts(config):
    config.tts_backend = "xtts"
    app = create_app(config=config, engine=make_engine(config))
    with TestClient(app) as c:
        body = c.get("/languages").json()
    assert body["locked"] is False
    assert body["active"] == "de"
    assert "en" in body["options"] and "ja" in body["options"]
    assert body["options"] == sorted(body["options"])


def test_ui_is_served_and_needs_no_api_key(config):
    # You must be able to load the page in order to type the key into it.
    config.api_key = "secret"
    app = create_app(config=config, engine=make_engine(config))
    with TestClient(app) as c:
        r = c.get("/")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        assert "Echokrautts" in r.text
        # …while the endpoints it calls stay protected.
        assert c.get("/samples").status_code == 401


# ---------------------------------------------------------------- rate limit
def test_rate_limit_off_by_default(client):
    for _ in range(5):
        r = client.post("/tts", json={"sample": "anna_de.wav", "text": "Hallo."})
        assert r.status_code == 200
    assert client.get("/health").json()["rate_limit"] == {"enabled": False}


def test_global_rate_limit_returns_429_with_retry_after(config):
    config.rate_limit_per_hour = 2
    app = create_app(config=config, engine=make_engine(config))
    with TestClient(app) as c:
        body = {"sample": "anna_de.wav", "text": "Hallo."}
        assert c.post("/tts", json=body).status_code == 200
        assert c.post("/tts", json=body).status_code == 200
        r = c.post("/tts", json=body)
        assert r.status_code == 429
        assert int(r.headers["Retry-After"]) > 0
        assert "rate limit" in r.json()["detail"]


def test_rate_limit_does_not_touch_other_endpoints(config):
    # The limit protects GPU time, so browsing voices must stay possible even
    # once /tts is exhausted — otherwise the web UI locks itself out.
    config.rate_limit_per_hour = 1
    app = create_app(config=config, engine=make_engine(config))
    with TestClient(app) as c:
        body = {"sample": "anna_de.wav", "text": "Hallo."}
        assert c.post("/tts", json=body).status_code == 200
        assert c.post("/tts", json=body).status_code == 429
        assert c.get("/samples").status_code == 200
        assert c.get("/languages").status_code == 200
        assert c.get("/health").status_code == 200
        assert c.get("/").status_code == 200


def test_health_reports_rate_limit_usage(config):
    config.rate_limit_per_hour = 5
    app = create_app(config=config, engine=make_engine(config))
    with TestClient(app) as c:
        c.post("/tts", json={"sample": "anna_de.wav", "text": "Hallo."})
        snap = c.get("/health").json()["rate_limit"]
    assert snap["enabled"] is True
    assert snap["per_hour"] == 5
    assert snap["used_this_window"] == 1


def test_a_rejected_request_never_reaches_the_engine(config):
    # 429 is decided before admission, so a hammering caller cannot fill the
    # queue and turn everyone else's requests into 503s.
    config.rate_limit_per_hour = 1
    engine = make_engine(config)
    app = create_app(config=config, engine=engine)
    with TestClient(app) as c:
        body = {"sample": "anna_de.wav", "text": "Hallo."}
        c.post("/tts", json=body)
        before = engine._pending
        assert c.post("/tts", json=body).status_code == 429
        assert engine._pending == before


def test_untrusted_forwarded_for_cannot_dodge_the_per_ip_limit(config):
    config.rate_limit_per_ip_per_hour = 1
    app = create_app(config=config, engine=make_engine(config))
    with TestClient(app) as c:
        body = {"sample": "anna_de.wav", "text": "Hallo."}
        assert c.post("/tts", json=body).status_code == 200
        # A caller inventing a fresh address per request must not get a fresh
        # bucket while trust_forwarded_for is off.
        r = c.post("/tts", json=body, headers={"X-Forwarded-For": "9.9.9.9"})
        assert r.status_code == 429


# ---------------------------------------------------------------------------
# Generation parameter bounds
#
# Both values reach the engine unchanged and are now editable in the built-in
# web UI. A speed of 0 is undefined rather than slow, and a huge nfe_step holds
# a worker (the pool hands out exactly one per request) for minutes.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("speed", [0, -1, 3.5])
def test_tts_rejects_impossible_speed(client, speed):
    r = client.post("/tts", json={"sample": "anna_de.wav", "text": "Hallo.", "speed": speed})
    assert r.status_code == 422


@pytest.mark.parametrize("nfe", [0, -8, 100000])
def test_tts_rejects_impossible_nfe_step(client, nfe):
    r = client.post("/tts", json={"sample": "anna_de.wav", "text": "Hallo.", "nfe_step": nfe})
    assert r.status_code == 422


def test_tts_still_accepts_the_useful_range(client):
    """The guard rail must not narrow what people actually use.

    nfe 16 is the cheap-quality setting worth A/B-ing on a weak GPU: it halves
    the flow-matching passes and therefore the GPU work.
    """
    r = client.post(
        "/tts",
        json={"sample": "anna_de.wav", "text": "Hallo.", "speed": 0.5, "nfe_step": 16},
    )
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# MOSS-TTS-Nano: multilingual in one model, and it streams
# ---------------------------------------------------------------------------

def test_languages_open_for_moss(config):
    config.tts_backend = "moss"
    engine = make_engine(config)
    with TestClient(create_app(config=config, engine=engine)) as c:
        body = c.get("/languages").json()
    assert body["locked"] is False
    assert {"en", "ja", "de", "fr"} <= set(body["options"])


def test_tts_moss_accepts_per_request_language(config):
    config.tts_backend = "moss"
    engine = make_engine(config)
    with TestClient(create_app(config=config, engine=engine)) as c:
        r = c.post("/tts", json={"sample": "anna_de.wav", "text": "Hallo.", "language": "ja"})
    assert r.status_code == 200


def test_tts_moss_rejects_an_untrained_language(config):
    """MOSS infers the language from the text, so a bad code would not crash —
    it would return confident nonsense. Rejecting it is the useful behaviour."""
    config.tts_backend = "moss"
    engine = make_engine(config)
    with TestClient(create_app(config=config, engine=engine)) as c:
        r = c.post("/tts", json={"sample": "anna_de.wav", "text": "Hi.", "language": "zh-cn"})
    assert r.status_code == 400
    assert "MOSS-TTS-Nano" in r.json()["detail"]
