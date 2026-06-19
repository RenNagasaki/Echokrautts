import os

import pytest

from src.config import Config
from src.samples import InvalidSample, SampleNotFound, SampleService


@pytest.fixture
def svc(tmp_path):
    sdir = tmp_path / "samples"
    sdir.mkdir()
    (sdir / "anna_de.wav").write_bytes(b"RIFF")
    (sdir / "tom_en.flac").write_bytes(b"fLaC")
    (sdir / "note.txt").write_text("not audio", encoding="utf-8")
    cfg = Config(samples_dir=str(sdir))
    return SampleService(cfg), sdir


def test_list_names(svc):
    service, _ = svc
    assert service.list_samples() == {"samples": ["anna_de.wav", "tom_en.flac"]}


def test_list_details(svc):
    service, sdir = svc
    (sdir / "anna_de.txt").write_text("hallo", encoding="utf-8")
    out = service.list_samples(details=True)["samples"]
    by_name = {e["name"]: e for e in out}
    assert by_name["anna_de.wav"]["has_ref_text"] is True
    assert by_name["tom_en.flac"]["has_ref_text"] is False
    assert by_name["anna_de.wav"]["bytes"] == 4


def test_resolve_valid(svc):
    service, sdir = svc
    assert service.resolve_path("anna_de.wav") == (sdir / "anna_de.wav").resolve()


@pytest.mark.parametrize("name", ["../secret.wav", "sub/anna_de.wav", "..", "."])
def test_resolve_rejects_traversal(svc, name):
    service, _ = svc
    with pytest.raises(InvalidSample):
        service.resolve_path(name)


def test_resolve_rejects_absolute(svc, tmp_path):
    service, _ = svc
    with pytest.raises(InvalidSample):
        service.resolve_path(str(tmp_path / "anna_de.wav"))


def test_resolve_rejects_disallowed_ext(svc):
    service, _ = svc
    with pytest.raises(InvalidSample):
        service.resolve_path("note.txt")


def test_resolve_missing(svc):
    service, _ = svc
    with pytest.raises(SampleNotFound):
        service.resolve_path("ghost.wav")


def test_ref_text_from_sidecar(svc):
    service, sdir = svc
    (sdir / "anna_de.txt").write_text("Hallo Welt", encoding="utf-8")
    path = service.resolve_path("anna_de.wav")
    assert service.resolve_ref_text(path) == "Hallo Welt"


def test_ref_text_from_request(svc):
    service, _ = svc
    path = service.resolve_path("anna_de.wav")
    assert service.resolve_ref_text(path, request_ref_text="aus Request") == "aus Request"


def test_ref_text_via_asr_and_cache(tmp_path):
    sdir = tmp_path / "samples"
    sdir.mkdir()
    (sdir / "anna_de.wav").write_bytes(b"RIFF")
    cfg = Config(samples_dir=str(sdir), asr_for_missing_ref_text=True)
    calls = {"n": 0}

    def transcriber(p):
        calls["n"] += 1
        return "asr text"

    service = SampleService(cfg, transcriber=transcriber)
    path = service.resolve_path("anna_de.wav")
    assert service.resolve_ref_text(path) == "asr text"
    # Sidecar written + in-memory cache → no second ASR call.
    assert (sdir / "anna_de.txt").read_text(encoding="utf-8") == "asr text"
    assert service.resolve_ref_text(path) == "asr text"
    assert calls["n"] == 1


def test_ref_text_none_when_asr_disabled(tmp_path):
    sdir = tmp_path / "samples"
    sdir.mkdir()
    (sdir / "anna_de.wav").write_bytes(b"RIFF")
    cfg = Config(samples_dir=str(sdir), asr_for_missing_ref_text=False)
    service = SampleService(cfg, transcriber=lambda p: "x")
    path = service.resolve_path("anna_de.wav")
    assert service.resolve_ref_text(path) is None


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs privilege on Windows")
def test_resolve_rejects_symlink_escape(tmp_path):
    sdir = tmp_path / "samples"
    sdir.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    (sdir / "link.wav").symlink_to(outside)
    cfg = Config(samples_dir=str(sdir))
    service = SampleService(cfg)
    with pytest.raises(InvalidSample):
        service.resolve_path("link.wav")
