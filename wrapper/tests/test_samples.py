import os
import random

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


def test_resolve_non_audio_is_not_found(svc):
    # note.txt is not audio: its stem "note.txt" matches no folder/audio file.
    service, _ = svc
    with pytest.raises(SampleNotFound):
        service.resolve_path("note.txt")


def test_resolve_ignores_request_extension(svc):
    # Only anna_de.wav exists; requesting a different (or no) extension still hits
    # it because only the stem matters.
    service, sdir = svc
    want = (sdir / "anna_de.wav").resolve()
    assert service.resolve_path("anna_de") == want
    assert service.resolve_path("anna_de.mp3") == want


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


# ------------------------------------------------------------------ voice folders


@pytest.fixture
def folder_svc(tmp_path):
    """A voice folder ``alph/`` with three same-voice samples."""
    sdir = tmp_path / "samples"
    sdir.mkdir()
    vdir = sdir / "alph"
    vdir.mkdir()
    (vdir / "a.wav").write_bytes(b"RIFF__a_")
    (vdir / "b.wav").write_bytes(b"RIFF__b_")
    (vdir / "c.mp3").write_bytes(b"ID3__c__")
    cfg = Config(samples_dir=str(sdir))
    return cfg, sdir, vdir


def test_resolve_folder_picks_a_member(folder_svc):
    cfg, _, vdir = folder_svc
    members = {(vdir / n).resolve() for n in ("a.wav", "b.wav", "c.mp3")}
    # Deterministic chooser (last of the sorted list) → always c.mp3, but any pick
    # must be a real member of the folder.
    service = SampleService(cfg, chooser=lambda files: files[-1])
    got = service.resolve_path("alph")
    assert got in members
    assert got == (vdir / "c.mp3").resolve()  # sorted: a.wav, b.wav, c.mp3


def test_resolve_folder_extension_ignored(folder_svc):
    cfg, _, _ = folder_svc
    service = SampleService(cfg, chooser=lambda files: files[0])
    # A trailing ".wav" is stripped to the stem "alph", which is the folder.
    assert service.resolve_path("alph.wav").name == "a.wav"


def test_resolve_folder_random_covers_all_members(folder_svc):
    cfg, _, vdir = folder_svc
    service = SampleService(cfg, chooser=random.Random(1234).choice)
    seen = {service.resolve_path("alph").name for _ in range(60)}
    assert seen == {"a.wav", "b.wav", "c.mp3"}


def test_folder_wins_over_same_stem_file(tmp_path):
    sdir = tmp_path / "samples"
    sdir.mkdir()
    (sdir / "alph.wav").write_bytes(b"SINGLE__")  # should be shadowed
    vdir = sdir / "alph"
    vdir.mkdir()
    (vdir / "inside.wav").write_bytes(b"RIFF____")
    cfg = Config(samples_dir=str(sdir))
    service = SampleService(cfg, chooser=lambda files: files[0])
    assert service.resolve_path("alph") == (vdir / "inside.wav").resolve()
    # Listing shows the folder (by name), not the shadowed file.
    assert service.list_samples()["samples"] == ["alph"]


def test_resolve_empty_folder_not_found(tmp_path):
    sdir = tmp_path / "samples"
    sdir.mkdir()
    (sdir / "empty").mkdir()
    (sdir / "empty" / "readme.txt").write_text("no audio", encoding="utf-8")
    cfg = Config(samples_dir=str(sdir))
    service = SampleService(cfg)
    with pytest.raises(SampleNotFound):
        service.resolve_path("empty")


def test_folder_not_recursive(tmp_path):
    sdir = tmp_path / "samples"
    sdir.mkdir()
    vdir = sdir / "alph"
    vdir.mkdir()
    (vdir / "top.wav").write_bytes(b"RIFF____")
    nested = vdir / "nested"
    nested.mkdir()
    (nested / "deep.wav").write_bytes(b"RIFF____")
    cfg = Config(samples_dir=str(sdir))
    service = SampleService(cfg, chooser=lambda files: files[0])
    # Only the top-level file is a candidate; the nested one is never reached.
    for _ in range(10):
        assert service.resolve_path("alph") == (vdir / "top.wav").resolve()


def test_list_details_folder(folder_svc):
    cfg, _, vdir = folder_svc
    (vdir / "b.txt").write_text("ref for b", encoding="utf-8")
    service = SampleService(cfg)
    entry = {e["name"]: e for e in service.list_samples(details=True)["samples"]}["alph"]
    assert entry["count"] == 3
    assert entry["has_ref_text"] is True  # at least one member has a sidecar
    assert entry["bytes"] == 24  # 3 × 8 bytes


def test_ref_text_from_sidecar_in_folder(folder_svc):
    cfg, _, vdir = folder_svc
    (vdir / "b.txt").write_text("Hallo aus b", encoding="utf-8")
    service = SampleService(cfg, chooser=lambda files: files[1])  # -> b.wav
    path = service.resolve_path("alph")
    assert path.name == "b.wav"
    assert service.resolve_ref_text(path) == "Hallo aus b"


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
