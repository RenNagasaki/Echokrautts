"""First-start voice pack download (src/voicepack.py).

No network here: the GitHub API and the asset download are served by a fake
opener, and the "zip" is built in memory.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from src import voicepack
from src.config import Config

PREFIX = "EK-VoicePack-"


@pytest.fixture
def config(tmp_path):
    cfg = Config()
    cfg.samples_dir = str(tmp_path / "samples")
    return cfg


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


def _opener(releases, zip_payload, calls=None):
    def opener(request):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        if calls is not None:
            calls.append(url)
        if "api.github.com" in url:
            return _FakeResponse(json.dumps(releases).encode())
        return _FakeResponse(zip_payload)

    return opener


def _release(tag, *, asset="pack.zip", prerelease=False, draft=False):
    return {
        "tag_name": tag,
        "prerelease": prerelease,
        "draft": draft,
        "assets": [
            {
                "name": asset,
                "size": 1234,
                "browser_download_url": f"https://example/{tag}/{asset}",
            }
        ]
        if asset
        else [],
    }


# ------------------------------------------------------------- version order
@pytest.mark.parametrize(
    "tag,expected",
    [
        ("EK-VoicePack-1.0.0", (1, 0, 0)),
        ("EK-VoicePack-1.10.0", (1, 10, 0)),
        ("EK-VoicePack-2", (2,)),
        ("EK-VoicePack-1.2.3-beta", (1, 2, 3, 0)),
    ],
)
def test_version_key(tag, expected):
    assert voicepack.version_key(tag, PREFIX) == expected


def test_ten_sorts_above_nine():
    # The whole reason for a numeric key: as strings, "1.10.0" < "1.9.0".
    releases = [_release("EK-VoicePack-1.9.0"), _release("EK-VoicePack-1.10.0")]
    assert voicepack.pick_release(releases, PREFIX)["tag_name"] == "EK-VoicePack-1.10.0"


def test_plugin_releases_are_ignored():
    # The repo publishes plugin releases too — and the newest release overall is
    # usually one of them, which is why GitHub's /releases/latest is unusable.
    releases = [
        _release("2.1.4.0"),  # a plugin release, published later
        _release("EK-VoicePack-1.1.0"),
    ]
    assert voicepack.pick_release(releases, PREFIX)["tag_name"] == "EK-VoicePack-1.1.0"


def test_prereleases_and_drafts_are_skipped():
    releases = [
        _release("EK-VoicePack-2.0.0", prerelease=True),
        _release("EK-VoicePack-1.9.0", draft=True),
        _release("EK-VoicePack-1.1.0"),
    ]
    assert voicepack.pick_release(releases, PREFIX)["tag_name"] == "EK-VoicePack-1.1.0"


def test_no_matching_release():
    assert voicepack.pick_release([_release("2.1.4.0")], PREFIX) is None


# ----------------------------------------------------------------- extraction
def test_extract_writes_files(config, tmp_path):
    payload = _zip_bytes({"a.wav": b"RIFF", "a.txt": b"hello"})
    count = voicepack.extract_zip(payload, config.samples_path)
    assert count == 2
    assert (config.samples_path / "a.wav").read_bytes() == b"RIFF"
    assert (config.samples_path / "a.txt").read_text() == "hello"


def test_extract_keeps_subfolders(config):
    # Voice folders are a supported layout, so nested entries must survive.
    payload = _zip_bytes({"anna/one.wav": b"a", "anna/two.wav": b"b"})
    voicepack.extract_zip(payload, config.samples_path)
    assert (config.samples_path / "anna" / "two.wav").read_bytes() == b"b"


def test_zip_slip_is_refused(config, tmp_path):
    # A crafted archive must not write outside the samples folder.
    payload = _zip_bytes({"../escaped.wav": b"x", "ok.wav": b"y"})
    count = voicepack.extract_zip(payload, config.samples_path)
    assert count == 1
    assert (config.samples_path / "ok.wav").exists()
    assert not (tmp_path / "escaped.wav").exists()


def test_absolute_entry_is_refused(config):
    payload = _zip_bytes({"/abs.wav": b"x"})
    assert voicepack.extract_zip(payload, config.samples_path) == 0


# --------------------------------------------------------------- has_samples
def test_missing_folder_counts_as_empty(config):
    assert voicepack.has_samples(config) is False


def test_sidecars_alone_do_not_count(config):
    config.samples_path.mkdir(parents=True)
    (config.samples_path / "a.txt").write_text("only a transcript")
    assert voicepack.has_samples(config) is False


def test_audio_anywhere_counts(config):
    (config.samples_path / "anna").mkdir(parents=True)
    (config.samples_path / "anna" / "clip.wav").write_bytes(b"RIFF")
    assert voicepack.has_samples(config) is True


# ------------------------------------------------------------------- end2end
def test_ensure_downloads_into_an_empty_folder(config):
    payload = _zip_bytes({"voice.wav": b"RIFF", "voice.txt": b"text"})
    calls: list[str] = []
    opener = _opener([_release("EK-VoicePack-1.1.0")], payload, calls)
    assert voicepack.ensure_voicepack(config, opener) is True
    assert (config.samples_path / "voice.wav").exists()
    assert any("api.github.com" in c for c in calls)
    assert any("1.1.0" in c for c in calls)


def test_ensure_skips_when_samples_exist(config):
    config.samples_path.mkdir(parents=True)
    (config.samples_path / "mine.wav").write_bytes(b"RIFF")

    def explode(_request):
        raise AssertionError("must not hit the network when samples exist")

    assert voicepack.ensure_voicepack(config, explode) is False


def test_ensure_respects_the_off_switch(config):
    config.voicepack_auto_download = False

    def explode(_request):
        raise AssertionError("must not download when disabled")

    assert voicepack.ensure_voicepack(config, explode) is False


def test_network_failure_is_not_fatal(config):
    def explode(_request):
        raise OSError("no route to host")

    # The wrapper must still start; the user can drop in their own sample.
    assert voicepack.ensure_voicepack(config, explode) is False
    assert not voicepack.has_samples(config)


def test_corrupt_download_is_not_fatal(config):
    opener = _opener([_release("EK-VoicePack-1.1.0")], b"this is not a zip")
    assert voicepack.ensure_voicepack(config, opener) is False


def test_release_without_zip_asset_is_not_fatal(config):
    opener = _opener([_release("EK-VoicePack-1.1.0", asset=None)], b"")
    assert voicepack.ensure_voicepack(config, opener) is False


def test_partial_download_file_is_cleaned_up(config, tmp_path):
    payload = _zip_bytes({"voice.wav": b"RIFF"})
    voicepack.ensure_voicepack(config, _opener([_release("EK-VoicePack-1.1.0")], payload))
    leftovers = list(tmp_path.glob(".voicepack-*"))
    assert leftovers == []


def test_partial_file_is_cleaned_up_after_a_bad_zip(config, tmp_path):
    opener = _opener([_release("EK-VoicePack-1.1.0")], b"not a zip at all")
    assert voicepack.ensure_voicepack(config, opener) is False
    # A leftover .part would be mistaken for a finished download next time.
    assert list(tmp_path.glob(".voicepack-*")) == []
