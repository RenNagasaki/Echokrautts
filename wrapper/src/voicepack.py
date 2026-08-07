"""Fetch the Echokraut voice pack when the samples folder is empty.

Without at least one sample the wrapper is useless — every ``/tts`` request
answers 404. That is the state every fresh install starts in, and every fresh
container too, because ``samples`` is a mounted volume. So on first start the
wrapper pulls the current voice pack from the Echokraut releases and unpacks it
into the samples folder.

Two things about that are easy to get wrong and are handled here:

* **"Latest" is not GitHub's ``/releases/latest``.** That repository publishes
  plugin releases as well, and its "latest" is whichever release was published
  last — usually a plugin build. The voice packs are identified by their tag
  prefix (``EK-VoicePack-``) and compared numerically per segment, so
  ``1.10.0`` sorts above ``1.9.0`` (a plain string compare gets that wrong).
* **A zip may not write outside its target.** Archive entries are sanitised
  before extraction; anything absolute, anything containing ``..`` and anything
  resolving outside the samples folder is skipped rather than trusted.

Nothing here may be fatal: no network, a rate-limited API or a corrupt download
must leave the wrapper starting normally (with an empty samples folder) instead
of refusing to run.
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

from . import ndjson
from .config import Config

GITHUB_API = "https://api.github.com"
# GitHub rejects API requests without one.
USER_AGENT = "echokrautts-wrapper"


class VoicePackError(RuntimeError):
    """Anything that stops the download. Always caught by ``ensure_voicepack``."""


def _http_get(url: str, opener: Optional[Callable] = None) -> bytes:
    opener = opener or urllib.request.urlopen
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with opener(request) as response:  # noqa: S310 — fixed https endpoints
        return response.read()


def version_key(tag: str, prefix: str) -> tuple[int, ...]:
    """Sort key for a release tag: the numeric parts of its version suffix.

    ``EK-VoicePack-1.10.0`` → ``(1, 10, 0)``. Segment-wise and numeric on
    purpose — as a string, "1.10.0" would sort below "1.9.0".
    """
    suffix = tag[len(prefix):] if tag.startswith(prefix) else tag
    parts: list[int] = []
    for chunk in suffix.replace("-", ".").split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def pick_release(releases: list[dict], prefix: str) -> Optional[dict]:
    """Newest non-prerelease whose tag starts with ``prefix``, or None."""
    candidates = [
        r
        for r in releases
        if str(r.get("tag_name", "")).startswith(prefix)
        and not r.get("prerelease")
        and not r.get("draft")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: version_key(r["tag_name"], prefix))


def pick_asset(release: dict) -> Optional[dict]:
    """The zip asset of a release (the packs ship exactly one)."""
    for asset in release.get("assets", []):
        if str(asset.get("name", "")).lower().endswith(".zip"):
            return asset
    return None


def has_samples(config: Config) -> bool:
    """True if the samples folder already holds a usable voice.

    Only audio counts: a folder containing nothing but leftover ``.txt``
    sidecars (or a stray ``.gitkeep``) is still an unusable install.
    """
    directory = config.samples_path
    if not directory.is_dir():
        return False
    exts = config.normalized_exts
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            return True
    return False


def _safe_members(archive: zipfile.ZipFile, target: Path) -> list[zipfile.ZipInfo]:
    """Entries that provably stay inside ``target`` (zip-slip guard)."""
    safe: list[zipfile.ZipInfo] = []
    resolved_target = target.resolve()
    for info in archive.infolist():
        if info.is_dir():
            continue
        name = info.filename.replace("\\", "/")
        if name.startswith("/") or ".." in Path(name).parts or Path(name).is_absolute():
            ndjson.log(f"voice pack: skipping unsafe entry {info.filename!r}", level="warning")
            continue
        destination = (resolved_target / name).resolve()
        if not destination.is_relative_to(resolved_target):
            ndjson.log(f"voice pack: skipping escaping entry {info.filename!r}", level="warning")
            continue
        safe.append(info)
    return safe


def _download_to(url: str, destination: Path, total: int, opener: Optional[Callable]) -> None:
    """Stream an asset to disk, reporting progress.

    Streamed rather than held in memory: the pack is >100 MB, and a machine that
    is already loading multi-GB models should not also carry the archive twice.
    Progress is emitted every ~5% so the host has something to show — a silent
    two-minute pause on first start looks like a hang.
    """
    opener = opener or urllib.request.urlopen
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    done = 0
    last_percent = -5
    with opener(request) as response, open(destination, "wb") as handle:  # noqa: S310
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            handle.write(block)
            done += len(block)
            if total > 0:
                percent = int(done * 100 / total)
                if percent >= last_percent + 5:
                    last_percent = percent
                    ndjson.progress(0, 1, "voicepack", "Lade Voice-Pack …", percent=percent)


def extract_zip(source, target: Path) -> int:
    """Unpack a voice-pack zip into ``target``. Returns the file count.

    ``source`` is either the archive's bytes or a path to it.
    """
    target.mkdir(parents=True, exist_ok=True)
    handle = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    with zipfile.ZipFile(handle) as archive:
        members = _safe_members(archive, target)
        for info in members:
            name = info.filename.replace("\\", "/")
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, open(destination, "wb") as handle:
                handle.write(source.read())
    return len(members)


def download_voicepack(config: Config, opener: Optional[Callable] = None) -> int:
    """Download and unpack the newest voice pack. Raises on any failure."""
    prefix = config.voicepack_tag_prefix
    api = f"{GITHUB_API}/repos/{config.voicepack_repo}/releases?per_page=100"
    ndjson.progress(0, 1, "voicepack", "Suche aktuelles Voice-Pack …")
    try:
        releases = json.loads(_http_get(api, opener))
    except Exception as exc:  # noqa: BLE001 — network/JSON/rate limit all land here
        raise VoicePackError(f"release list unavailable: {exc}") from exc
    if not isinstance(releases, list):
        raise VoicePackError(f"unexpected release payload: {type(releases).__name__}")

    release = pick_release(releases, prefix)
    if release is None:
        raise VoicePackError(f"no release tagged {prefix}* in {config.voicepack_repo}")
    asset = pick_asset(release)
    if asset is None:
        raise VoicePackError(f"release {release['tag_name']} has no zip asset")

    size = int(asset.get("size") or 0)
    ndjson.progress(
        0, 1, "voicepack", f"Lade {release['tag_name']} ({size / 1_048_576:.0f} MB) …"
    )
    # Downloaded next to the samples folder, not into it: a partial file must
    # never be mistaken for content, and the temp file is removed either way.
    config.samples_path.mkdir(parents=True, exist_ok=True)
    archive_path = config.samples_path.parent / f".voicepack-{release['tag_name']}.part"
    try:
        _download_to(asset["browser_download_url"], archive_path, size, opener)
        count = extract_zip(archive_path, config.samples_path)
    except zipfile.BadZipFile as exc:
        raise VoicePackError(f"downloaded file is not a zip: {exc}") from exc
    except VoicePackError:
        raise
    except Exception as exc:  # noqa: BLE001 — network, disk, permissions
        raise VoicePackError(f"download failed: {exc}") from exc
    finally:
        archive_path.unlink(missing_ok=True)

    ndjson.log(
        f"voice pack {release['tag_name']} installed: {count} files in {config.samples_path}"
    )
    return count


def ensure_voicepack(config: Config, opener: Optional[Callable] = None) -> bool:
    """Install the voice pack if the samples folder has no audio yet.

    Returns True when something was installed. **Never raises**: a missing voice
    pack is a wrapper with no voices, which is recoverable by dropping in a wav
    — a wrapper that refuses to start is not.
    """
    if not config.voicepack_auto_download:
        return False
    if has_samples(config):
        return False
    try:
        download_voicepack(config, opener)
        return True
    except VoicePackError as exc:
        ndjson.log(
            f"voice pack not installed ({exc}); put your own samples in "
            f"{config.samples_path} — synthesis needs at least one",
            level="warning",
        )
        return False


if __name__ == "__main__":  # pragma: no cover — install-time entry point
    from .config import load_config

    ensure_voicepack(load_config([]))
