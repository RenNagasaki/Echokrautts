"""The release archive builder.

The layout is not a preference, it is a contract with the C# host: it extracts
this archive into ``<installRoot>/echokrautts`` and runs
``bootstrap/bootstrap.py`` from there. A wrong prefix, a missing module or a
stray ``.venv`` are all silent failures — the archive builds, uploads, and only
breaks on a user's machine. Hence these tests.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "build-release-zip.py"

# The whole module asks git what to package, so without git there is nothing to
# assert. This matters in CI: the pipeline runs on `python:3.12-slim`, where the
# checkout is done by the runner's helper container and the job image itself has
# no git binary at all.
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git on PATH")


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_release_zip_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    if not SCRIPT.is_file():
        pytest.skip("build script not present")
    return _load_builder()


@pytest.fixture(scope="module")
def archive(builder, tmp_path_factory):
    try:
        out = builder.build(tmp_path_factory.mktemp("dist") / "EchokrauTTS.zip")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("needs a git checkout with git on PATH")
    with zipfile.ZipFile(out) as zf:
        yield {"path": out, "names": zf.namelist(), "zip": zf}


def test_contents_sit_at_the_archive_root(archive):
    """No ``wrapper/`` prefix: the host extracts INTO the install directory."""
    assert "bootstrap/bootstrap.py" in archive["names"]
    assert not any(n.startswith("wrapper/") for n in archive["names"])


def test_everything_the_wrapper_needs_to_run_is_present(archive):
    """A missing file here is a broken install, not a failed build."""
    for required in (
        "bootstrap/bootstrap.py",
        "bootstrap/install_win.ps1",
        "bootstrap/install_linux.sh",
        "config.json",
        "pyproject.toml",
        "src/__init__.py",
        "src/server.py",
        "src/engine.py",
        "src/static/index.html",  # the built-in web UI 404s without it
    ):
        assert required in archive["names"], required


def test_runtime_state_and_build_dirt_stay_out(archive):
    """These are gitignored, so asking git is what keeps them out."""
    for name in archive["names"]:
        assert "__pycache__" not in name, name
        assert not name.startswith(".venv"), name
        assert not name.startswith(".state"), name
        assert not name.startswith("models/"), name
        assert not name.endswith(".zip"), name  # never package the archive itself


def test_samples_folder_is_shipped_empty(archive):
    """The voice pack lands here at first start; the folder must exist."""
    assert "samples/.gitkeep" in archive["names"]
    assert not [n for n in archive["names"] if n.startswith("samples/") and n != "samples/.gitkeep"]


def test_uncommitted_files_are_packaged_too(builder, tmp_path, monkeypatch):
    """The trap this replaced: `git ls-files` alone drops brand-new modules.

    A release built right after writing a file would then be missing exactly
    the code the release was for.
    """
    wrapper = tmp_path / "wrapper"
    (wrapper / "src").mkdir(parents=True)
    (wrapper / "src" / "brand_new.py").write_text("x = 1", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

    assert "src/brand_new.py" in builder.collect_files(wrapper)


def test_gitignored_files_are_not_packaged(builder, tmp_path):
    """The flip side: 'would git track it' is the whole inclusion rule."""
    wrapper = tmp_path / "wrapper"
    (wrapper / "src").mkdir(parents=True)
    (wrapper / "src" / "keep.py").write_text("x = 1", encoding="utf-8")
    (wrapper / "src" / "secret.log").write_text("noise", encoding="utf-8")
    (wrapper / ".gitignore").write_text("*.log\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

    files = builder.collect_files(wrapper)
    assert "src/keep.py" in files
    assert "src/secret.log" not in files


def test_build_is_reproducible(builder, tmp_path):
    """Two builds of the same sources must be byte-identical.

    Otherwise 'has the package actually changed?' cannot be answered by
    comparing two files, which is the only cheap check a human release does.
    """
    try:
        first = builder.build(tmp_path / "one.zip").read_bytes()
        second = builder.build(tmp_path / "two.zip").read_bytes()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("needs a git checkout with git on PATH")
    assert first == second


def test_no_partial_file_survives_a_build(archive):
    """An interrupted build must not leave something that looks finished."""
    assert not list(archive["path"].parent.glob("*.part"))


def test_archive_entries_are_readable(archive):
    """A zip whose entries cannot be read is a zip nobody can install."""
    assert archive["zip"].testzip() is None
    assert archive["zip"].read("config.json").strip().startswith(b"{")
