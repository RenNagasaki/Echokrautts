#!/usr/bin/env python3
"""Build ``wrapper/EchokrauTTS.zip`` — the asset attached to a GitHub release.

Run it, then create the release by hand and upload the file:

    python build-release-zip.py

Layout is dictated by the C# host, not chosen here: it downloads this archive
and extracts it into ``<installRoot>/echokrautts``, then runs
``bootstrap/bootstrap.py`` from that directory. So the archive holds the
CONTENTS of ``wrapper/`` at its root — no ``wrapper/`` prefix — exactly like the
asset published with 0.0.0.4. Extracting it over an existing install therefore
replaces the code and leaves ``.venv``, ``.state``, ``models`` and ``samples``
(everything the bootstrap creates) untouched.

WHAT GOES IN is asked of git, never hand-listed: every file under ``wrapper/``
that git tracks OR would track (``--others --exclude-standard``). That is one
decision with three consequences worth stating:

* a new module is in the archive the moment it exists on disk — the reason not
  to use ``git ls-files`` alone, which silently drops files not yet committed
  and would ship a package missing exactly the code somebody just wrote;
* everything ``.gitignore`` covers stays out for free — ``.venv``, ``.state``,
  ``models``, downloaded ``samples``, ``__pycache__`` and this archive itself;
* nothing here needs updating when the wrapper gains a file.

The 0.0.0.4 asset did contain ``__pycache__`` directories with compiled 3.12
bytecode. That is build dirt, not structure: it is ~180 KB of the 226 KB
archive, it is stale the moment a source file changes, and on a machine running
a different Python it is dead weight the interpreter ignores. It is left out.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
WRAPPER_DIR = REPO_ROOT / "wrapper"
DEFAULT_OUTPUT = WRAPPER_DIR / "EchokrauTTS.zip"

# Fixed timestamp for every entry so two builds of the same sources produce a
# byte-identical archive. Without it the zip differs on every run and "did the
# package actually change?" cannot be answered by comparing files. The value is
# arbitrary but must stay >= 1980 (the zip epoch).
FIXED_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


def collect_files(wrapper_dir: Path = WRAPPER_DIR) -> list[str]:
    """Wrapper-relative paths to ship, sorted.

    Sorted because zip entry order is otherwise git's, which varies between
    versions and platforms — and an archive whose bytes depend on the git
    build is not reproducible.
    """
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", str(wrapper_dir)],
        cwd=wrapper_dir.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    prefix = wrapper_dir.name + "/"
    names = set()
    for line in result.stdout.splitlines():
        line = line.strip().replace("\\", "/")
        if not line.startswith(prefix):
            continue
        rel = line[len(prefix):]
        if not (wrapper_dir / rel).is_file():  # deleted but still in the index
            continue
        names.add(rel)
    return sorted(names)


def build(output: Path = DEFAULT_OUTPUT, wrapper_dir: Path = WRAPPER_DIR) -> Path:
    """Write the archive and return its path."""
    files = collect_files(wrapper_dir)
    if not files:
        raise SystemExit("build-release-zip: nothing to package — is this a git checkout?")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary name and move it into place, so an interrupted build
    # cannot leave a half-written archive that looks like a finished one — the
    # same reason the voice-pack download uses a .part file.
    tmp = output.with_suffix(output.suffix + ".part")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for rel in files:
                info = zipfile.ZipInfo(rel, date_time=FIXED_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                zf.writestr(info, (wrapper_dir / rel).read_bytes())
        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUTPUT,
        help="archive to write (default: wrapper/EchokrauTTS.zip)",
    )
    parser.add_argument(
        "--list", action="store_true", help="print what would be packaged, write nothing",
    )
    args = parser.parse_args(argv)

    if args.list:
        for rel in collect_files():
            print(rel)
        return 0

    output = build(args.output)
    files = collect_files()
    size_kb = output.stat().st_size / 1024
    print(f"{output}: {len(files)} files, {size_kb:.0f} KB")
    print("Next: create the GitHub release by hand and upload this file as EchokrauTTS.zip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
