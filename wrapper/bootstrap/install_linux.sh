#!/usr/bin/env bash
# Optional Linux starter for the F5-TTS wrapper (SPEC §3).
# Used only when no Python is available to run bootstrap.py directly: it fetches
# uv, then runs bootstrap.py via uv's managed Python.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
UV_DIR="$ROOT/.uv"
UV="$UV_DIR/uv"

if [ ! -x "$UV" ]; then
    if command -v uv >/dev/null 2>&1; then
        mkdir -p "$UV_DIR"
        cp "$(command -v uv)" "$UV"
    else
        ASSET="uv-x86_64-unknown-linux-gnu.tar.gz"
        URL="https://github.com/astral-sh/uv/releases/latest/download/$ASSET"
        TMP="$(mktemp -d)"
        curl -fL "$URL" -o "$TMP/$ASSET"
        mkdir -p "$UV_DIR"
        tar -xzf "$TMP/$ASSET" -C "$TMP"
        find "$TMP" -name uv -type f -exec cp {} "$UV" \;
        chmod +x "$UV"
        rm -rf "$TMP"
    fi
fi

exec "$UV" run --python 3.11 python "$SCRIPT_DIR/bootstrap.py" "$@"
