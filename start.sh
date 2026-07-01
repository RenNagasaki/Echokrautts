#!/usr/bin/env bash
# ============================================================================
#  Echokrautts - One-Click-Starter (Linux / macOS)
#  Startet den F5-TTS-Wrapper: holt uv -> installiert beim ersten Mal alles
#  (Python, GPU-Erkennung, Abhaengigkeiten, Modelle) -> serviert.
#  Optionale Argumente (z.B. --language en) werden durchgereicht.
# ============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/wrapper/bootstrap/install_linux.sh" --start "$@"
