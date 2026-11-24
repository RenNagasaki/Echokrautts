#!/usr/bin/env bash
# ============================================================================
#  Echokrautts - One-Click-Starter (Linux / macOS) - F5-TTS-Backend
#  Startet den Wrapper mit dem F5-TTS-Backend: holt uv -> installiert beim
#  ersten Mal alles (Python, GPU-Erkennung, Abhaengigkeiten, Modelle) ->
#  serviert. Fuer XTTS stattdessen start-xtts.sh verwenden.
#  Optionale Argumente (z.B. --language en) werden durchgereicht.
# ============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/wrapper/bootstrap/install_linux.sh" --start --tts-backend f5 "$@"
