#!/usr/bin/env bash
# ============================================================================
#  Echokrautts - One-Click-Starter (Linux / macOS) - XTTS-v2-Backend
#  Startet den Wrapper mit dem XTTS-v2-Backend: holt uv -> installiert beim
#  ersten Mal alles (Python, GPU-Erkennung, Abhaengigkeiten, Modelle) ->
#  serviert. Fuer F5-TTS stattdessen start-f5tts.sh verwenden.
#  XTTS ist mehrsprachig (Sprache pro Request); optionale Argumente
#  (z.B. --language en) werden durchgereicht.
# ============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/wrapper/bootstrap/install_linux.sh" --start --tts-backend xtts "$@"
