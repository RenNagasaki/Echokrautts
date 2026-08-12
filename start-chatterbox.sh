#!/usr/bin/env bash
# ============================================================================
#  Echokrautts - One-Click-Starter (Linux / macOS) - Chatterbox Multilingual
#  Startet den Wrapper mit Resembles Chatterbox Multilingual: holt uv ->
#  installiert beim ersten Mal alles (Python, GPU-Erkennung, Abhaengigkeiten,
#  Modelle) -> serviert. Fuer F5-TTS bzw. XTTS stattdessen start-f5tts.sh bzw.
#  start-xtts.sh verwenden.
#  Chatterbox ist mehrsprachig (23 Sprachen, Sprache pro Request), klont ohne
#  Transkript, kennt aber KEIN Streaming und keinen speed-Parameter.
#  Optionale Argumente (z.B. --language en) werden durchgereicht.
# ============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/wrapper/bootstrap/install_linux.sh" --start --tts-backend chatterbox "$@"
