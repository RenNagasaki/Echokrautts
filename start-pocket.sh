#!/usr/bin/env bash
# ============================================================================
#  Echokrautts - One-Click-Starter (Linux / macOS) - Pocket-TTS-Backend (Kyutai, CPU)
#  Startet den Wrapper mit dem Pocket-TTS-Backend (Kyutai, CPU): holt uv -> installiert beim
#  ersten Mal alles (Python, GPU-Erkennung, Abhaengigkeiten, Modelle) ->
#  serviert. Laeuft auf der CPU; ein Modell je Sprache (kein Japanisch).
#  Sprache per --language (de/en/fr/it/es/pt); optionale Argumente
#  (z.B. --language en) werden durchgereicht.
# ============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/wrapper/bootstrap/install_linux.sh" --start --tts-backend pocket "$@"
