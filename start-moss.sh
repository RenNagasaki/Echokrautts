#!/usr/bin/env bash
# ============================================================================
#  Echokrautts - One-Click-Starter (Linux / macOS) - MOSS-TTS-Nano-Backend
#  Startet den Wrapper mit dem MOSS-TTS-Nano-Backend: holt uv -> installiert beim
#  ersten Mal alles (Python, GPU-Erkennung, Abhaengigkeiten, Modelle) ->
#  serviert. Fuer die anderen Engines start-f5tts.sh bzw. start-xtts.sh.
#
#  MOSS ist das kleinste der drei Modelle (312 MB) und laeuft auf der CPU
#  praktisch genauso schnell wie auf der GPU. Es startet deshalb ABSICHTLICH auf
#  der CPU und laesst die Grafikkarte dem Spiel - dafuer ist kein Argument
#  noetig. Wer die GPU trotzdem will:  ./start-moss.sh --moss-device cuda
#  Optionale Argumente (z.B. --language en) werden durchgereicht.
# ============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/wrapper/bootstrap/install_linux.sh" --start --tts-backend moss "$@"
