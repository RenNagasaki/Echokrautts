#!/usr/bin/env bash
# ============================================================================
#  Echokrautts - One-Click-Starter (Linux / macOS) - Qwen-0.6b-Backend (Qwen3-TTS)
#  Startet den Wrapper mit dem Qwen-0.6b-Backend (Qwen3-TTS): holt uv -> installiert beim
#  ersten Mal alles (Python, GPU-Erkennung, Abhaengigkeiten, Modelle) ->
#  serviert. Braucht eine NVIDIA-Grafikkarte (AMD/ROCm experimentell).
#  Qwen ist mehrsprachig (Sprache pro Request); optionale Argumente
#  (z.B. --language en) werden durchgereicht.
# ============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/wrapper/bootstrap/install_linux.sh" --start --tts-backend qwen-0.6b "$@"
