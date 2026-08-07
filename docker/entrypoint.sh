#!/usr/bin/env bash
# ============================================================================
#  Echokrautts container entrypoint.
#
#  Deliberately does NOT run wrapper/bootstrap/bootstrap.py: steps 1-4 (uv,
#  Python, GPU detection, dependency install) already happened at image build
#  time, and re-running them would rebuild a venv inside the container on every
#  start. What is left of the bootstrap is exactly two things — download the
#  weights of the ACTIVE backend into the models volume, then serve — and that
#  is what this script does.
# ============================================================================
set -euo pipefail

BACKEND="${F5W_TTS_BACKEND:-f5}"
HOST="${F5W_HOST:-0.0.0.0}"
PORT="${F5W_PORT:-8765}"
LOG_LEVEL="${UVICORN_LOG_LEVEL:-warning}"

log() { printf '[entrypoint] %s\n' "$*"; }

download_models() {
    if [ "${ECHOKRAUTTS_SKIP_DOWNLOAD:-0}" = "1" ]; then
        log "model download skipped (ECHOKRAUTTS_SKIP_DOWNLOAD=1)"
        return
    fi
    # Only the active backend's weights: F5 pulls all four language finetunes
    # (~5 GB), XTTS one multilingual model (~2 GB). Both are idempotent — the
    # HF cache and Coqui's ModelManager skip files that are already in the
    # volume, so a restart costs a few seconds, not a re-download.
    case "$BACKEND" in
        xtts) log "ensuring XTTS-v2 weights in ${F5W_MODELS_DIR:-models} …"
              python -m src.xtts_backend ;;
        *)    log "ensuring F5-TTS weights in ${F5W_MODELS_DIR:-models} …"
              python -m src.models ;;
    esac
}

case "${1:-serve}" in
    serve)
        download_models
        log "starting server on ${HOST}:${PORT} (backend=${BACKEND}, version=${ECHOKRAUTTS_VERSION:-dev}, variant=${ECHOKRAUTTS_VARIANT:-?})"
        exec python -m uvicorn src.server:create_app --factory \
            --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
        ;;
    download)
        # `docker compose run --rm echokrautts download` — pre-fill the volume
        # without starting the server.
        download_models
        ;;
    *)
        # Anything else is run verbatim (bash, python -c …, pytest) with no
        # download side effect, so the image stays debuggable.
        exec "$@"
        ;;
esac
