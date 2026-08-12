# syntax=docker/dockerfile:1
# ============================================================================
#  Echokrautts — container image
#
#  ONE Dockerfile, two variants, selected purely by build args:
#    CUDA : --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
#    CPU  : --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
#  There is no nvidia/cuda base image on purpose: the cu128 torch wheels ship
#  their own CUDA runtime (nvidia-*-cu12 packages), exactly like the native
#  bootstrap install, and the driver comes from the host via
#  nvidia-container-toolkit. Saves ~2 GB and keeps both variants identical
#  apart from one index URL.
#
#  Model weights are NOT baked in. Two of the three engines ship non-commercial
#  weights (F5 finetunes CC-BY-NC-4.0, XTTS-v2 CPML) — redistributing them
#  inside a public image would be a licensing problem, and Chatterbox's are
#  simply large. The entrypoint downloads the ACTIVE backend's weights on first
#  start into the /data/models volume instead.
# ============================================================================

ARG PYTHON_VERSION=3.11

# ------------------------------------------------------------------ builder
FROM python:${PYTHON_VERSION}-slim AS builder

# Mirrors bootstrap.step_deps. Keep the defaults in sync with wrapper/config.json
# (torch_version / torchaudio_version / transformers_constraint).
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.7.0
ARG TORCHAUDIO_VERSION=2.7.0
ARG TRANSFORMERS_CONSTRAINT=transformers>=4.57,<5
# Chatterbox is installed --no-deps + a curated list, because its own metadata
# pins torch==2.6.0, transformers==5.2.0 and gradio — see config.chatterbox_install
# (keep both in sync). setuptools<81 is NOT in this list: it carries a `<`, which
# an unquoted shell expansion below would read as a redirect, so it is installed
# as its own quoted argument.
ARG CHATTERBOX_PACKAGE=chatterbox-tts==0.1.7
ARG CHATTERBOX_DEPS="s3tokenizer diffusers==0.29.0 conformer==0.3.2 resemble-perth pyloudnorm pykakasi omegaconf einops"

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential/git only exist in this stage — the runtime stage just copies
# the finished venv, so no toolchain ends up in the shipped image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY wrapper/pyproject.toml /src/wrapper/pyproject.toml
COPY wrapper/src /src/wrapper/src

# Order is load-bearing and mirrors bootstrap.step_deps:
#   1. pinned torch from the backend-specific index
#   2. BOTH engines in a SINGLE resolution (f5-tts via the wrapper project +
#      coqui-tts) with the transformers pin in the same resolve, so pip finds
#      one mutually compatible set instead of two installs stomping each other
#   3. Chatterbox with --no-deps + its curated deps (its pins are unresolvable
#      against 2. — torch 2.6, transformers 5.2, gradio), carrying the
#      transformers pin so a dep cannot drag it to 5.x
#   4. re-pin torch — the engine deps can win the resolution and drag in a
#      newer torch that requires torchcodec (→ system FFmpeg)
#   5. drop the unused torchcodec (f5-tts declares it but only calls
#      torchaudio.load, which uses the bundled soundfile backend on 2.7.x)
RUN pip install "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
        --index-url "${TORCH_INDEX_URL}" \
 && pip install /src/wrapper coqui-tts "${TRANSFORMERS_CONSTRAINT}" \
 && pip install --no-deps "${CHATTERBOX_PACKAGE}" \
 && pip install ${CHATTERBOX_DEPS} "setuptools<81" "${TRANSFORMERS_CONSTRAINT}" \
 && pip install "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
        --index-url "${TORCH_INDEX_URL}" \
 && pip uninstall -y torchcodec || true

# Same guards as bootstrap._verify_torch / _verify_transformers: fail the BUILD
# rather than ship an image that only breaks on the first inference request.
RUN python - <<'PY'
import importlib.util as u, os, sys, torch
want = os.environ.get("TORCH_VERSION", "2.7.0")
base = torch.__version__.split("+")[0]
if base != want:
    sys.exit(f"torch verification failed: installed {torch.__version__}, expected {want}")
if u.find_spec("torchcodec") is not None:
    sys.exit("torch verification failed: torchcodec is present (would require system FFmpeg)")
from transformers.pytorch_utils import isin_mps_friendly  # noqa: F401 — XTTS needs it (gone in transformers 5.x)
import perth
from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: F401 — must import on our transformers pin
if perth.PerthImplicitWatermarker is None:
    # perth swallows the ImportError and degrades the class to None; the model
    # constructor then dies with "'NoneType' object is not callable". Usually
    # means pkg_resources is gone, i.e. setuptools >= 81.
    sys.exit("chatterbox verification failed: perth watermarker unavailable")
print(f"ok: torch {torch.__version__}, no torchcodec, transformers pin holds, chatterbox loads")
PY

# ------------------------------------------------------------------ runtime
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG ECHOKRAUTTS_VERSION=dev
ARG VARIANT=cuda
# Hardware backend baked into the image, because the runtime probes cannot see
# the truth from inside a slim container:
#   cuda → "auto"  (nvidia-smi is injected by the container toolkit, and auto
#                   degrades to CPU gracefully when no GPU was passed)
#   rocm → "rocm"  (no rocminfo, no /opt/rocm here → auto would answer CPU and
#                   the GPU would sit idle)
#   cpu  → "cpu"   (closes the --gpus-on-a-CPU-image trap: the injected
#                   nvidia-smi would otherwise select a CUDA device this torch
#                   build cannot serve)
ARG GPU_BACKEND=auto

LABEL org.opencontainers.image.title="Echokrautts" \
      org.opencontainers.image.description="Streaming voice-cloning TTS wrapper (F5-TTS + XTTS-v2 + Chatterbox Multilingual) with an HTTP API" \
      org.opencontainers.image.source="https://github.com/RenNagasaki/Echokrautts" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${ECHOKRAUTTS_VERSION}"

# libsndfile1: soundfile backend for torchaudio.load (no FFmpeg needed).
# curl: HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app/wrapper
COPY wrapper/src ./src
COPY wrapper/config.json ./config.json
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Volumes. Samples and models are the only mutable state; everything else in
# the image is read-only. Both are absolute paths, which config.py honours.
ENV F5W_SAMPLES_DIR=/data/samples \
    F5W_MODELS_DIR=/data/models \
    F5W_GPU_BACKEND=${GPU_BACKEND} \
    F5W_HOST=0.0.0.0 \
    F5W_PORT=8765 \
    F5W_TTS_BACKEND=xtts \
    F5W_LANGUAGE=de \
    TTS_HOME=/data/models \
    COQUI_TOS_AGREED=1 \
    ECHOKRAUTTS_VERSION=${ECHOKRAUTTS_VERSION} \
    ECHOKRAUTTS_VARIANT=${VARIANT} \
    PYTHONUNBUFFERED=1

RUN mkdir -p /data/samples /data/models
VOLUME ["/data/samples", "/data/models"]

EXPOSE 8765

# Weights download on first start (several GB) — give it room before the
# container is called unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45m --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${F5W_PORT}/health" || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["serve"]
