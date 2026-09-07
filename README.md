# Echokrautts

A lightweight, plug-and-play Python wrapper around two **voice-cloning TTS** engines —
[F5-TTS](https://github.com/SWivid/F5-TTS) and [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) —
that a host application (e.g. a C#/Dalamud plugin) starts as a **separate process** and drives
over stdout (NDJSON events) and HTTP (streaming PCM). It provides zero-shot voice cloning with
sentence-level streaming, a VRAM-aware worker pool, and self-contained installation via
[`uv`](https://github.com/astral-sh/uv) — no system Python or git required.

The wrapper lives in [`wrapper/`](wrapper/), and the licenses of everything it installs are in
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

## Quick start

The simplest entry point is the one-click launcher in the repo root — it fetches `uv`, installs
everything on first run, then serves. There is one launcher pair per TTS backend:

```bash
start-f5tts.bat        # Windows, F5-TTS backend (visible window, pauses at the end)
./start-f5tts.sh       # Linux/macOS, F5-TTS backend
start-xtts.bat         # Windows, XTTS-v2 backend
./start-xtts.sh        # Linux/macOS, XTTS-v2 backend
```

Both engines and all their weights are installed either way; the launcher only picks (via
`--tts-backend f5` / `xtts`) which engine the worker pool loads at startup, so
switching is a restart, not a reinstall. All launchers forward extra arguments, e.g.
`start-xtts.bat --language en`.

Under the hood these call `wrapper/bootstrap/install_win.ps1` / `install_linux.sh` (which fetch `uv`)
with `--start --tts-backend <…>`. A host process that already has a Python can run the bootstrap directly:

```bash
python wrapper/bootstrap/bootstrap.py --start --parent-pid <host_pid>
```

The bootstrap runs a fixed 6-step sequence (obtain `uv` → pin Python → detect GPU → install deps →
preload models → serve) and reports each step as an NDJSON `progress` event on stdout. When the
server is listening it emits a `ready` event with host/port/backend. **All TTS engines and all of
their model weights are installed once**, so switching engine or language is only a restart (see
[TTS backends](#tts-backends)).

### Why torch is pinned to 2.7.x

`torch`/`torchaudio` are pinned (configurable via `torch_version`/`torchaudio_version`). Newer
torchaudio routes `torchaudio.load` through **torchcodec**, which requires system **FFmpeg** shared
libraries — an external, non-self-contained dependency. On 2.7.x, `torchaudio.load` still uses the
bundled-libsndfile **soundfile** backend, so the wrapper installs and runs with **no external
binaries**. The TTS engines declare `torchcodec` as a dependency but never import it (they only call
`torchaudio.load`), so the bootstrap re-pins torch after the deps install and drops the unused
torchcodec. Bump the pins deliberately and re-verify the soundfile path if you change them. The
container image applies the same pin and the same re-pin/uninstall, verified at build time.

## Docker

The container is the same wrapper, minus the bootstrap: `uv`, the Python pin, GPU detection and the
dependency install all happened at **build** time. What is left at runtime is downloading the model
weights into the volume and serving. **Both engines are in every image** — which one runs is one
environment variable, so switching engine or language is a restart, never a reinstall.

### Images

| Tag | Variant | Requires on the host |
| --- | --- | --- |
| `ghcr.io/rennagasaki/echokrautts:latest` · `:<version>` | CUDA (cu128) | NVIDIA driver + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| `ghcr.io/rennagasaki/echokrautts:latest-rocm` · `:<version>-rocm` | AMD / ROCm 6.4 | amdgpu kernel driver; on Windows: WSL2 + Docker Desktop |
| `ghcr.io/rennagasaki/echokrautts:latest-cpu` · `:<version>-cpu` | CPU | nothing — but synthesis is far slower than real time |
| `:main` · `:main-cpu` · `:main-rocm` · `:sha-<commit>` | any | untagged builds from a branch, published on demand |

There is no `nvidia/cuda` base image: the cu128 torch wheels bring their own CUDA runtime, exactly
like the bare-metal install, and the driver is injected by the container toolkit. The same holds for
ROCm — the rocm wheels carry the ROCm userspace, the host only provides the kernel driver.

**GPU coverage.** The CUDA image needs no per-card tag: `torch 2.7.0+cu128` is compiled for
`sm_50 … sm_120`, i.e. Maxwell (GTX 900) through Blackwell (RTX 50xx). The limiting factor is the
host driver, not the card — CUDA 12.x wants ≥ 525 (Linux) / the R525 branch (Windows), and newer
cards need a newer driver anyway. The ROCm image targets RDNA3/RDNA4 (RX 7900 XTX/XT, RX 9070/XT and
the matching Pro/Instinct parts); older AMD cards and every AMD GPU under native Windows still land
on CPU (see [AMD GPUs](#amd-gpus)).

### Getting it running

```bash
curl -O https://raw.githubusercontent.com/RenNagasaki/Echokrautts/main/docker-compose.yml
mkdir -p samples models          # bind-mount targets, see below
docker compose up -d             # CPU host: -f docker-compose.cpu.yml
docker compose logs -f           # first start downloads the weights (2-5 GB)
curl localhost:8765/health
```

Pin a version with `ECHOKRAUTTS_TAG=0.0.0.6 docker compose up -d`.

### What you have to set

**Volumes — two, and both matter.** Everything else in the image is read-only.

| Container path | What belongs there | If you skip it |
| --- | --- | --- |
| `/data/samples` | Your voice samples. Either `<name>.wav` (also `.flac`/`.mp3`) or a folder `<name>/` holding several clips of the same voice — one is picked at random per request. The request only uses the **stem**, so `X`, `X.wav` and `X.mp3` all resolve to the same voice. Empty on first start → the container fetches the [Echokraut voice pack](#voice-samples) into it. | The voice pack is re-downloaded on every `docker run` and lost with the container. |
| `/data/models` | Model weights plus the HuggingFace and Coqui caches. Deliberately **not** baked into the image: both engines ship non-commercially licensed weights (F5 finetunes CC-BY-NC-4.0, XTTS-v2 CPML), so the container fetches the active backend's weights on first start. | Works, but every `docker run` re-downloads several GB into the container's throwaway layer. |

**Port.** The server listens on `8765` inside the container (`EXPOSE 8765`); publish it with
`-p 8765:8765` or the compose `ports:` entry. Change the *inside* port with `F5W_PORT` only if you
have a reason to — publishing a different host port is the usual way.

**Environment.** Every `F5W_*` variable is a field of `wrapper/config.json`, upper-cased — there is
no separate Docker configuration schema. The ones that actually matter in a container:

| Variable | Image default | Meaning |
| --- | --- | --- |
| `F5W_TTS_BACKEND` | `xtts` | `xtts` = XTTS-v2, clones from the sample alone, multilingual per request. `f5` = F5-TTS, loads **one** language finetune per process and needs a transcript (or ASR) for the reference clip. |
| `F5W_LANGUAGE` | `de` | F5: which finetune is loaded at startup. XTTS: the fallback when a request omits `language`. |
| `F5W_XTTS_FP16` | unset (`false`) | XTTS half precision, ~1.4× faster. CUDA only — silently ignored on CPU. |
| `F5W_API_KEY` | unset | Requires `Authorization: Bearer <key>` on every endpoint. **Set it if the port is reachable from anywhere but the host** — the server binds `0.0.0.0`. |
| `F5W_MAX_WORKERS` | `1` | Worker-pool ceiling. Each worker is a full model copy on the device, and one request is served by exactly one worker — more workers buy **concurrency, not speed**. Raise it only if several requests really arrive at once; free VRAM (`(free − reserve) ÷ per_job`) may still cap the count lower. `null` = derive it from VRAM, at most 4. |
| `F5W_VOICEPACK_AUTO_DOWNLOAD` | `true` | Fetch the Echokraut voice pack when the samples volume is empty. Set `false` if you only ever use your own voices. |
| `F5W_RATE_LIMIT_PER_HOUR` · `F5W_RATE_LIMIT_PER_IP_PER_HOUR` | `0` (off) | Sliding-window request limits on `/tts` → 429 + `Retry-After`. Worth setting whenever the port is reachable beyond the host; see [Rate limits](#rate-limits). |
| `F5W_TRUST_FORWARDED_FOR` | `false` | Take the caller address from `X-Forwarded-For`. Only behind a proxy you control — a container behind one otherwise counts every caller as the proxy. |
| `F5W_GPU_BACKEND` | `auto` (cuda image) · `cpu` · `rocm` | Forces the hardware backend instead of probing. Each image ships the right value; override only to deliberately fall back (`cpu`). Values: `auto`, `cuda`, `rocm`, `dml`, `xpu`, `cpu`. |
| `F5W_SAMPLES_DIR` / `F5W_MODELS_DIR` | `/data/samples` · `/data/models` | Only change these if you mount somewhere else — the defaults match the volumes above. |
| `F5W_PORT` / `F5W_HOST` | `8765` · `0.0.0.0` | Bind address inside the container. |
| `F5W_STREAM_CHUNK_SIZE` | `20` | XTTS token-streaming granularity; lower = earlier first audio, slightly more overhead. F5 ignores it. |
| `F5W_MAX_CHARS_PER_CHUNK` | `250` | Sentence-chunking limit for long texts. |

Two container-only variables exist next to those: `ECHOKRAUTTS_SKIP_DOWNLOAD=1` skips the weight
download at start, and `UVICORN_LOG_LEVEL` (default `warning`) sets uvicorn's own verbosity — the
wrapper's NDJSON log on stdout is unaffected by either.

Type coercion follows the config schema: bools accept `true/false/1/0/yes/no/on/off`, `F5W_API_KEY=""`
means unset, `F5W_ALLOWED_SAMPLE_EXT` is comma-separated, `F5W_LANGUAGES` is JSON.
`F5W_PARENT_PID` is for the desktop host's watchdog and must stay unset in a container — otherwise
the server shuts itself down as soon as that PID is not alive.

**Each image forces its own backend**, so detection cannot pick something the installed torch build
cannot serve. That closes two traps that would otherwise be silent: `--gpus all` on the CPU image
(the injected `nvidia-smi` would make the wrapper choose a CUDA device it has no wheel for), and the
ROCm image (a slim container has no `rocminfo`, so probing would answer "CPU" and the GPU would sit
idle). If you override `F5W_GPU_BACKEND=auto` yourself, both traps come back.

**`/health` is the source of truth**: it reports the backend, language and effective fp16 the
process actually came up with, not what you *meant* to set.

```bash
docker compose up -d                                       # xtts, per compose file
F5W_TTS_BACKEND=f5 F5W_LANGUAGE=de docker compose up -d    # F5, German
curl -s localhost:8765/health
```

### Without compose

```bash
# --gpus all belongs to the CUDA image only; drop it entirely for :latest-cpu.
docker run -d --name echokrautts \
  --gpus all \
  -p 8765:8765 \
  -v "$PWD/samples:/data/samples" \
  -v "$PWD/models:/data/models" \
  -e F5W_TTS_BACKEND=xtts \
  -e F5W_LANGUAGE=de \
  -e F5W_XTTS_FP16=true \
  ghcr.io/rennagasaki/echokrautts:latest
```

Then synthesize (raw PCM, s16 mono 24 kHz — bytes ÷ 2 ÷ 24000 = seconds):

```bash
curl -X POST localhost:8765/tts -H 'Content-Type: application/json' \
  -d '{"sample":"my_voice","text":"Hallo Welt."}' --output out.pcm
```

### Operating it

```bash
docker compose run --rm echokrautts download   # pre-fill the model volume, don't serve
docker compose run --rm echokrautts bash       # shell in the image, no download
docker compose pull && docker compose up -d    # update — the volumes survive
```

The entrypoint only downloads the **active** backend's weights (F5 pulls all four language
finetunes, XTTS one multilingual model). Switch backend later and the next start fetches what is
missing; nothing is re-downloaded twice. `ECHOKRAUTTS_SKIP_DOWNLOAD=1` suppresses the step entirely,
e.g. when the volume was filled by hand.

A **custom model** works the same as on bare metal: drop it into `models/echokraut_custom/` on the
host (an F5 checkpoint, or an XTTS directory with `config.json` + `model.pth`) and it overrides the
configured model of the active engine — auto-detected, no variable to set.

The `HEALTHCHECK` has a 45-minute start period because of that first download; a container reported
as `starting` for a long time is normal on first run and `docker compose logs -f` shows the progress.

### Building the image yourself

Two GitHub workflows, both publishing to GHCR:

- **Docker build (manual)** — Actions → Run workflow. Pick a branch, `variant` (cuda/cpu/both) and a
  `tag` (default `main`). Pushes `<tag>`, `<tag>-cpu` and `sha-<commit>`; never touches `latest`.
- **Docker release** — runs when a GitHub release is published, pushing `<version>` and `latest`
  (plus the `-cpu` pair). Pre-releases skip the `latest` tags.

Locally, the variant is three build args — CUDA is the default:

```bash
docker build -t echokrautts:dev .                                        # CUDA
docker build -t echokrautts:dev-cpu  --build-arg GPU_BACKEND=cpu \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu .
docker build -t echokrautts:dev-rocm --build-arg GPU_BACKEND=rocm \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm6.4 \
  --build-arg TORCH_VERSION=2.8.0 --build-arg TORCHAUDIO_VERSION=2.8.0 .
```

## Hardware / GPU acceleration

The bootstrap **detects your hardware** (the "detect GPU" step) and installs the matching torch wheel
+ device automatically — no configuration needed. What you actually get:

| Hardware | Windows | Linux |
|----------|---------|-------|
| **NVIDIA** | ✅ CUDA (cu128 for Blackwell/sm≥12, else cu126) | ✅ CUDA |
| **AMD** | ✅ ROCm for Radeon 9000 / select 7000 (AMD's cp312 wheels, installed automatically); other cards ⚠️ DirectML → **CPU**. See [AMD GPUs](#amd-gpus) | ✅ ROCm (reported as a CUDA device) |
| **Intel dGPU** | ⚠️ XPU detected, but XTTS maps it to CPU and F5 self-tests → typically **CPU** | ⚠️ same |
| **No GPU / other** | 🐢 CPU | 🐢 CPU |

- **Every backend runs on any of these** — the question is only *how fast*. On CPU (incl. AMD-on-Windows
  and Intel) expect real-time factor **> 1** (slower than real time): fine for testing, too slow for
  live in-game TTS.
- For fragile devices (dml/xpu) the engine runs a tiny **self-test** at startup and rebuilds the worker
  pool on CPU if it fails, so you always get working audio, just not always on the GPU.
- The **`xtts_fp16`** speedup applies only where the device is CUDA (NVIDIA or ROCm); see
  [TTS backends](#tts-backends).
- **Workers are concurrency, not speed.** A worker is one model instance loaded on the device, and a
  request occupies exactly one for its whole duration — so a second worker never makes a single line
  faster, it lets a *second* line be synthesised at the same time. Each one costs a full copy of the
  weights in VRAM, which is why the default is **`max_workers: 1`**: raise it when several requests
  genuinely arrive at once, and free VRAM still caps the real count below your number
  (`(free − vram_reserve_gb) ÷ per_job_gb`; `null` = derive it automatically, at most 4). When every
  worker is busy the request queues, and only past `max_queue` does it become a 503.
- Detection can be overruled with **`gpu_backend`** (`auto` · `cuda` · `rocm` · `rocm_win` · `dml` ·
  `xpu` · `cpu`).
  Use it where probing cannot see the truth — inside containers, or to force CPU deliberately. An
  unknown value is a hard error, never a silent CPU fallback.

### AMD GPUs

**Linux, and Windows via WSL2/Docker:** use the **`-rocm` image**. It ships `torch 2.8.0+rocm6.4` and
forces `gpu_backend=rocm`, and it covers RDNA3/RDNA4 — RX 7900 XTX/XT, RX 9070/XT and the matching
Pro/Instinct parts. On a Linux host it needs only the amdgpu kernel driver; on Windows it runs inside
WSL2 with Docker Desktop, which AMD supports for exactly these cards with a recent Adrenalin driver.

```bash
docker compose -f docker-compose.rocm.yml up -d
```

The container is handed `/dev/kfd` and `/dev/dri` instead of `--gpus` — that is how AMD GPUs reach a
container. Cards whose gfx version ROCm does not accept can often be spoofed to the nearest supported
one with `HSA_OVERRIDE_GFX_VERSION` (commented out in the compose file).

**Native Windows, no container:** supported through AMD's own ROCm build, automatically. The
bootstrap reads the display-adapter name and, if it matches AMD's supported hardware (Radeon 9000
series and select 7000 — the pattern lives in `rocm_windows.gpu_pattern`), installs the ROCm stack
instead of the DirectML fallback. Nothing to configure; `/health` reports what was chosen.

Three things about that path are unlike every other backend, all of them AMD's doing:

- **Python 3.12.** The wheels are cp312-only, so this one venv is built on 3.12 while the rest of the
  wrapper runs on the configured 3.11. `uv` fetches the interpreter itself.
- **Wheels by URL, not an index.** AMD publishes no pip index for Windows, so the ROCm runtime
  (`rocm_sdk_*`) and torch/torchaudio/torchvision are installed from individual URLs, listed under
  `rocm_windows` in `config.json`. **Bumping to a newer ROCm release is an edit there, not a code
  change.**
- **torch 2.9.1**, whose torchaudio turns `load` into a torchcodec alias that would need system
  FFmpeg. The wrapper patches `torchaudio.load` back to its bundled soundfile decoder at startup
  (`src/audio_compat.py`), so the install stays free of external binaries like every other backend.

Because the card is matched **by name**, the engine treats this backend as fragile: if the worker
fails to build or fails its self-test, it rebuilds on CPU and says so in the log rather than
refusing to start. Force the decision yourself with `gpu_backend` (`rocm_win` or `cpu`).

Not chosen, for the record: **DirectML**. Microsoft has it in maintenance mode — security fixes
only, Windows ML is the successor — and `torch-directml` has not been released since September 2024.
It remains the fallback for AMD cards outside AMD's ROCm list, where it self-tests and lands on CPU.

## Web UI

Open **`http://localhost:8765/`** in a browser: pick a voice, pick a language, type text, hit
Generate, listen. It is a single self-contained HTML file the wrapper serves itself — no build step,
no CDN, works offline and inside the container.

- **Voices** come from `/samples`; if the folder is empty the page says so instead of failing.
- **Language** is driven by `/languages`, which applies the backend rule: with **XTTS** you can pick
  any supported language per request, with **F5** the selector is pre-filled with the loaded model's
  language and **disabled** — that backend serves one language per process.
- **Audio**: the page asks `/tts` for `format: "wav"`, because browsers cannot play the raw PCM the
  plugin streams. It also prints how much audio was produced and the real-time factor.
- **API key**: if `api_key` is set, the page itself still loads (you have to be able to type the key
  somewhere) and everything it calls stays protected. The key is remembered in `localStorage`.

## HTTP API

| Method & path        | Purpose                                                            |
|----------------------|--------------------------------------------------------------------|
| `POST /tts`          | Streaming synthesis. Body = raw PCM (`pcm_s16le`, mono). Metadata in `X-Job-Id` / `X-Sample-Rate` / `X-Channels` / `X-Sample-Format` headers. Add `"format": "wav"` to get a buffered `audio/wav` file instead (what the web UI uses; gives up streaming). |
| `GET /samples`       | Usable voice names (`?details=true` adds `has_ref_text`/`bytes`/`count`). |
| `GET /languages`     | What may go in a request's `language`: `{active, options, locked, reason}`. `locked` is true on F5 (one model per process). |
| `GET /`              | The built-in [web UI](#web-ui). The only endpoint never behind the API key. |
| `POST /cancel/{id}`  | Cancel a running job.                                              |
| `GET /jobs/{id}`     | Live progress (`sentences_done`/`sentences_total`/`percent`).      |
| `GET /health`        | Backend/device/worker/queue status, plus `rate_limit` usage.       |
| `POST /shutdown`     | Graceful shutdown.                                                 |

Configuration lives in `wrapper/config.json` (overridable by `F5W_*` env vars and `--kebab-case` CLI
flags; precedence JSON < ENV < CLI). The server **binds `0.0.0.0` by default** so the host reaches it
with no config edit; this also exposes it on your LAN, so set an `api_key` (then all requests need
`Authorization: Bearer <key>`) — or narrow `host` back to `127.0.0.1` — if that isn't what you want.

### Rate limits

Off by default — a wrapper serving one game client has no reason to ration itself. Turn them on when
the port is reachable by more than you:

| Setting | Meaning |
| --- | --- |
| `rate_limit_per_hour` | Ceiling across **all** callers. |
| `rate_limit_per_ip_per_hour` | Ceiling **per caller address**. Both may be set; the stricter one answers first. |
| `trust_forwarded_for` | Read `X-Forwarded-For` instead of the socket address. **Only enable behind a proxy you control** — otherwise a caller invents an address per request and the per-IP limit means nothing. |

Both use a **sliding** window, not hourly buckets: with buckets you could spend a full quota at 10:59
and another at 11:01. Only requests that were actually accepted count, so a client that keeps
retrying is not extending its own lockout. Exceeding a limit gives **429** with a `Retry-After`
header pointing at the moment the next slot frees up.

This is not back-pressure. `max_queue` already answers **503** when the engine is saturated
("busy right now"); a 429 says "you have had your share this hour". Both stay in place, and the rate
limit is checked first so a rejected caller never occupies queue space. Only `/tts` is limited —
`/samples`, `/languages`, `/health` and the web UI keep working, or the page would lock itself out.
`GET /health` reports the current usage.

## Voice samples

**You start with voices.** On first start — an empty `samples` folder, which is also every fresh
container volume — the wrapper downloads the current **Echokraut voice pack** and unpacks it there
(~107 MB, 260 voices with reference transcripts). It is skipped as soon as the folder holds any
audio, and it never blocks startup: no network just means no voices yet, which you fix by dropping in
a file. Turn it off with `voicepack_auto_download: false` (`F5W_VOICEPACK_AUTO_DOWNLOAD=false`).

The pack is found by **release tag prefix** (`voicepack_tag_prefix`, default `EK-VoicePack-`) in
`voicepack_repo`, not by GitHub's "latest release" — that repository also publishes plugin releases,
and its newest release is usually one of those. Versions are compared numerically per segment, so
`1.10.0` correctly outranks `1.9.0`.

Drop `*.wav`/`*.flac`/`*.mp3` files into `wrapper/samples/` to add your own. A voice can be either:

* a **single audio file** — `samples/Alphinaud.wav`, or
* a **voice folder** — `samples/Alphinaud/` holding several clips of the *same* voice. One clip is
  picked **at random per request** so repeated lines vary naturally. Folders are **one level only**
  (sub-directories are ignored) and a folder **shadows** a same-named single file
  (`samples/Alphinaud/` wins over `samples/Alphinaud.wav`).

Requests reference a voice by **basename only** (path traversal is rejected); the extension is
**ignored** — `Alphinaud`, `Alphinaud.wav` and `Alphinaud.mp3` all resolve to the same voice. For the
`f5` backend a reference transcript is taken, for the *chosen* clip, from a sidecar `<clip>.txt`
(optional, same name as the audio), else the request's `ref_text`, else auto-transcribed via F5-TTS's
built-in ASR and cached. The `xtts` backend needs no transcript (it clones from the audio alone).

## TTS backends

The wrapper ships three interchangeable engines. A process loads **one** at startup, selected by
`tts_backend` (config) or `--tts-backend <f5|xtts>` (default `f5`). `GET /health` reports
the active backend. All engines are installed by the bootstrap, so switching is only a restart.

| Backend | Model | Reference transcript | Notes |
|---------|-------|----------------------|-------|
| `f5`   | F5-TTS finetunes (per language) | needed — from a sidecar `.txt`, the request's `ref_text`, or auto-transcribed via F5's built-in ASR | code MIT, weights CC-BY-NC |
| `xtts` | Coqui XTTS-v2 (one multilingual model) | **not needed** — clones from the audio alone | code MPL-2.0, weights CPML (non-commercial) |

## Languages

Every request may carry a `language` field; how it is treated depends on the backend. `f5` serves
**one language per process**, chosen at startup via `language` (config) or `--language <code>`, with
each language mapping to a distinct model in the `languages` config block. `xtts`
each cover all their languages with a **single multilingual model**, so the per-request `language`
picks the target language on the fly — **no restart or reload**. Omitting `language` falls back to
the startup language in every case. `GET /languages` answers this per backend (which is what the web
UI uses to grey the selector out for `f5`).

| Lang | F5 model | Source |
|------|----------|--------|
| `en` | `F5TTS_v1_Base` | official multilingual base (auto-downloaded) |
| `de` | `F5TTS_Base` finetune | `hvoss-techfak/F5-TTS-German` (CC-BY-NC) |
| `fr` | `F5TTS_Base` finetune | `RASPIAUDIO/F5-French-MixedSpeakers-reduced` (CC-BY-NC) |
| `ja` | `F5TTS_Base` finetune | `Jmica/F5TTS` (CC-BY-NC) |

The bootstrap **downloads all four** F5 checkpoints (and the XTTS-v2 model) at install time, so
switching the F5 language (or the backend) is just a restart with a different `--language` /
`--tts-backend` (no re-download). `GET /health` reports the startup language. Language handling is
backend-aware: on `f5`, a per-request `language` is **ignored** — the loaded/startup model is always
used (F5 can only voice its one finetune); on `xtts`, any of its supported codes
(`en es fr de it pt pl tr ru nl cs ar zh-cn ja hu ko hi`) is accepted per request and an unsupported
code returns `400` (note that Chinese is spelled `zh-cn`). Provide a **reference sample in the
target language** (and, for `f5`, ideally a matching `.txt` transcript) for best results. To add/replace an F5 language, edit the `languages`
map in `wrapper/config.json` (verify repo + file names against F5-TTS `SHARED.md`).

## Development

```bash
cd wrapper
uv venv .venv-test --python 3.11
uv pip install --python .venv-test pytest pytest-asyncio httpx fastapi numpy
.venv-test/Scripts/python -m pytest -q   # 0 failures expected
```

The unit suite mocks F5-TTS/torch, so it runs anywhere without GPU or multi-GB downloads.

## Licensing

- **Wrapper code: AGPL-3.0.**
- **The weights differ per engine, and two of the three are NON-COMMERCIAL.** They are kept strictly
  separate from this AGPL code and are *not* shipped in this repo — the bootstrap downloads them at
  runtime into `wrapper/models/`.
  - F5-TTS weights (base + finetunes): **CC-BY-NC-4.0** — non-commercial.
  - XTTS-v2 weights: **Coqui Public Model License (CPML)** — non-commercial.
  The synthesized audio (model *output*) inherits the terms of whichever engine produced it, so with
  `f5` or `xtts` **you may not use it commercially** without a separate license from the rights
  holder. Keep each model's license notice with any distribution and do not relicense the weights.
- The installed Python packages carry their own (mostly permissive) licenses — f5-tts is **MIT**,
  coqui-tts is **MPL-2.0**, the rest are BSD/MIT/Apache.
- See [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) for the complete manifest of installed
  packages and models with their licenses, and [`licenses/`](licenses/) for vendored full-text
  licenses (currently the CPML, whose original host `coqui.ai` is offline).
