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
start-f5tts.bat      # Windows, F5-TTS backend (visible window, pauses at the end)
./start-f5tts.sh     # Linux/macOS, F5-TTS backend
start-xtts.bat       # Windows, XTTS-v2 backend
./start-xtts.sh      # Linux/macOS, XTTS-v2 backend
```

Both engines and all their weights are installed either way; the launcher only picks (via
`--tts-backend f5` / `--tts-backend xtts`) which engine the worker pool loads at startup, so
switching is a restart, not a reinstall. All launchers forward extra arguments, e.g.
`start-xtts.bat --language en`.

Under the hood these call `wrapper/bootstrap/install_win.ps1` / `install_linux.sh` (which fetch `uv`)
with `--start --tts-backend <…>`. A host process that already has a Python can run the bootstrap directly:

```bash
python wrapper/bootstrap/bootstrap.py --start --parent-pid <host_pid>
```

The bootstrap runs a fixed 6-step sequence (obtain `uv` → pin Python → detect GPU → install deps →
preload models → serve) and reports each step as an NDJSON `progress` event on stdout. When the
server is listening it emits a `ready` event with host/port/backend. **Both TTS engines and all of
their model weights are installed once**, so switching engine or language is only a restart (see
[TTS backends](#tts-backends)).

### Why torch is pinned to 2.7.x

`torch`/`torchaudio` are pinned (configurable via `torch_version`/`torchaudio_version`). Newer
torchaudio routes `torchaudio.load` through **torchcodec**, which requires system **FFmpeg** shared
libraries — an external, non-self-contained dependency. On 2.7.x, `torchaudio.load` still uses the
bundled-libsndfile **soundfile** backend, so the wrapper installs and runs with **no external
binaries**. The TTS engines declare `torchcodec` as a dependency but never import it (they only call
`torchaudio.load`), so the bootstrap re-pins torch after the deps install and drops the unused
torchcodec. Bump the pins deliberately and re-verify the soundfile path if you change them.

## HTTP API

| Method & path        | Purpose                                                            |
|----------------------|--------------------------------------------------------------------|
| `POST /tts`          | Streaming synthesis. Body = raw PCM (`pcm_s16le`, mono). Metadata in `X-Job-Id` / `X-Sample-Rate` / `X-Channels` / `X-Sample-Format` headers. |
| `GET /samples`       | Usable voice names (`?details=true` adds `has_ref_text`/`bytes`/`count`). |
| `POST /cancel/{id}`  | Cancel a running job.                                              |
| `GET /jobs/{id}`     | Live progress (`sentences_done`/`sentences_total`/`percent`).      |
| `GET /health`        | Backend/device/worker/queue status.                               |
| `POST /shutdown`     | Graceful shutdown.                                                 |

Configuration lives in `wrapper/config.json` (overridable by `F5W_*` env vars and `--kebab-case` CLI
flags; precedence JSON < ENV < CLI). For remote use set `host` to `0.0.0.0` and an `api_key` (then
all requests need `Authorization: Bearer <key>`).

## Voice samples

Drop `*.wav`/`*.flac`/`*.mp3` files into `wrapper/samples/`. A voice can be either:

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

The wrapper ships two interchangeable engines. A process loads **one** at startup, selected by
`tts_backend` (config) or `--tts-backend <f5|xtts>` (default `f5`). `GET /health` reports the active
backend. Both engines are installed by the bootstrap, so switching is only a restart.

| Backend | Model | Reference transcript | Notes |
|---------|-------|----------------------|-------|
| `f5`   | F5-TTS finetunes (per language) | needed — from a sidecar `.txt`, the request's `ref_text`, or auto-transcribed via F5's built-in ASR | code MIT, weights CC-BY-NC |
| `xtts` | Coqui XTTS-v2 (one multilingual model) | **not needed** — clones from the audio alone | code MPL-2.0, weights CPML (non-commercial) |

Both output mono `pcm_s16le` at 24000 Hz, so the HTTP contract is identical. `f5` emits one PCM
block per sentence; `xtts` does **token streaming** (`inference_stream`), emitting many smaller PCM
parts as they are generated for much lower first-audio latency (`stream_chunk_size` tunes the
granularity, default 20). XTTS is served via the maintained
[`coqui-tts`](https://github.com/idiap/coqui-ai-TTS) fork (the original `TTS` package is
unmaintained). See [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) for the full license matrix.

## Languages

Every request may carry a `language` field; how it is treated depends on the backend. `f5` serves
**one language per process**, chosen at startup via `language` (config) or `--language <code>`, with
each language mapping to a distinct model in the `languages` config block. `xtts` covers all
languages with its **single multilingual model**, so the per-request `language` picks the target
language on the fly — **no restart or reload**. Omitting `language` falls back to the startup
language in both cases.

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
code returns `400`. Provide a **reference sample in the target language** (and, for `f5`, ideally a
matching `.txt` transcript) for best results. To add/replace an F5 language, edit the `languages`
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
- **All speech-model weights are NON-COMMERCIAL** and are kept strictly separate from this AGPL
  code: F5-TTS weights (base + finetunes) are **CC-BY-NC-4.0**, XTTS-v2 weights are under the
  **Coqui Public Model License (CPML)**. They are *not* shipped in this repo — the bootstrap
  downloads them at runtime into `wrapper/models/`. The synthesized audio (model *output*) inherits
  these terms, so **you may not use it commercially** without a separate license from the rights
  holder. Keep each model's license notice with any distribution and do not relicense the weights.
- The installed Python packages carry their own (mostly permissive) licenses — f5-tts is **MIT**,
  coqui-tts is **MPL-2.0**, the rest are BSD/MIT/Apache.
- See [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) for the complete manifest of installed
  packages and models with their licenses, and [`licenses/`](licenses/) for vendored full-text
  licenses (currently the CPML, whose original host `coqui.ai` is offline).
