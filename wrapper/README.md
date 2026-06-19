# F5-TTS Wrapper

A lightweight, plug-and-play Python wrapper around [F5-TTS](https://github.com/SWivid/F5-TTS)
that a host application (e.g. a C#/Dalamud plugin) starts as a **separate process** and drives
over stdout (NDJSON events) and HTTP (streaming PCM). It provides zero-shot **voice-cloning TTS**
with sentence-level streaming, a VRAM-aware worker pool, and self-contained installation via
[`uv`](https://github.com/astral-sh/uv) — no system Python or git required.

See [`../F5-TTS-Wrapper-SPEC.md`](../F5-TTS-Wrapper-SPEC.md) for the full specification.

## Quick start

```bash
# From the host: starts (and on first run installs) everything, then serves.
python bootstrap/bootstrap.py --start --parent-pid <host_pid>
```

On Windows/Linux without a system Python, use the bundled starters instead
(`bootstrap/install_win.ps1` / `bootstrap/install_linux.sh`) — they fetch `uv` first.

The bootstrap runs a fixed 6-step sequence (obtain `uv` → pin Python → detect GPU → install deps →
preload model → serve) and reports each step as an NDJSON `progress` event on stdout. When the
server is listening it emits a `ready` event with host/port/backend.

### Why torch is pinned to 2.7.x

`torch`/`torchaudio` are pinned (configurable via `torch_version`/`torchaudio_version`). Newer
torchaudio routes `torchaudio.load` through **torchcodec**, which requires system **FFmpeg** shared
libraries — an external, non-self-contained dependency. On 2.7.x, `torchaudio.load` still uses the
bundled-libsndfile **soundfile** backend, so the wrapper installs and runs with **no external
binaries**. f5-tts declares `torchcodec` as a dependency but never imports it (it only calls
`torchaudio.load`), so the bootstrap re-pins torch after the project install and drops the unused
torchcodec. Bump the pins deliberately and re-verify the soundfile path if you change them.

## HTTP API

| Method & path        | Purpose                                                            |
|----------------------|--------------------------------------------------------------------|
| `POST /tts`          | Streaming synthesis. Body = raw PCM (`pcm_s16le`, mono). Metadata in `X-Job-Id` / `X-Sample-Rate` / `X-Channels` / `X-Sample-Format` headers. |
| `GET /samples`       | Usable voice-sample names (`?details=true` adds `has_ref_text`/`bytes`). |
| `POST /cancel/{id}`  | Cancel a running job.                                              |
| `GET /jobs/{id}`     | Live progress (`sentences_done`/`sentences_total`/`percent`).      |
| `GET /health`        | Backend/device/worker/queue status.                               |
| `POST /shutdown`     | Graceful shutdown.                                                 |

Configuration lives in `config.json` (overridable by `F5W_*` env vars and `--kebab-case` CLI flags;
precedence JSON < ENV < CLI). For remote use set `host` to `0.0.0.0` and an `api_key` (then all
requests need `Authorization: Bearer <key>`).

## Voice samples

Drop `*.wav`/`*.flac`/`*.mp3` files into `samples/`. Requests reference a sample by **basename only**
(path traversal is rejected). A reference transcript is taken from a sidecar `<name>.txt`, else the
request's `ref_text`, else auto-transcribed via F5-TTS's built-in ASR and cached.

## Languages

A wrapper process serves **one language**, chosen at startup via `language` (config) or
`--language <code>`. Each language maps to a model in the `languages` config block:

| Lang | Model | Source |
|------|-------|--------|
| `en` | `F5TTS_v1_Base` | official multilingual base (auto-downloaded) |
| `de` | `F5TTS_Base` finetune | `hvoss-techfak/F5-TTS-German` (CC-BY-NC) |
| `fr` | `F5TTS_Base` finetune | `RASPIAUDIO/F5-French-MixedSpeakers-reduced` (CC-BY-NC) |
| `ja` | `F5TTS_Base` finetune | `Jmica/F5TTS` (CC-BY-NC) |

The bootstrap **downloads all four** checkpoints at install time, so switching language is just a
restart with a different `--language` (no re-download). `GET /health` reports the active language.
A `/tts` request whose `language` field doesn't match the loaded language is rejected with `400`
(prevents synthesizing in the wrong accent). Provide a **reference sample in the target language**
(and ideally a matching `.txt` transcript) for best results. To add/replace a language, edit the
`languages` map in `config.json` (verify repo + file names against F5-TTS `SHARED.md`).

## Development

```bash
uv venv .venv-test --python 3.11
uv pip install --python .venv-test pytest pytest-asyncio httpx fastapi numpy
.venv-test/Scripts/python -m pytest -q   # 0 failures expected
```

The unit suite mocks F5-TTS/torch, so it runs anywhere without GPU or multi-GB downloads.

## Licensing

- **Wrapper code: AGPL-3.0.**
- **F5-TTS model weights: CC-BY-NC** — kept strictly separate. They are *not* part of this AGPL code
  and are *not* shipped here; they are downloaded at runtime from HuggingFace into `models/`. Keep
  the model's license notice with any distribution and do not relicense the weights.
