# Third-party licenses

This wrapper **does not bundle or redistribute** any of the packages or model weights below — the
bootstrap downloads them at runtime into the machine-local `.venv/` and `models/` directories. This
file is a manifest of what gets installed and under which license, so downstream users can comply
with those licenses (especially the **non-commercial** model terms).

The wrapper's own code is **AGPL-3.0** (see [`README.md`](README.md) → Licensing).

## ⚠ Most model weights are NON-COMMERCIAL

Of the three engines, **two ship non-commercial weights** (F5-TTS and XTTS-v2) and one does not
(Chatterbox Multilingual, MIT). The synthesized audio (the model *output*) inherits the terms of
whichever model produced it — with F5 or XTTS you may **not** use it for any purpose that earns
direct or indirect payment without a separate commercial license from the respective rights holder.
Since the active engine is a startup flag, **check which backend produced a given clip** before
using it commercially; `GET /health` reports it.

| Model | Language(s) | HuggingFace repo | License |
|-------|-------------|------------------|---------|
| F5-TTS base (`F5TTS_v1_Base`) | en (multilingual base) | [`SWivid/F5-TTS`](https://huggingface.co/SWivid/F5-TTS) | [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| F5-TTS German finetune | de | [`hvoss-techfak/F5-TTS-German`](https://huggingface.co/hvoss-techfak/F5-TTS-German) | CC-BY-NC-4.0 |
| F5-TTS French finetune | fr | [`RASPIAUDIO/F5-French-MixedSpeakers-reduced`](https://huggingface.co/RASPIAUDIO/F5-French-MixedSpeakers-reduced) | CC-BY-NC-4.0 |
| F5-TTS Japanese finetune | ja | [`Jmica/F5TTS`](https://huggingface.co/Jmica/F5TTS) | CC-BY-NC-4.0 |
| XTTS-v2 | en/de/fr/ja (+13 more) | [`coqui/XTTS-v2`](https://huggingface.co/coqui/XTTS-v2) | Coqui Public Model License (CPML) 1.0.0 — see [`licenses/XTTS-v2-CPML.txt`](licenses/XTTS-v2-CPML.txt) |
| Chatterbox Multilingual (`t3_mtl23ls_v2` + `s3gen`/`ve`) | 23 languages incl. en/de/fr/ja | [`ResembleAI/chatterbox`](https://huggingface.co/ResembleAI/chatterbox) | **MIT** (© 2025 Resemble AI) — *not* restricted to non-commercial use |

**CPML note:** Coqui Inc. shut down in January 2024, so there is currently no vendor to sell an XTTS
commercial license — treat XTTS-v2 as strictly non-commercial. The full CPML text is vendored at
[`licenses/XTTS-v2-CPML.txt`](licenses/XTTS-v2-CPML.txt) because its original home (`coqui.ai`) is no
longer guaranteed to be online.

## Python packages (installed into `.venv/`)

The bootstrap installs these via `uv pip install` (torch from a backend-specific index; f5-tts +
coqui-tts in one resolution, chatterbox-tts separately with `--no-deps` because its own pins are
incompatible with the other two). Transitive dependencies not listed here carry their own licenses.

| Package | Role | License |
|---------|------|---------|
| [`torch`](https://github.com/pytorch/pytorch) / `torchaudio` | inference runtime | BSD-3-Clause |
| [`f5-tts`](https://github.com/SWivid/F5-TTS) | F5 backend engine (code only — weights above) | MIT (© 2024 Yushen CHEN) |
| [`coqui-tts`](https://github.com/idiap/coqui-ai-TTS) | XTTS backend engine (maintained idiap fork; code only) | MPL-2.0 |
| [`chatterbox-tts`](https://github.com/resemble-ai/chatterbox) | Chatterbox backend engine (installed `--no-deps`, see below) | MIT (© 2025 Resemble AI) |
| [`resemble-perth`](https://github.com/resemble-ai/Perth) | audio watermarker Chatterbox applies to every clip | MIT |
| [`diffusers`](https://github.com/huggingface/diffusers) · [`conformer`](https://github.com/lucidrains/conformer) · [`s3tokenizer`](https://github.com/xingchensong/S3Tokenizer) | Chatterbox model building blocks | Apache-2.0 · MIT · Apache-2.0 |
| [`librosa`](https://github.com/librosa/librosa) · [`pyloudnorm`](https://github.com/csteinmetz1/pyloudnorm) · [`omegaconf`](https://github.com/omry/omegaconf) · [`einops`](https://github.com/arogozhnikov/einops) | Chatterbox audio / config helpers | ISC · MIT · BSD-3-Clause · MIT |
| [`pykakasi`](https://github.com/miurahr/pykakasi) | Japanese text processing for Chatterbox — **GPL-3.0-or-later**, see note below | GPL-3.0-or-later |
| [`fastapi`](https://github.com/fastapi/fastapi) | HTTP API | MIT |
| [`uvicorn`](https://github.com/encode/uvicorn) | ASGI server | BSD-3-Clause |
| [`soundfile`](https://github.com/bastibe/python-soundfile) | audio I/O (`torchaudio.load` backend) | BSD-3-Clause |
| [`numpy`](https://github.com/numpy/numpy) | arrays / PCM conversion | BSD-3-Clause |
| [`pydantic`](https://github.com/pydantic/pydantic) | request models | MIT |
| [`huggingface_hub`](https://github.com/huggingface/huggingface_hub) | model download (pulled transitively) | Apache-2.0 |

## Tooling (not installed into the venv)

| Tool | Role | License |
|------|------|---------|
| [`uv`](https://github.com/astral-sh/uv) | fetched by the bootstrap to create the venv / install deps | Apache-2.0 OR MIT |

## Note on the three engines

- **f5-tts code is MIT** (permissive) but its **model weights are CC-BY-NC** — the two are separate.
  You may use the f5-tts code commercially; the weights only non-commercially.
- **coqui-tts code is MPL-2.0** (permissive, file-level copyleft) but the **XTTS-v2 weights are
  CPML** (non-commercial). Same split.
- **chatterbox-tts is MIT for both code and weights** — no such split, and the only engine here
  whose output carries no non-commercial restriction. Note that every clip it produces contains
  Resemble's inaudible `perth` watermark; that is a provenance marker, not a license term.

For F5 and XTTS the *weights* — and therefore any synthesized audio — are the binding non-commercial
constraint for typical use of this wrapper.

### `pykakasi` is GPL-3.0-or-later

It is the one copyleft item in the list, installed because Chatterbox uses it for Japanese text
processing. This wrapper is **AGPL-3.0**, so combining with GPL-3.0-or-later code is compatible; it
matters only if you plan to redistribute the installed environment under different terms. Chatterbox
imports it **lazily and optionally** — without it, it logs "pykakasi not available - Japanese text
processing skipped" and continues — so it can simply be removed from the `chatterbox_install.deps`
list in `wrapper/config.json` if you do not need Japanese. The same list omits `spacy-pkuseg`
(Chinese segmentation) by default; add it there if you need `zh`.
