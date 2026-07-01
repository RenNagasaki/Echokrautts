# Third-party licenses

This wrapper **does not bundle or redistribute** any of the packages or model weights below — the
bootstrap downloads them at runtime into the machine-local `.venv/` and `models/` directories. This
file is a manifest of what gets installed and under which license, so downstream users can comply
with those licenses (especially the **non-commercial** model terms).

The wrapper's own code is **AGPL-3.0** (see [`README.md`](README.md) → Licensing).

## ⚠ Model weights are NON-COMMERCIAL

**Every** speech model this wrapper can download is licensed for non-commercial use only. The
synthesized audio (the model *output*) inherits these terms — you may **not** use it for any purpose
that earns direct or indirect payment without a separate commercial license from the respective
rights holder.

| Model | Language(s) | HuggingFace repo | License |
|-------|-------------|------------------|---------|
| F5-TTS base (`F5TTS_v1_Base`) | en (multilingual base) | [`SWivid/F5-TTS`](https://huggingface.co/SWivid/F5-TTS) | [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| F5-TTS German finetune | de | [`hvoss-techfak/F5-TTS-German`](https://huggingface.co/hvoss-techfak/F5-TTS-German) | CC-BY-NC-4.0 |
| F5-TTS French finetune | fr | [`RASPIAUDIO/F5-French-MixedSpeakers-reduced`](https://huggingface.co/RASPIAUDIO/F5-French-MixedSpeakers-reduced) | CC-BY-NC-4.0 |
| F5-TTS Japanese finetune | ja | [`Jmica/F5TTS`](https://huggingface.co/Jmica/F5TTS) | CC-BY-NC-4.0 |
| XTTS-v2 | en/de/fr/ja (+13 more) | [`coqui/XTTS-v2`](https://huggingface.co/coqui/XTTS-v2) | Coqui Public Model License (CPML) 1.0.0 — see [`licenses/XTTS-v2-CPML.txt`](licenses/XTTS-v2-CPML.txt) |

**CPML note:** Coqui Inc. shut down in January 2024, so there is currently no vendor to sell an XTTS
commercial license — treat XTTS-v2 as strictly non-commercial. The full CPML text is vendored at
[`licenses/XTTS-v2-CPML.txt`](licenses/XTTS-v2-CPML.txt) because its original home (`coqui.ai`) is no
longer guaranteed to be online.

## Python packages (installed into `.venv/`)

The bootstrap installs these via `uv pip install` (torch from a backend-specific index; f5-tts +
coqui-tts as the two TTS engines). Transitive dependencies not listed here carry their own licenses.

| Package | Role | License |
|---------|------|---------|
| [`torch`](https://github.com/pytorch/pytorch) / `torchaudio` | inference runtime | BSD-3-Clause |
| [`f5-tts`](https://github.com/SWivid/F5-TTS) | F5 backend engine (code only — weights above) | MIT (© 2024 Yushen CHEN) |
| [`coqui-tts`](https://github.com/idiap/coqui-ai-TTS) | XTTS backend engine (maintained idiap fork; code only) | MPL-2.0 |
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

## Note on the two engines

- **f5-tts code is MIT** (permissive) but its **model weights are CC-BY-NC** — the two are separate.
  You may use the f5-tts code commercially; the weights only non-commercially.
- **coqui-tts code is MPL-2.0** (permissive, file-level copyleft) but the **XTTS-v2 weights are
  CPML** (non-commercial). Same split.

In both cases the *weights* — and therefore any synthesized audio — are the binding non-commercial
constraint for typical use of this wrapper.
