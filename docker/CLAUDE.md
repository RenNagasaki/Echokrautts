# Docker-Image + GitHub-Actions
Aus der Root-`CLAUDE.md` verschoben (2026-11-13). Liegt hier, weil `docker/entrypoint.sh` hier wohnt;
gilt für `Dockerfile`, `docker-compose*.yml` und `.github/workflows/` im Repo-Root.

## Docker image + GitHub Action (`Dockerfile`, `.github/workflows/docker-release.yml`)
- **The image does NOT run the bootstrap.** Steps 1–4 (uv, Python pin, GPU detection, deps) happen at
  *build* time; `docker/entrypoint.sh` only does what is left: download the ACTIVE backend's weights
  into the volume (`python -m src.models` for f5 / `python -m src.xtts_backend` for xtts, both
  idempotent) and `exec python -m uvicorn src.server:create_app --factory`. `serve` (default),
  `download` (pre-fill volume, no server) and "anything else" (`bash`, run verbatim, no download) are
  the three entrypoint modes; `ECHOKRAUTTS_SKIP_DOWNLOAD=1` suppresses the download.
- **`python -m` / `python -c` with cwd `/app/wrapper` is load-bearing**: both prepend cwd to
  `sys.path`, so `src` resolves to `/app/wrapper/src` — NOT the copy pip installed into the venv —
  and `WRAPPER_ROOT` (= parent of the `src` package) points at `/app/wrapper`, where `config.json`
  lives. Same relationship as the native install (venv + cwd = wrapper root).
- **AMD (`-rocm` variant, added 2026-08-07).** Third image from the same Dockerfile:
  `TORCH_INDEX_URL=…/rocm6.4` + `TORCH_VERSION=2.8.0` + `GPU_BACKEND=rocm`. **Why 2.8.0 and not the
  2.7.0 pin:** the rocm6.4 index has no 2.7.x (rocm6.3 does, but 6.3 predates RDNA4/RX 9000), and
  **2.8 is the LAST torchaudio whose `load` decodes natively** — from 2.9 it is an alias for
  `load_with_torchcodec`, which needs system FFmpeg, exactly what the pin avoids. So the variant
  moves the pin by the minimum that buys RDNA4 without giving up the FFmpeg-free property. Compose
  file passes `/dev/kfd` + `/dev/dri` + `group_add: video,render` (AMD is not `--gpus`). Covers
  Linux hosts AND Windows users through WSL2/Docker Desktop — which is the *only* way AMD accelerates
  under Windows today. **Untested on real hardware** (no AMD GPU here; user plans a vast.ai box).
  **Native Windows AMD is deliberately NOT done (step 2):** ROCm 7.2.1 supports it, but only with
  Python 3.12 + torch 2.9.1 from individual wheel URLs (`repo.radeon.com/rocm/windows/rocm-rel-7.2.1/`,
  plus four `rocm_sdk_*` wheels) — a second install path in the bootstrap, and 2.9's torchaudio drags
  in FFmpeg. DirectML is a dead end: Microsoft has it in **maintenance mode** (security fixes only,
  Windows ML is the successor) and `torch-directml` has not shipped since 2024-09.
- **One Dockerfile, three variants, build args only.** `TORCH_INDEX_URL` selects cu128 / cpu / rocm; there is
  deliberately **no `nvidia/cuda` base image** — the cu128 wheels bundle their CUDA runtime
  (`nvidia-*-cu12`) exactly like the bare-metal install, the driver comes from
  nvidia-container-toolkit. Builder stage (build-essential/git, `/opt/venv`) is thrown away; the
  runtime stage copies only the venv + `src` + `config.json` and adds `libsndfile1` (soundfile
  backend — no FFmpeg, matching the torch 2.7 pin) and `curl` (healthcheck).
- The build **mirrors `bootstrap.step_deps` exactly**, order included: pinned torch → both engines in
  ONE resolution (`/src/wrapper` + `coqui-tts` + transformers pin) → re-pin torch → uninstall
  torchcodec → verify (torch version, torchcodec absent, `isin_mps_friendly` importable). The verify
  step fails the BUILD, so a bad resolve can never ship as an image that only dies on first inference.
- **Weights are never baked in** — non-commercial licenses (F5 CC-BY-NC-4.0, XTTS-v2 CPML) make
  redistribution in a public image a licensing problem. They land in the `/data/models` volume on
  first start. `HEALTHCHECK --start-period=45m` exists because of that first download.
- **Volumes:** `/data/samples` + `/data/models` (`F5W_SAMPLES_DIR`/`F5W_MODELS_DIR` absolute → honored
  by `config.py`). `TTS_HOME` also points at `/data/models` so the Coqui cache shares the volume;
  the HF cache does anyway (`cache_dir=config.models_path` in `models.py`).
- **Action triggers on `release: published` only** (plus `workflow_dispatch` with a version input) —
  *not* on push: GitHub only ever sees mirrored pushes, and each would burn ~40 min of runner time.
  Tags: `<version>` + `latest` (CUDA), `<version>-cpu` + `latest-cpu`; `latest*` is skipped for
  pre-releases. GHCR auth is the job's `GITHUB_TOKEN` — **no extra secret**. The image name is
  lower-cased in the `meta` step (`RenNagasaki` → `rennagasaki`; GHCR rejects upper-case paths). The
  runner needs the `Free disk space` step for the ~8 GB CUDA image. The CPU job additionally smoke-
  tests the pushed image (import torch + `TTS.utils.manage` + `create_app`, no weights).
- `docker-compose.yml` (GPU, nvidia device reservation) and `docker-compose.cpu.yml` (standalone —
  a device *reservation* cannot be cleanly removed by an override file) are the one-click deploys.
- **`.github/workflows/docker-build.yml` — build on demand, no release needed.** Actions → "Docker
  build (manual)", pick the branch (main by default), `variant` (cuda/cpu/both) and `tag` (default
  `main`). Pushes `<tag>` / `<tag>-cpu` **and** `sha-<commit>` to GHCR; never touches `latest`, which
  belongs to releases. Same cache scope as the release workflow (`<variant>`), so a manual build
  right after a release rebuilds only what changed. The matrix is computed by a `prepare` job because
  the `matrix` context is NOT available in a job-level `if` — variants cannot be filtered otherwise.
- ⚠ **Not yet built/verified anywhere** — no Docker on the dev machine. First real proof is a
  `workflow_dispatch` run on GitHub (use the test workflow first: it pushes nothing).

## Voice-Pack beim Container-Start (`wrapper/src/voicepack.py`)
- `src/voicepack.py` — first-start voice download. `ensure_voicepack(config, opener=None)` is a no-op
  when `samples` already holds **audio** (sidecar `.txt` files alone do NOT count — that folder is
  still unusable) or when `voicepack_auto_download` is false; otherwise it fetches the newest pack
  and unpacks it into `samples_path`. **`pick_release` filters by TAG PREFIX
  (`EK-VoicePack-`), never GitHub's `/releases/latest`** — that repo also publishes plugin releases,
  and the newest release overall is usually one (verified live 2026-08-07: newest = `0.19.3.0`,
  picked = `EK-VoicePack-1.1.0`). `version_key` compares numerically per segment so `1.10.0` beats
  `1.9.0`. Drafts/prereleases skipped. The asset is **streamed to a `.part` file next to the samples
  folder** (>100 MB — not held in memory; a partial file must never look like content, and it is
  removed in a `finally`), progress every ~5%. `_safe_members` is a **zip-slip guard**: absolute
  paths, `..` and anything resolving outside the target are skipped, not trusted. **Never raises** —
  no voices is recoverable by dropping in a wav, a wrapper that refuses to start is not.
  ⚠ **NUR aus `docker/entrypoint.sh` aufgerufen — der native Bootstrap holt seit 2026-09-10 KEINE
  Stimmen mehr** (User-Entscheidung: „Docker sollte voicepack selber laden, standalone nicht“). Die
  Stimmen gehören dem Echokraut-Plugin, das sie selbst installiert, und eine native Installation wird
  immer von diesem Plugin gefahren — der Wrapper wäre dort ein zweiter Schreiber im selben Ordner.
  Im Container gibt es kein Plugin und ein leeres Volume macht jedes `/tts` zu einem 404, deshalb
  bleibt der Weg dort. Ein Test in `test_bootstrap_repair.py` prüft die abgesetzten BEFEHLE, nicht die
  Existenz des Moduls — das Modul bleibt ja, also würde nur der Aufruf einen Rückfall verraten.
  Entry point `python -m src.voicepack`. **Live-verified
  2026-08-07:** 523 files (260 wav + 262 txt sidecars + 1 csv, flat, no top-level folder) in ~5 s,
  `SampleService` then lists 260 voices; rerun skips; no `.part` left behind.
