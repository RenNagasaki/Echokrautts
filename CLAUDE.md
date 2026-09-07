# Echokrautts — F5-TTS Wrapper

**Not a C# Dalamud plugin.** This repo is a standalone **Python** wrapper around F5-TTS, started as a
separate OS process by a C#/Dalamud host (the Echokraut plugin lives elsewhere) and driven over
stdout (NDJSON) + HTTP (streaming PCM). The parent `Dalamud/CLAUDE.md` conventions are C#-specific
(`dotnet build`, xUnit, SonarQube) and **do not apply here**. (The old `F5-TTS-Wrapper-SPEC.md`
was removed on user request — this CLAUDE.md + the code are now the source of truth. NOTE: many
code docstrings still cite "SPEC §…" section numbers as historical anchors; the referenced doc no
longer exists.)

## Build / Test
- Tooling: `uv` (at `~/.local/bin/uv`) + system Python 3.12. No dotnet here.
- **Unit tests** (mock torch/f5-tts, run anywhere): `wrapper/.venv-test` with `pytest pytest-asyncio
  httpx fastapi numpy`, then `.venv-test/Scripts/python -m pytest -q` → **96 passed, 1 skipped**
  (skip = symlink test, Windows privilege). Suite must stay green.
- **Runtime venv** (`wrapper/.venv`, gitignored): created by the bootstrap with torch+f5-tts.

## One-Click-Starter (Repo-Root)
- **Pro Backend ein Starter-Paar** im Repo-Root, manuelle 1-Klick-Launcher: `start-f5tts.bat`/
  `start-f5tts.sh` (F5-TTS) und `start-xtts.bat`/`start-xtts.sh` (XTTS-v2). Der einzige Unterschied
  ist das durchgereichte `--tts-backend f5` bzw. `xtts` — der Bootstrap installiert eh
  BEIDE Engines + alle Weights, der Starter wählt nur, welche der Worker-Pool beim Start lädt
  (Umschalten = Neustart, kein Reinstall). Sie rufen die jeweiligen `wrapper/bootstrap/install_*`-
  Starter mit `--start --tts-backend <…>` auf (uv holen → installieren → servieren) und reichen
  Extra-Args durch (z.B. `--language en`). Die `.bat`-Varianten laufen in sichtbarem Fenster mit
  `pause` am Ende; die `.sh`-Varianten machen `exec` auf `install_linux.sh`. Unterschied zum C#-Host:
  der startet `bootstrap.py` direkt/versteckt — diese Starter sind für den Menschen am Rechner
  gedacht. (Früher: ein einzelnes `start.bat`/`start.sh` ohne Backend-Wahl.)

## Project layout (`wrapper/`)
- `src/config.py` — `Config` dataclass; load order JSON < ENV (`F5W_*`) < CLI (`--kebab`). Holds
  `language` (active, startup) + `languages` map (en/de/fr/ja → arch/hf_repo/ckpt_file/vocab_file)
  + `tts_backend` (`"f5"` | `"xtts"`, one backend per process) + `xtts_fp16` (opt-in half-precision
  for XTTS, default off; only takes effect on a CUDA device). `host` default is `0.0.0.0` (binds all
  interfaces so the host reaches it with no config edit — set `api_key` or narrow to `127.0.0.1` to
  lock down). Bool ENV/CLI coercion is name-listed in `_coerce` (`asr_for_missing_ref_text`,
  `xtts_fp16`) — add new bool fields there or they stay strings.
- `src/models.py` — F5 model resolution. `resolve_model(config, lang)` → `ResolvedModel(arch,
  ckpt_file, vocab_file)` via `hf_hub_download` (cached); `download_all(config)` for install;
  `python -m src.models` entry. Used by BOTH bootstrap (download) and engine (load) — single source
  of truth, no duplication.
- **Custom model override** (`CUSTOM_MODEL_DIRNAME = "echokraut_custom"` in `config.py`,
  `config.custom_model_path` = `models/echokraut_custom`): a user-supplied model dropped in that
  folder (by the host's "install custom data" flow) overrides the configured model for the ACTIVE
  engine at load time, auto-detected by folder presence (no config passing). F5:
  `models.resolve_custom_model(config)` returns local `ResolvedModel` when the folder holds a
  checkpoint (`*.safetensors`→`*.pt`→`*.ckpt`, first sorted; optional `vocab.txt`; `arch.txt`
  overrides arch, default `F5TTS_Base`) — applied only for the active language so other languages
  keep their HF mapping. XTTS: `xtts_backend._resolve_custom_model_dir(config)` returns the dir when
  it holds `config.json` + `model.pth`/`model.safetensors`. The two formats are disjoint
  (`.pth`≠`.pt`, XTTS needs `config.json`) so both backends share the one folder safely.
- `src/xtts_backend.py` — XTTS-v2 backend (second engine). `XTTSWorker(WorkerProtocol)`: same
  `infer()` contract as F5 (float32 @ 24000 Hz per chunk), but clones from audio only (`transcribe`
  is a no-op, no ref-text). Conditioning latents cached per sample. `_resolve_model_dir(config)`
  downloads XTTS-v2 via coqui `ModelManager` (used by both bootstrap + worker, mirrors `models.py`);
  `download_model(config)` + `python -m src.xtts_backend` entry for install. **fp16 speedup:**
  `_should_use_fp16(config, device)` (pure, unit-tested) gates half precision after load — true only
  when `config.xtts_fp16` AND the *resolved* device is `"cuda"` (dml/xpu resolve to cpu → excluded;
  ROCm reports `"cuda"` → included). Inference output is cast back to float32 for PCM, so the HTTP
  contract is unchanged. **A blanket `model.half()` crashes** — coqui feeds the reference-audio
  front-end raw float32 waveform/mel that it never casts, so `get_conditioning_latents` dies with
  `Input type (torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor) should be the same`
  (speaker encoder first, then the GPT style encoder), and the engine just rebuilds the worker on
  every request. `_apply_fp16(model)` therefore halves the model and restores the modules in
  `FP16_FLOAT32_MODULES` (`hifigan_decoder.speaker_encoder`, `gpt.conditioning_encoder`,
  `gpt.conditioning_perceiver`) to float32 — missing paths are skipped, so a coqui rename degrades
  instead of failing at load. They are tiny and run once per sample (latents cached), so the fp16
  win is untouched; `_conditioning()` casts the resulting latents `.half()` before they meet the
  half GPT/vocoder. **Second, disjoint fp16 trap:** coqui replaces the GPT2 position embedding with
  `functools.partial(null_position_embeddings, …)` (`TTS/tts/layers/tortoise/autoregressive.py`),
  which returns `torch.zeros(...)` with **no dtype** — always float32 — and, being a plain callable
  rather than an `nn.Module`, is invisible to `.half()`. `inputs_embeds(half) + position_embeds(f32)`
  promotes the hidden state back to float32 → `expected scalar type Float but found Half` in every
  LayerNorm. `_patch_callable_embeddings` wraps such callables (`FP16_CALLABLE_EMBEDDINGS`, currently
  `gpt.gpt.wpe`) in `_HalfOutput`; anything that *is* a Module is left alone (already halved, and
  replacing it would unregister it). Both traps only ever showed up after the bootstrap started
  forwarding the whole config (before that, fp16 was silently never on). **Live-verified 2026-08-02**
  on the RTX 5090 against the user's custom XTTS model: streaming + one-shot produce audio, no NaN,
  same level (peak 0.62 vs 0.67, rms 0.1264 vs 0.1269), rtf 0.633 (fp32) → 0.448 (fp16), ~1.4×. `engine.health()` reports the effective `xtts_fp16` (backend==xtts AND flag
  AND device==cuda).
- `src/audio_compat.py` — `ensure_native_audio_loading()`: replaces `torchaudio.load` with a
  soundfile-based implementation **iff** torchaudio ≥ 2.9 AND torchcodec is absent (`_needs_shim` is
  pure → unit-testable without torch). Exists for exactly one backend: AMD's native-Windows ROCm
  build ships torch/torchaudio **2.9.1** and nothing older, and from 2.9 `load` is an alias for
  `load_with_torchcodec` → system FFmpeg. Patching the attribute (rather than the call sites) is
  deliberate: both engines call `torchaudio.load` from inside third-party code
  (`f5_tts.utils_infer`, coqui XTTS) that we do not fork, and attribute lookup happens at call time,
  so one patch reaches all of them. Called at the top of `Engine.start()`, before any worker — and
  therefore any engine library — is imported. Idempotent (flag on the function).
- `src/ndjson.py` — the ONLY stdout writer; events: starting/progress/ready/log/error/shutdown.
  Every event is stamped with a UTC ISO-8601 `ts` (millisecond precision) in `_write` (additive,
  `setdefault` so an explicit `ts` wins) → the host can time when things happen.
  **`log_once(msg, level)`** dedupes per process (keyed by level+message, `reset_log_once()` for
  tests) — for lines stating a *fact about this process* ("Nutze eigenes XTTS-Modell", "XTTS fp16
  enabled on …"). Those sit in the model resolvers, which run **once per worker** (pool size =
  `max_workers_hint`, VRAM-derived), so a plain `log` printed the same line n times and looked like
  a bug. Deduping happens at the call site, not in `_write` — real repeated events stay repeated.
- `src/progress.py` — per-file download progress. `patched_downloads(emit)` swaps the `tqdm` class
  (on `tqdm`/`tqdm.auto`/`tqdm.std`) for a subclass that turns byte-download bars into throttled
  `ndjson.progress(percent=…)` events; `ModelProgress` wraps it for the bootstrap "model" step
  (reads `F5W_STEP_INDEX`/`F5W_STEP_TOTAL`, defaults 5/6). Both HF (F5 checkpoints, `unit="B"`,
  desc=filename) and coqui (XTTS, `unit="iB"`, no desc → label "Download") route through it because
  they bind their `tqdm` reference on first import and the wrapper imports them lazily. tqdm import
  is lazy so the module (and the tqdm-less test venv) imports fine; absent tqdm → silent no-op.
- `src/procutil.py` — subprocess helper with `CREATE_NO_WINDOW` (no console flashes, SPEC §8.5).
- `src/streaming.py` — `chunk_text()` sentence chunking (pure, model-free).
- `src/samples.py` — `SampleService`: basename-only resolution + traversal guard. A voice is EITHER a
  single audio file OR a **voice folder** (`samples/<stem>/`, ONE level, not recursive) holding
  several clips of the same voice; `resolve_path()` picks one clip **at random per request** (variance)
  via an injectable `chooser` (default `random.choice`, tests pass a stub/seeded `Random`). The request
  **extension is ignored** — only the stem matters (`_stem()` strips a trailing audio ext, else keeps
  the whole name), so `X`, `X.wav`, `X.mp3` resolve identically; a folder **shadows** a same-stem file.
  Single-file lookup tries the configured exts in priority order (`_ext_priority()`); folder files come
  from `_audio_files_in()` (sorted, symlink-escape-guarded, one level). `list_samples()` lists folders
  by name + single files with ext (folder shadows file), sorted, details add `count` (clips), `bytes`
  (sum) and `has_ref_text` (any clip has a sidecar). Non-audio/unknown stems now → 404 `SampleNotFound`
  (was 400). Ref-text (per CHOSEN clip): sidecar `<clip>.txt` → request → injected ASR transcriber
  (cached, keyed by full path so same-named clips in different folders don't collide). `engine.stream`
  logs the chosen `file=<name>` in the `tts request start:` line.
- `src/gpu_detect.py` — `detect_backend()`: NVIDIA(cu128≥sm12 else cu126)→AMD(rocm/dml)→Intel(xpu)→cpu;
  VRAM→`max_workers_hint`. Wheel URLs centralized in `TORCH_INDEX`. **`config.gpu_backend`
  (default `auto`) short-circuits the whole chain** via `FORCED_BACKENDS` (data, not branches) —
  needed wherever the probes cannot see the truth: a slim ROCm container has no `rocminfo` and no
  `/opt/rocm`, so `auto` answers CPU and the GPU sits idle; the inverse is a CPU-only build handed
  `--gpus all`, where the injected `nvidia-smi` selects a CUDA device that torch cannot serve. An
  unknown value **raises** rather than falling back — a misspelled `rocm` answered with a CPU pool
  just looks like the wrapper being slow. Forced detections carry no `free_vram_gb` (that probe is
  what the caller opted out of), so the worker hint falls to its conservative default; set
  `max_workers` if you want more. `torch_index_override` still applies on the forced path.
  **Native-Windows ROCm (`rocm_win`, added 2026-08-07):** `_detect_amd(config)` matches the Windows
  display-adapter NAME against `config.rocm_windows["gpu_pattern"]` (AMD supports only Radeon 9000 +
  select 7000); a match yields a Detection carrying `wheel_urls` / `torch_wheel_urls` /
  `python_version` / `torch_version` from config, a non-match keeps the old DirectML→CPU path.
  Name-matching is a heuristic, chosen because the alternative — installing a multi-GB ROCm stack on
  every AMD machine and finding out at first inference — is worse; the engine compensates by
  treating `rocm_win` as **fragile** (worker build wrapped in try/except AND self-tested, rebuild on
  CPU with a warning, never a dead server). All four URL/version fields live in `config.json` under
  `rocm_windows`, so an AMD release bump is a data edit. All seven URLs verified 200 on 2026-08-07.
- **Ein Marker heißt „fertig installiert", NICHT „funktioniert noch" (2026-09-07).** `step_deps`
  ruft bei gesetztem `deps.done` erst `_existing_venv_problem` → `_verify_venv` (torch, transformers,
  **f5-tts-Import**) gegen das VORHANDENE venv; nur ein sauberes Ergebnis überspringt den
  Schritt, sonst wird der Marker gelöscht und neu gebaut. Anlass waren zwei Live-Meldungen von
  Installationen, die „erfolgreich" meldeten und danach starben: der Marker stammte aus einer älteren
  Wrapper-Version, also übersprang jeder Neulauf genau den Schritt, der die Reparatur gewesen wäre.
  `_existing_venv_problem` wirft nie — jeder Fehlschlag dort bedeutet „neu bauen", und eine
  abgebrochene Sonde darf nicht den Bootstrap verhindern, der repariert hätte.
- **`config.datasets_constraint` (`datasets>=3.0`) gehört in DIESELBE Auflösung wie der
  transformers-Pin.** f5-tts lässt `datasets` unbeschränkt; alles vor 2.16 erbt von
  `pyarrow.PyExtensionType`, das pyarrow entfernt hat → der Install gelingt und der erste
  `import f5_tts.api` stirbt mit einem AttributeError, dessen Traceback weder das Paket noch den
  Wrapper nennt (live gemeldet). `_verify_f5` importiert genau diesen Pfad vor `deps.done` und fängt
  damit auch die nächste kaputte Auflösung, nicht nur diese.
- **DirectML: das ganze venv folgt `torch-directml`, nicht umgekehrt (2026-09-07).**
  `torch-directml` deklariert hart `torch==2.4.1` und ist seit 2024-09-14 (eine `dev`-Version) nicht
  mehr erschienen; als bloßes Extra installiert hat es torch heruntergezogen, worauf `_verify_torch`
  die Installation **in jeder veröffentlichten Version auf jeder AMD-Windows-Maschine** scheitern
  ließ — DirectML hat also nie funktioniert. `gpu_detect.DML_TORCH_VERSION`/`DML_TORCHAUDIO_VERSION`
  setzen jetzt `Detection.torch_version`/`torchaudio_version`, der Torch-Pin reist mit dem Extra
  (`--extra-index-url`, NICHT `--index-url`: das Extra selbst liegt auf PyPI), und
  **Auf echter Hardware ungetestet** (keine AMD-Karte hier); offen ist besonders, ob
  coqui/numpy auf torch 2.4.1 sauber laufen. Recherche dazu: es gibt **keinen Ersatz** — `torchruntime`
  (gepflegt, Release 2026-09-03) wählt für jede AMD-Karte unter Windows weiterhin DirectML, Microsofts
  Nachfolger ist ONNX-Runtime (kein PyTorch-Backend), ZLUDA ist Alpha und wieder Hobbyprojekt.
- **`bootstrap.step_deps` has two install shapes now:** index-based (`torch==<pin> --index-url …`,
  every other backend) and **wheel-URL-based** (`det.wheel_urls` → ROCm runtime first, then
  `det.torch_wheel_urls`). The venv is created with `det.python_version or config.python_version`
  (AMD's wheels are cp312-only; uv fetches the interpreter itself), and `_verify_torch` takes an
  `expected_version` so the 2.9.1 build is not rejected against the 2.7.0 pin. The re-pin after the
  engine install happens on BOTH shapes — f5-tts/coqui can otherwise pull a stock torch over AMD's
  build. torchcodec stays uninstalled everywhere (`audio_compat` covers 2.9's `load`).
- **Worker pool sizing (`max_workers`, default changed to 1 on 2026-08-09).** A worker is one loaded
  model instance on the device; `stream()` takes exactly one out of `_free` for the whole request, so
  extra workers buy **concurrency, not speed** — and each costs a full copy of the weights in VRAM.
  Default is therefore `1`; `null` restores the old auto behaviour (VRAM-derived, cap 4). The value
  is a **ceiling**: `_apply_worker_hint` still lowers it when free VRAM says so.
- `src/jobs.py` — `JobRegistry`/`Job`: cancel events, progress, thread-safe.
- `src/engine.py` — `Engine`: worker pool (model loaded once), `admit()` (503 backpressure) +
  `stream()` (per-chunk PCM, cancel checks, worker rebuild on crash). `worker_factory` injectable
  for tests; `_default_factory` picks `F5TTSWorker` vs `XTTSWorker` by `config.tts_backend` (XTTS
  imported lazily). Per sentence chunk, `stream()` branches on `worker.supports_streaming`: XTTS uses
  `_stream_chunk()` (token streaming — pumps the worker's sync `infer_stream` generator one item at a
  time via `_pump_next` in the executor, yields PCM parts as produced, cancel-checked between parts);
  F5 uses one-shot `_infer_chunk()`. `float_to_pcm16()` helper. dml/xpu self-test → CPU fallback.
  `health()` reports `tts_backend`. **Per-request timing:** `stream()` logs a `tts request start:`
  line (job/sample/lang/chars/chunks) up front and, on DONE, calls `_log_timing()` → a `tts request
  done:` line with `generated=<s>` (wall clock, `time.monotonic`), `audio=<s>` (from total PCM bytes
  ÷ 2 ÷ sample_rate — s16 mono), `rtf=` (generated ÷ audio; <1.0 = faster than real time), plus
  **`first=<s>`** (time until the FIRST PCM left the engine, `n/a` if none) and **`parts=<n>`**
  (pieces the response was delivered in). `first` ≪ `generated` with `parts` > 1 proves the wrapper
  streamed — so a consumer that only starts playing at the end is buffering on its own side. Both
  yield paths run through ONE consume loop in `stream()`: `_stream_chunk()` (token streaming) and
  `_one_shot_chunk()` (thin async-iterator adapter over `_infer_chunk`) so the byte/part/first
  bookkeeping and the cancel check exist exactly once.
- `src/server.py` — `create_app(config, engine=None)`: lifespan builds+starts engine, emits `ready`,
  parent-PID watchdog. Endpoints: `/tts` (StreamingResponse), `/samples`, `/languages`, `/cancel/{id}`,
  `/jobs/{id}`, `/health`, `/shutdown`, `/` (web UI). Optional Bearer `api_key`.
  - **`TtsRequest.format`** (`"pcm"` default · `"wav"`): wav **buffers the whole clip** and prepends
    a RIFF header (`src/wav.py`), because a browser cannot play headerless PCM. Streaming is given
    up knowingly in that mode — a correct header needs the total length, and the alternative (a
    header claiming an unknown size) is honoured inconsistently across browsers. Validation,
    admission, job and cancel are identical on both paths; an unknown format is a 400.
  - **`/languages`** answers `{active, options, locked, reason}`. It exists so the UI does not have
    to re-implement the backend rule: F5 loads one finetune per process (→ `locked: true`, options =
    just the active language), XTTS is multilingual in one model (→ all `XTTS_LANGUAGES`).
  - **`/` serves `src/static/index.html` and is the ONE endpoint not behind the API key** — you have
    to be able to load the page in order to type the key into it; everything the page then calls is
    protected as usual. Missing file → 404 JSON, never a crash.
- `src/static/index.html` — the built-in test page: one self-contained file, no build step, no CDN
  (must work offline and in the container). Requests `format:"wav"` and plays the blob in an
  `<audio>` element; revokes the previous blob URL on each run or every generation leaks one. Key is
  kept in `localStorage`. Ships in the image via `COPY wrapper/src ./src`.
- `src/ratelimit.py` — `RateLimiter` (sliding one-hour window; `rate_limit_per_hour` global +
  `rate_limit_per_ip_per_hour`, both 0 = off) + `client_address()`. **Not back-pressure:**
  `engine.admit()` answers 503 "busy now", this answers **429** "your share for this hour" — and it
  is checked BEFORE `admit()` so a rejected caller never occupies queue space (test asserts
  `_pending` is unchanged). Only `/tts` is limited; `/samples`, `/languages`, `/health` and the UI
  stay open or the page locks itself out. Four decisions that are easy to get wrong: **sliding**
  window (fixed buckets let a caller spend two quotas across the boundary); **only accepted requests
  are counted** (counting rejections lets a hammering client extend its own lockout forever); both
  counters are charged **together or not at all** (else the per-IP window drains while the global
  limit does the rejecting); `X-Forwarded-For` only with `trust_forwarded_for` (otherwise any caller
  invents an address per request and the per-IP limit is worthless). Memory: entries are pruned on
  their own check, plus a **periodic sweep** (`SWEEP_EVERY`, injectable) for clients that go quiet —
  they are never checked again, so without it the table aged only by hitting `MAX_TRACKED_CLIENTS`.
  The clock is `time.monotonic` (an NTP/DST jump must not hand out free quota) and injectable, so a
  one-hour window is tested in microseconds.
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
  no voices is recoverable by dropping in a wav, a wrapper that refuses to start is not. Called from
  `bootstrap.step_model` (non-fatal, after the weights) and from `docker/entrypoint.sh`, mirroring
  how `models.py` serves both paths. Entry point `python -m src.voicepack`. **Live-verified
  2026-08-07:** 523 files (260 wav + 262 txt sidecars + 1 csv, flat, no top-level folder) in ~5 s,
  `SampleService` then lists 260 voices; rerun skips; no `.part` left behind.
- `src/wav.py` — `wav_header()` / `wrap_pcm()`: the 44-byte canonical RIFF header, hand-written and
  unit-tested against stdlib `wave` (a parser accepting it is the actual contract).
- `bootstrap/bootstrap.py` — idempotent 6-step install (markers in `.state/`); see below.
  **`_server_env` forwards the WHOLE resolved config** to the server subprocess as `F5W_*`
  (`_config_env` + `_env_value`, the inverse of `config._coerce`). The server is a subprocess that
  re-runs `load_config` from scratch, so a hand-written list of forwarded fields silently drops every
  other CLI flag — that bug cost a live install its fp16 (`--xtts-fp16 true` reached the bootstrap,
  the server read `xtts_fp16: false` from config.json and `/health` honestly reported it off).
  `F5W_PARENT_PID` is set AFTER the bulk pass (it falls back to the bootstrap's own pid; the bulk
  pass would write an empty string). Round-trip-tested in `tests/test_bootstrap_env.py` — including a
  blanket "every field survives" check, so a new config field can't fall out of the handover.
- `bootstrap/install_win.ps1` / `install_linux.sh` — thin starters that fetch `uv` then run bootstrap.

## Release-Asset bauen (`build-release-zip.py`, Repo-Root)
- `python build-release-zip.py` schreibt **`wrapper/EchokrauTTS.zip`** (gitignored); Release danach
  von Hand auf GitHub anlegen und die Datei hochladen. `--list` zeigt nur, was hineinkäme.
- **Layout ist Vorgabe des C#-Hosts, keine Wahl:** er lädt das Asset, entpackt es nach
  `<installRoot>/echokrautts` und startet dort `bootstrap/bootstrap.py`. Im Archiv liegt deshalb der
  INHALT von `wrapper/` in der Wurzel, ohne `wrapper/`-Präfix — verifiziert gegen das echte
  0.0.0.4-Asset (identische Top-Level-Struktur, keine Datei verloren).
- **Was hineinkommt, entscheidet git**, nie eine Handliste: `git ls-files --cached --others
  --exclude-standard`. Bewusst NICHT nur `--cached` — ein Release, direkt nach dem Schreiben einer
  neuen Datei gebaut, hätte sonst genau den Code nicht drin, für den es gemacht wurde. Umgekehrt
  fallen `.venv`, `.state`, `models`, geladene `samples`, `__pycache__` und das Archiv selbst
  automatisch raus, weil `.gitignore` sie schon kennt.
- **`__pycache__` aus dem 0.0.0.4-Asset ist bewusst weg** (dort ~180 KB von 226 KB): Bytecode ist
  veraltet, sobald sich eine Quelle ändert, und auf einer anderen Python-Version toter Ballast.
  Neues Archiv: 47 Dateien, 107 KB.
- **Reproduzierbar**: feste Zeitstempel + sortierte Reihenfolge ⇒ zwei Bauläufe derselben Quellen
  sind bytegleich, sonst lässt sich „hat sich das Paket wirklich geändert?" nicht durch einen
  Dateivergleich beantworten. Geschrieben wird über eine `.part`-Datei (wie beim Voice-Pack), damit
  ein Abbruch kein halbes Archiv hinterlässt, das fertig aussieht.

## Docs & licenses (repo ROOT, not `wrapper/`)
- `README.md` — user-facing docs (backends, **Docker**, HTTP API, languages, licensing). The Docker
  section is the deployment reference: images/tags, both volumes and what breaks without them, the
  `F5W_*` table with image defaults, the two container-only vars (`ECHOKRAUTTS_SKIP_DOWNLOAD`,
  `UVICORN_LOG_LEVEL`), a compose-free `docker run`, and the two build workflows. Two traps are
  spelled out there because they are non-obvious: **`--gpus` must not be passed to the `-cpu` image**
  (detection keys off the injected `nvidia-smi` and would pick a CUDA device the CPU torch cannot
  serve), and `F5W_MAX_WORKERS` is an **upper cap** on the VRAM-derived pool size, not a fixed count.
  Note `config.log_level` is dead config — nothing reads it; uvicorn's level comes from the
  entrypoint's `UVICORN_LOG_LEVEL`. Lives at repo root;
  paths inside are root-relative (e.g. `wrapper/bootstrap/…`, `wrapper/config.json`). The old
  `wrapper/README.md` was removed — there is only ONE README, at the root.
- `LICENSE.md` — **AGPL-3.0**, wortgleiche Kopie aus `Dalamud/Echokraut`. Kam erst mit `bfe825e`
  (2026-09-07) dazu: bis dahin lag KEIN Lizenztext im Wurzelverzeichnis, obwohl README und
  `THIRD_PARTY_LICENSES.md` den Code seit jeher als AGPL-3.0 führen — also formal ‚alle Rechte
  vorbehalten‘, trotz öffentlichem GitHub-Mirror.
- `THIRD_PARTY_LICENSES.md` + `licenses/` — license manifest for everything the bootstrap installs
  (code deps + model weights) and vendored full-text licenses. Also at repo ROOT (moved out of
  `wrapper/` on user request). Key point: **all model weights are non-commercial** (F5 base +
  finetunes CC-BY-NC-4.0; XTTS-v2 CPML). Code deps are permissive (f5-tts MIT, coqui-tts MPL-2.0,
  torch/uvicorn/soundfile/numpy BSD, fastapi/pydantic MIT, uv Apache/MIT). CPML text is vendored
  (`licenses/XTTS-v2-CPML.txt`) since its host `coqui.ai` is offline. All git-tracked (NOT ignored).

## Multi-language (F5: one language per process · XTTS: per-request)
- `--language <code>` at startup selects which model the worker pool loads; install downloads all 4
  (en/de/fr/ja) so switching = restart only. Bootstrap forwards the active language/api_key to the
  server subprocess via `F5W_LANGUAGE`/`F5W_API_KEY` env (server re-runs `load_config`).
- Verified finetune repos (F5-TTS SHARED.md, all CC-BY-NC): DE `hvoss-techfak/F5-TTS-German`
  (`model_f5tts_german.safetensors`+`vocab.txt`, arch `F5TTS_Base`); FR
  `RASPIAUDIO/F5-French-MixedSpeakers-reduced` (`model_last_reduced.pt`); JA `Jmica/F5TTS`
  (`JA_21999120/model_21999120.pt`+`JA_21999120/vocab_japanese.txt`). EN = base, auto-downloaded.
- **Sprachauflösung ist ab 2026-08-11 generisch:** `server._request_languages(config)` liefert die
  Sprachmenge des AKTIVEN Backends oder `None` (gesperrt = F5); `_resolve_language` und `/languages`
  benutzen beide nur noch diese eine Funktion, statt XTTS als Sonderfall zu verzweigen. Ein viertes
  mehrsprachiges Backend ist damit ein Eintrag, keine zweite Verzweigung. `BACKEND_NAMES` liefert den
  Namen für die 400-Meldung.
- **Per-request `language` is backend-aware** (`server._resolve_language`): F5 loads one finetune per
  process → a `language` field is **ignored** (never rejected), the loaded/startup model is used
  regardless; XTTS is multilingual in one model → any code in `xtts_backend.XTTS_LANGUAGES`
  (en/es/fr/de/it/pt/pl/tr/ru/nl/cs/ar/zh-cn/ja/hu/ko/hi) is honored per request (no reload), an
  unsupported code is a clean 400 (not a deep 500). Omitting `language` always falls back to the
  active/startup language. The resolved language flows through `TtsParams.language` →
  `worker.infer(..., language)`/`infer_stream(..., language)` → XTTS's `model.inference(text, lang,
  ...)`; F5's worker accepts it for protocol parity but ignores it (and for F5 the resolved value is
  always the startup language anyway). `/health` reports the active/startup language.
- F5TTS finetunes load via `F5TTS(model=arch, ckpt_file=<local>, vocab_file=<local>, ...)`.
- Live-verified: German model downloads, loads (`F5TTS_Base`), synthesizes German (7.9s clip).
- **NOTE for this machine:** the German model is already in `models/` cache — restart the server
  (language defaults to `de`) and it serves German. For a full 4-language install, delete
  `.state/model.done` to re-trigger `download_all`.

## Chatterbox: entfernt am 2026-09-07 (nicht vergessen, sondern verworfen)
- Das dritte Backend (Resemble Chatterbox Multilingual) ist **komplett aus dem Repo geflogen** —
  Code, Tests, Starter, Config-Felder, Docker-Schritte, README- und Lizenz-Einträge. Falls es je
  zurücksoll: `git log --all -- wrapper/src/chatterbox_backend.py`.
- **Warum, gemessen auf der RTX 5090:** rtf 0,97 gegen XTTS fp16 0,44 und F5 0,32 (nfe 32) — also
  rund doppelt so langsam wie die langsamere der beiden verbliebenen Engines; dazu der größte
  VRAM-Bedarf (3,22 GB Gewichte / 3,73 GB Peak gegen 0,74/2,09 bei F5) und **kein Streaming**
  (`parts=1`, `first` = `generated`). Der Grund war architektonisch und nicht behebbar: die
  T3-Token-Schleife ist submit-latenz-gebunden (~1100 Kernel-Starts je Token bei 31 µs Enqueue),
  weshalb fp16, TF32 und torch.compile allesamt nichts brachten (torch.compile war 15× LANGSAMER).
- **Was der Wegfall an Komplexität spart:** die `--no-deps`-Installation mit kuratierter Paketliste,
  der `setuptools<81`-Zwang (sonst degradiert `resemble-perth` still zu `None`), das Aufräumen
  geleakter Attention-Hooks je Anfrage, `pykakasi` als einziges GPL-Paket im Baum, und die
  Sonderfälle in `_request_languages`/`BACKEND_NAMES` (Chinesisch `zh` vs. XTTS `zh-cn`).

## Multi-backend (install all engines, select one at start)
- **Install-all / select-at-start:** the bootstrap installs BOTH engines and ALL their weights
  into the single `.venv`/`models`. `--tts-backend <f5|xtts>` (config `tts_backend`, default `f5`) only
  selects which engine the worker pool loads at startup — **switching is a restart, never a
  reinstall.** `_default_factory` in `engine.py` picks `F5TTSWorker` vs `XTTSWorker` (XTTS imported
  lazily). Bootstrap forwards the choice via `F5W_TTS_BACKEND`; `/health` reports `tts_backend`.
- **XTTS-v2 (Coqui)** — natively multilingual (en/de/fr/ja), clones from the reference sample
  **without a transcript** (so `ref_text`/ASR is unused; `XTTSWorker.transcribe` returns `""`).
  Output SR 24000 (same as F5). Uses the maintained community fork **`coqui-tts`** (idiap); the
  original `TTS` package is unmaintained. Weights under **CPML (non-commercial)** — accepted
  non-interactively via `COQUI_TOS_AGREED=1`; kept under `models/` (`TTS_HOME`). API: load `Xtts`
  from `XttsConfig` + `load_checkpoint`, `get_conditioning_latents(audio_path=[sample])` (cached per
  sample), `model.inference(text, lang, gpt_cond_latent, speaker_embedding, speed=…)` → `{"wav"}`.
  `lang` comes from the per-request `language` (see Multi-language above) — one loaded model serves
  every XTTS language, so switching languages costs nothing (no reload).
- **Token streaming (XTTS only):** `XTTSWorker.infer_stream()` wraps `model.inference_stream(...,
  stream_chunk_size=config.stream_chunk_size)` (default 20), yielding flat float32 chunks (torch
  tensors → `.detach().to("cpu").numpy()`). `supports_streaming = True` flags it; F5 sets `False`
  (its `infer()` returns a whole clip). The HTTP contract is unchanged — just smaller, more frequent
  PCM parts. **Live-verified (2026-07-01):** a 9 s German clip streamed as 10 chunks with
  first-audio at ~2.4 s vs ~5.9 s total.
- **Deps install (`step_deps`):** one `uv pip install <wrapper_root> coqui-tts <transformers_pin>` —
  a SINGLE resolution so uv finds one mutually-compatible set for f5-tts + coqui-tts (or fails
  loudly), instead of two sequential installs stomping shared deps. The torch re-pin +
  torchcodec-uninstall + `_verify_torch` + `_verify_transformers` guards run afterwards, so any torch
  or transformers drift fails loudly instead of freezing a broken venv. `step_model` downloads F5
  (all langs) then XTTS-v2.
- **transformers MUST be pinned `<5` (`config.transformers_constraint`, default
  `"transformers>=4.57,<5"`).** coqui-tts imports `transformers.pytorch_utils.isin_mps_friendly`,
  which transformers **removed in 5.x** (idiap issue #558). coqui-tts declares only
  `transformers>=4.57` with no upper bound, so an unconstrained resolve picks 5.x → XTTS installs
  fine but crashes at model-load with `cannot import name 'isin_mps_friendly'`. **4.57.x is the last
  4.x line** — it still has the symbol AND satisfies `>=4.57`; f5-tts declares no transformers bound
  so it accepts 4.57.x too. `_verify_transformers` imports the symbol before writing `deps.done`.
- **Live-verified (2026-07-01)** on an NVIDIA cu128 GPU: f5-tts + coqui-tts co-resolve in one venv
  against pinned torch 2.7.0+cu128; all 4 F5 models load; XTTS-v2 downloads, loads (18.5 s), and
  **synthesizes German** (15.3 s audio in 8.4 s, float32 24 kHz). The only blocker was the
  transformers 5.x issue above, now pinned.
- **Migrating an existing F5-only install:** `.state/deps.done` + `model.done` predate the two-engine
  install, so coqui-tts + XTTS-v2 are NOT present yet. Delete both markers and re-run `start.bat` to
  install both engines (rebuilds the venv via `uv venv --clear` — a one-time cost; F5 reinstalled
  too). Then `start.bat --tts-backend xtts` serves XTTS. **On the AllTalk checkout
  (`C:\alltalk_tts\echokrautts`) the venv was hot-patched** (`transformers==4.57.6`) after its first
  install hit the 5.x bug, so XTTS already works there without a full rebuild.

## Download progress (per-file bars)
- The bootstrap "model" step (5/6) downloads all F5 checkpoints + XTTS-v2 as sub-processes whose
  stdout NDJSON is forwarded live. `src/progress.py` gives each individual file its own animated bar
  by hooking `tqdm` (see project layout). **Ordering is load-bearing:** huggingface_hub/coqui bind
  their `tqdm` reference at *import* time, so `patched_downloads()` must wrap the FIRST import of
  those libs — which works only because `download_all`/`download_model` import them lazily *inside*
  the `with mp.patch():` block. Patch ONCE around the whole `download_all` loop (not per language):
  HF's `class tqdm(old_tqdm)` is defined once on first import, so re-patching per language would bind
  all later languages to the first one's label — the per-language label instead comes from
  `ModelProgress.stage()` mutating shared state the subclass reads at emit time.
- **Force-enable on non-tty:** coqui creates bars with tqdm's default `disable=None`, which
  auto-disables on the piped (non-tty) bootstrap stdout → `n` never advances → no progress. The
  subclass forces `disable=False` when it's `None`, but respects an explicit `True/False` (HF's
  globally-disabled case). Only *byte* bars (`unit` contains `b`) are reported; iteration/file-count
  bars are ignored. Reporting is throttled to ~3% (+ a final 100%) and never re-emits a lower/equal
  percent. A failing `emit` is swallowed so progress can never break a download.
- **Live-verified (2026-07-03)** against the real huggingface_hub AND coqui in the venv: HF's
  `class tqdm(old_tqdm)` and coqui's `from tqdm import tqdm` both end up with `_NdjsonTqdm` in their
  MRO, emit 0→100 per file, and the original tqdm is restored on context exit. Unit tests
  (`test_progress.py`, 11 tests) drive the subclass against a fake base so no real tqdm is needed.

## Key decisions & gotchas (learned, non-obvious)
- **torch pinned to 2.7.0 (configurable `torch_version`/`torchaudio_version`).** Newer torchaudio
  routes `torchaudio.load` through **torchcodec → needs system FFmpeg**. 2.7.x uses the bundled
  **soundfile** backend → no external binaries. f5-tts *declares* torchcodec but only calls
  `torchaudio.load` (`utils_infer.py:403`), so bootstrap re-pins torch after `pip install .` and
  **uninstalls the unused torchcodec** (it would otherwise drag torch back to 2.11+cu128).
  `step_deps` now **verifies** (`_verify_torch`) torch==config + torchcodec absent BEFORE writing
  `deps.done` — a bad resolution raises `FatalError` (no marker) so the next run rebuilds, instead
  of freezing a venv that crashes on every infer with `Could not load libtorchcodec`.
- **`uv run` MUST use `--no-project` in the install_* starters.** The starters call
  `uv run python bootstrap.py` and their cwd is `wrapper/`, which has a `pyproject.toml`. Without
  `--no-project`, `uv run` treats it as a project and **auto-creates + syncs `.venv` from pyproject**
  (f5-tts → torch **2.12.1+cpu** + torchcodec from PyPI) *before bootstrap.py even starts* — silently
  clobbering the pinned 2.7.0+cu128 torch that `step_deps` installs. This was the actual cause of the
  "stale venv" / `Could not load libtorchcodec` infer crash. `bootstrap.py` owns `.venv`; the starters
  only provide a Python to run it. `step_deps` also passes `uv venv --clear` so any pre-existing venv
  is replaced rather than failing "already exists".
- **Bootstrap→server stdout:** the server is a child of bootstrap; do NOT just inherit the fd — a
  block-buffered inherited pipe **swallowed the `ready` event**. `step_serve` pipes the child stdout
  and forwards it line-by-line (keeps bootstrap as the stable parent the C# Process handle tracks).
- F5-TTS API (verified): `from f5_tts.api import F5TTS`; `infer(ref_file, ref_text, gen_text,
  nfe_step, speed, remove_silence, show_info, progress) -> (wav float32, sr, spec)`; **SR = 24000**.
- Live-verified on an NVIDIA **Blackwell (cu128)** GPU: 4 workers, `/tts` → ~9.6s real audio.
- `/tts` request bodies with umlauts via curl on Windows bash get mangled (shell codepage), causing a
  legit 400 "error parsing the body" — send a UTF-8 JSON file with `--data-binary @file` to test.

## Git
- Branch: `feature/f5-tts-wrapper`. `.gitignore` excludes `.venv/.venv-test/.uv/.state/models/samples
  contents/__pycache__/.pytest_cache`. Only 26 source/test/config files are tracked.
- Global rules still apply: no AI attribution in commits; ask before commit. **`CLAUDE.md` IST hier
  committbar** (globale Regel seit 2026-07-19 umgekehrt, Ausnahme sind nur Repos unter `Brunata/`);
  gitignored bleibt `.claude/` (Worklog + verify.ps1).
- **Remotes:** `origin` = GitLab (`gitlab.echotools.cloud/echotools/echokrautts.git`, primary).
  GitHub (`RenNagasaki/Echokrautts`) is a read-only mirror, kept additive by CI. Default branch `main`.

## CI (`.gitlab-ci.yml`)
- **Not the C# Dalamud pipeline** — no SonarQube/.NET here. Two stages:
  - `test` (every branch): `python:3.12-slim`, installs only dev deps (`pytest pytest-asyncio
    httpx fastapi numpy`) — NOT `pip install .[dev]` (that pulls f5-tts→torch, which tests mock),
    then `cd wrapper && python -m pytest -q`.
  - `mirror-to-github` (`main` only): additive `git push github --all --force` + `--tags` to
    hardcoded `RenNagasaki/Echokrautts`. **Never `--mirror`** (deletes GitHub-only release tags).
- CI/CD vars needed: `GITHUB_USER` (PAT owner / auth identity, ≠ repo owner) + `GITHUB_PAT`.
- **Mirror is live and full-history** (verified 2026-08-07: GitHub `main` = GitLab `main`, tags
  `0.0.0.1`–`0.0.0.5` present). Unlike Echokraut, which switched to a *snapshot* mirror (orphan
  commit per push, no history on GitHub) — Echokrautts still pushes the real history. Switching to
  the snapshot form is a deliberate, separate decision; the release-triggered Docker build works
  either way (the workflow file travels in the tree, tags get repointed).

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
