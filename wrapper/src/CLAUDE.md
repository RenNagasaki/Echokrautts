# Wrapper-Kern (`wrapper/src/`)
Aus der Root-`CLAUDE.md` verschoben (2026-11-13). Weitere Bereiche: Verweis-Tabelle in der Root-`CLAUDE.md`;
HTTP-Schicht `docs/claude/http-api.md`, Engines `docs/claude/engines.md`, MOSS `docs/claude/moss.md`.

## Project layout (`wrapper/`)
- `src/config.py` — `Config` dataclass; load order JSON < ENV (`F5W_*`) < CLI (`--kebab`). Holds
  `language` (active, startup) + `languages` map (en/de/fr/ja → arch/hf_repo/ckpt_file/vocab_file)
  + `tts_backend` (`"f5"` | `"xtts"`, one backend per process) + `xtts_fp16` (opt-in half-precision
  for XTTS, default off; only takes effect on a CUDA device). `host` default is `0.0.0.0` (binds all
  interfaces so the host reaches it with no config edit — set `api_key` or narrow to `127.0.0.1` to
  lock down). Bool ENV/CLI coercion is name-listed in `_coerce` (`asr_for_missing_ref_text`,
  `xtts_fp16`) — add new bool fields there or they stay strings.
- `src/engines.py` — **die EINE Engine-Liste** (2026-11-12, Plan `docs/plans/qwen3-und-lazy-install.md`):
  `EngineSpec(id, env, model_module, per_language)`, `KNOWN_BACKENDS` (von `engine.py` re-exportiert),
  `spec_for()` lehnt Unbekanntes mit Namensliste ab. Daten ohne torch-Import (Bootstrap liest es).
- `src/models.py` — F5 model resolution. `resolve_model(config, lang)` → `ResolvedModel(arch,
  ckpt_file, vocab_file)` via `hf_hub_download` (cached); `download_all(config)` for install;
  `python -m src.models` entry. Used by BOTH bootstrap (download) and engine (load) — single source
  of truth, no duplication.
- `src/hfcache.py` — **die EINE Stelle für den Umgang mit dem HuggingFace-Cache** (2026-09-10).
  Vorher entschied das jede Engine für sich, woraus zwangsläufig zwei leicht verschiedene Politiken
  werden. `use_models_dir(config)` **SETZT** `HF_HOME`/`HF_HUB_CACHE`/`HF_MODULES_CACHE` (kein
  setdefault — ein maschinenweiter Cache schickt den Worker sonst ein zweites Mal auf Download, im
  Container außerhalb des Volumes; zweimal live passiert) und muss VOR dem Import von transformers
  laufen. `silence_symlink_warning()` stellt `HF_HUB_DISABLE_SYMLINKS_WARNING` — der Hub kann auf
  einem normalen Windows-Konto keine Symlinks anlegen, erklärt das in zwei Absätzen und kopiert dann
  einfach, was funktioniert. Die Warnung ist also Rauschen vor einem Nicht-Problem, und Rauschen im
  Installationslog kostet Supportzeit (siehe die pydub/ffmpeg-Warnung, wegen der ein Nutzer schrieb).
  ⚠ **Der Fallback hat ein Race (Nutzer-Log 2026-11-01, `WinError 1314` beim XTTS-Download):**
  `are_symlinks_supported` schreibt VOR dem Test `True` in seinen Cache; `snapshot_download` (8
  Threads, so lädt Coqui XTTS) liest das in dem Fenster und ruft echtes `os.symlink`.
  `serialize_symlink_probe()` legt einen Lock um die Probe (ersetzt das Modul-Attribut in
  `huggingface_hub.file_download`) und MUSS vor jedem parallelen Hub-Download laufen — aufgerufen
  in `xtts_backend._resolve_model_dir`. F5 (`hf_hub_download`, einzeln) und MOSS (`local_dir`) sind
  nicht betroffen. Workaround für Nutzer ohne Fix: Windows-Entwicklermodus einschalten.
  **`silence_xet_warning()` (2026-09-10)** stellt `HF_HUB_DISABLE_XET` — aber **nur, wenn `hf_xet`
  wirklich fehlt**. Repos auf Xet-Speicher lassen den Hub JE DATEI warnen, dass `hf_xet` schneller
  waere; bei einem Modell mit Dutzenden Dateien ist das eine Wand aus rotem Text (vom User gemeldet:
  „die Warnungen nerven“). Ohne das Paket faellt der Download ohnehin auf normales HTTP zurueck — der
  Schalter aendert also nichts ausser der Ansage. **Die Bedingung ist tragend:** derselbe Schalter
  schaltet die Funktion AUS, unbedingt gesetzt haette er einem Nutzer, der `hf_xet` spaeter
  installiert, genau den Tempogewinn genommen, fuer den er es installiert hat.
  **Live belegt** an einem echten 44-MB-Download aus einem Xet-Repo: ohne Schalter 1 Warnung, mit
  Schalter 0, Datei kommt beide Male an.
  Benutzt von `models.py` (F5) und `moss_backend.py`.
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
- `src/i18n.py` — **Nutzertexte des NDJSON-Stroms in der Clientsprache (2026-11-13).** Der Installer
  zeigt jede NDJSON-Zeile im Konsolenfenster/Log, Spieler lesen sie also. EINE Tabelle `MESSAGES`
  (Schlüssel → en/de/fr/ja, ein Test erzwingt alle vier Sprachen mit gleichen Platzhaltern);
  `t(key, **kw)` → `Text` (str mit `.key`), Rückfall en bei fehlender Übersetzung/kaputtem Platzhalter
  (wirft nie). `ndjson._write` schreibt den Schlüssel als Feld `"key"` → Konsumenten matchen darauf,
  nie auf den Text. Sprache: `ui_language` > `language` > en, gesetzt in `load_config()` (jeder Prozess
  lädt zuerst seine Config — Bootstrap, Server, `python -m`-Downloads, Docker-Entrypoint). Das Plugin
  reicht die Clientsprache schon als `--language` durch (`LanguageCodeForScd`) → **keine eigene
  Plugin-/Installer-Änderung nötig**; `ui_language` ist nur für abweichende Wünsche (Standalone).
  **Bewusst englisch & NICHT in der Tabelle (Entwickler-Log):** `tts request start/done`, orphan
  protection, `backend selected`, `inference failed`/`rebuilt worker`, `XTTS fp16 enabled`,
  Pocket-Konfigzeile, audio_compat, voicepack-zip-slip, `asr-check failed`, alle
  `*verification failed`-Diagnosen (mit pip-Ausgabe). Plugin/Installer parsen keine Meldungstexte
  (nur `event`, Stand 2026-11-13). Neue Nutzermeldung = neuer Schlüssel in allen 4 Sprachen.
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
- **Worker pool sizing (`max_workers`, default changed to 1 on 2026-08-09).** A worker is one loaded
  model instance on the device; `stream()` takes exactly one out of `_free` for the whole request, so
  extra workers buy **concurrency, not speed** — and each costs a full copy of the weights in VRAM.
  Default is therefore `1`; `null` restores the old auto behaviour (VRAM-derived, cap 4). The value
  is a **ceiling**: `_apply_worker_hint` still lowers it when free VRAM says so.
- `src/jobs.py` — `JobRegistry`/`Job`: cancel events, progress, thread-safe.
- **Fatal device errors end the process (2026-11-12).** After `CUDA error: device-side assert
  triggered` (or illegal memory access / unspecified launch failure / HIP error) the CUDA context of
  the **whole process** is dead: the in-process `_rebuild_worker` always failed and every later
  request answered 500 (User-Log 2026-11-08: 543 follow-up errors). `is_fatal_device_error(exc)`
  (walks the `__cause__` chain, excludes "out of memory") decides; on a hit `stream()` does **not**
  rebuild but calls `_enter_fatal`: one `error` NDJSON event with `fatal: true`, `health()` →
  `status: "failed"` + `fatal_error`, `admit()` raises `EngineFailed` (→ HTTP 503), then after
  `FATAL_EXIT_DELAY_S` (1 s) `_hard_exit` → `os._exit(EXIT_RESTART_REQUIRED = 75)`. `os._exit`
  on purpose: a graceful uvicorn shutdown exits 0 (host can't tell crash from stop) and tearing
  down a broken CUDA context can hang. The code reaches the host unchanged (`step_serve` →
  `bootstrap.main`, `uv run`, `install_linux.sh` uses `exec`). **EchokrautLocalInstaller restarts
  on exactly 75** (`WrapperRestartPolicy`, max 3 per 15 min) — keep both constants in sync.
  Engine takes `on_fatal` (injectable; tests pass a recorder). Tests: `tests/test_fatal_device_error.py`.
  **Cause of the `Indexing.cu: srcIndex < srcSelectDimSize` assert (found 2026-11-24, not fp16):**
  a cancel inside `run_in_executor` returned at once while the thread kept computing; `finally`
  handed the worker back and the next request ran on the SAME model concurrently. Coqui stores the
  text prefix on the model (`store_prefix_emb`) and indexes the mel position embedding with
  `cache_len - prefix_len`, so the old thread got a negative / >= 608 index. Up to 0.0.1.8 a client
  disconnect cancelled the stream this way; since the response cache only `cancel_event` does.
  Fix: every executor call that holds a worker goes through `_run_to_completion` (waits for the
  call, then re-raises the cancel; `asyncio.wait` never cancels what it waits for) and
  `_stream_chunk` closes the generator in `finally`. Guard: `XTTSWorker._exclusive()` (non-blocking
  lock per step, never across a `yield`) raises `WorkerBusyError` instead of a CUDA assert.
  **Never `await loop.run_in_executor(...)` directly on a worker call.** Tests:
  `test_cancel_mid_inference_never_overlaps_the_next_request`, `test_second_call_while_computing_raises_busy`.
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
- `src/voicepack.py` — Voice-Pack-Download, nur vom Docker-Entrypoint genutzt → Doku in `docker/CLAUDE.md`.
