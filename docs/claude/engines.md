# Engines F5 / XTTS / Qwen / Pocket
Aus der Root-`CLAUDE.md` verschoben (2026-11-13). MOSS: `docs/claude/moss.md`.

## Module
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
- `src/qwen_backend.py` — **Qwen3-TTS über faster-qwen3-tts 0.5.4 (CUDA-Graphs), Engines `qwen-0.6b` /
  `qwen-1.7b`, ein Backend** (2026-11-13, Plan `docs/plans/qwen3-und-lazy-install.md`). Eigene
  venv-Gruppe `.venv-qwen` (transformers **==5.15.1**; 5.22 bricht den Tokenizer-Encoder; coqui
  braucht <5). Darf NICHTS aus den klassischen Engines importieren. `QwenWorker`: nur `device=="cuda"`
  (ROCm meldet sich so), `warmup()` im Konstruktor, Stimm-Prompt SELBST gebaut
  (`model.model.create_voice_clone_prompt(ref_audio=(array,sr), ref_text)`, ICL + 0,5 s Stille, xvec
  ohne) und als `voice_clone_prompt=` übergeben — bitgleich zum Pfad-Weg der Bibliothek, Kodieren
  0,03 s; LRU-Cache je Worker (Datei: Pfad+mtime+Text+xvec, Inline: voice_id). Kein `ref_text` →
  xvec-Rückfall + `log_once`. `transcribe()` = Whisper (`asr.py`) für den Sidecar-Fallback.
  **Option `qwen_xvec_only` entfernt (2026-11-15, User):** klang nur weniger nach der Stimme; xvec ist
  jetzt AUSSCHLIESSLICH der Rückfall ohne Transkript. Optionen bleiben `qwen_non_streaming_mode`,
  `qwen_temperature`, `qwen_chunk_size`.
  **`QwenWorker.max_chars_per_chunk = 1000`** (2026-11-15, Live-Befund): die Engine teilt Text sonst
  nach `max_chars_per_chunk` (250) — jeder Teil ist ein eigener Sampling-Lauf, und die zweite Hälfte
  einer 255-Zeichen-Zeile klang hörbar anders (gleiche Referenz). `Engine._chunk_limit()` nimmt die
  vom Worker erklärte Grenze vor der Config; XTTS/F5/MOSS/Pocket erklären keine (XTTS-Deutsch-Limit ~253).
  Gewichte per `snapshot_download(local_dir=models/qwen/<repo>)`. `EngineSpec.gpu_backends`
  (`cuda`/`rocm`/`rocm_win`): Bootstrap lehnt andere Geräte VOR dem Download ab
  (`engines.unsupported_device_reason`), die Engine baut auf fragilen Geräten NICHT auf die CPU um,
  sondern bricht ab. **`_prime()`**: eine Wegwerf-Zeile beim Start — ohne sie brauchte die ERSTE echte
  Anfrage 1,6–8 s bis zum ersten Ton (Kernel-Nachladen), mit ihr 0,44 s. **Live Windows (5090,
  `tools/windows-test.sh qwen-0.6b qwen-1.7b`, 2026-11-13):** Start bis `ready` ~11 s, warm erster Ton
  0,30 s, rtf 0,25–0,30 (0.6B) / 0,27–0,29 (1.7B); `.venv-qwen` 6,2 GB, Gewichte 2,3 + 4,5 GB.
  VRAM 4,2 GB (1.7B ~4,9), **GPU-Auslastung 93 %** (XTTS 26 %) — Spieltest offen. AMD erlaubt, ungetestet.
  **SoX-Rauschen** (`_quiet_sox()`, Kontextmanager um den MODELL-LADEVORGANG `model_factory(source)`, 2026-11-15 —
  NICHT nur um den Import: `qwen_tts` und damit `sox` lädt `faster_qwen3_tts` erst in `from_pretrained`):
  pysox ruft `os.popen('sox -h')` (Windows-cmd schreibt „Der Befehl "sox" ist …" direkt auf stderr) und
  setzt in `sox/log.py` selbst `setLevel(WARNING)` — ein vorher gesetztes Level wirkt also NICHT. Daher
  `popen` mit `2>os.devnull` nur während des Imports + `logging.Filter` (<ERROR) auf `sox`.
- `src/pocket_backend.py` — **Pocket TTS (Kyutai, CPU), Engine `pocket`** (2026-11-13). Klassische
  venv-Gruppe (braucht kein transformers; hub/tokenizers/numpy passen), `per_language` wie F5
  (`pocket_languages`: de/fr/it/es/pt → `*_24l`, en → `english`; **kein ja** → `ValueError` vor dem
  Download). Klon-Gewichte aus `pocket_repo@pocket_revision` (seit 2026-11-13 eigener Mirror
  `jkatzwinkel/pocket-tts-mirror@f78b470`, bytegleich zu `lunahr/pocket-tts-ungated@d03cd73`, sha256 je
  Datei verglichen). ⚠ Quelle = ältere Revision,
  läuft NUR mit dessen `tokenizer.model` (sentencepiece) → `write_config` schreibt je Sprache die
  Paketvorlage (`pocket_tts/config/<ordner>.yaml`) mit lokalem `weights_path` + Tokenizer nach
  `models/pocket/<rev12>/`. Stimm-Zustand (`get_state_for_audio_prompt`, auch Tensor für Inline) im LRU.
  `pocket_decode_steps` Default 4 (1 = gelegentlich Hall). **`pocket_layers` 24 (Default) / 6**
  (User 2026-11-13: 6 „hörbar schlechter“, aber rtf 0,29 statt 0,69 bei 4 Threads → beides als
  OPTION, keine zweite Engine). `pocket_languages` ist je Größe verschachtelt (`"24"`/`"6"`); fehlt
  eine Kombination (24: kein en, 6: kein fr), nimmt `language_folder` die andere Größe + Warnung.
  `decode_steps()` hebt 1 bei 6 Layern auf 4 (WER 4,6 % statt 1,8 %). Modell-Marker enthält die
  Größe UND Revision (`EngineSpec.model_variant=("pocket_layers","pocket_revision")` → `model-pocket-de-6-f78b470a26e7`), ein Wechsel lädt nach (ohne Revision überlebte ein Mirror-Wechsel den Marker und der Start starb am leeren Ordner, live 2026-11-13).
  Benchmark 2026-11-13 (`E:\TTSBench\out3`, 7 Refs × 12 Sätze): 6/1 rtf 0,21 WER 4,6 % sim 0,58 ·
  6/4 0,29 1,8 % 0,60 · 6/8 0,41 1,4 % 0,60 · 24/1 0,60 9,4 % 0,62 · 24/4 0,69 2,2 % 0,61 · 24/8 0,81
  1,4 % 0,62. **Live Windows (`tools/windows-test.sh pocket pocket-6`, 2026-11-13):** 6 Layer warm
  rtf 0,31, erster Ton 0,09 s; 24 Layer rtf 0,85, 0,28 s. `pocket-6` = Pocket mit `--pocket-layers 6`
  (Skript reicht `-Extra` an den Bootstrap durch). `_migrate_legacy_markers` kennt nur
  `LEGACY_ENGINES` (f5/xtts/moss) — sonst gälte Pocket auf Altinstallationen als vorhanden.
  **Threads immer explizit** (`cpu_threads`: `pocket_threads` oder automatisch ≤ 4): ein gestarteter
  Server bekam live nur 1 torch-Thread (rtf 0,95); 4 → 0,78, 8 → 0,75, 16 → 0,88 (torch 2.7; torch 2.15
  wäre ~15 % schneller). `_prime()` beim Start. **Live Windows (2026-11-13):** warm erster Ton 0,27 s,
  rtf ~0,86; erste Anfrage je Stimme ~1,8 s (Referenz-Kodierung, danach Cache).
  **Eigener Mirror:** `tools/pocket-mirror.py <owner>/<repo>` kopiert die Dateien bytegleich samt
  CC-BY-Namensnennung + Kyutai-Nutzungsrichtlinie und druckt die zwei Config-Zeilen.

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

## XTTS im Multi-Backend-Betrieb
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
