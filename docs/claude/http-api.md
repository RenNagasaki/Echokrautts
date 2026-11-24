# HTTP-Schicht (`wrapper/src/server.py` und Umfeld)
Aus der Root-`CLAUDE.md` verschoben (2026-11-13).

## Module
- `src/inline_ref.py` — **Inline-Referenz (2026-11-08, Plan `echokraut/docs/plans/game-voice-reference.md`
  §2.4, Phase 3).** `/tts` nimmt statt `sample` auch `ref_audio` (base64 PCM s16le 24 kHz mono) +
  `ref_text` (Pflicht, Spieltext) + optional `ref_ids`. `build_reference` → `InlineReference`
  (`voice_id` = sha256 der Clip-IDs, ohne IDs von Audio+Text; Text und Audio NICHT im repr). **Nur RAM:**
  kein `samples/`, kein Temp-File, die `SampleService` (Sidecar, ASR-Fallback, Transkript-Cache) wird
  für Inline nie gefragt. Der Worker bekommt das OBJEKT statt `str(path)` (`engine._worker_ref`) und
  meldet `supports_inline_reference`; `Engine.supports_inline_reference` ist alles-oder-nichts über den
  Pool → `/health inline_reference`, sonst `/tts` 501. `ConditioningCache` = LRU je Worker
  (`inline_ref_cache_size`). Grenzen `inline_ref_max_seconds` (15 s; F5s Budget `22 - ref_s` würde bei
  langer Referenz negativ).
  **Je Engine geprüft (Quellen gelesen, nicht live):** ⚠ **F5 `F5TTS.infer` schreibt bei JEDER neuen
  Referenz eine Temp-WAV** (`preprocess_ref_audio_text`, f5-tts 1.1.22) → `F5TTSWorker._infer_inline`
  ruft `infer_batch_process((tensor, sr), …)` direkt und wiederholt nur Satzende + `max_chars`-Formel
  von `infer_process` (kein Stille-Trimmen: würde Audio und Text entkoppeln). **XTTS**
  `_inline_latents` baut `get_conditioning_latents` aus `get_speaker_embedding` +
  `get_gpt_cond_latents` (Tensoren, 22050 Hz, Längen 6/6 = coqui-Defaults). **MOSS ONNX**
  `_encode_inline` = Vendor-`encode_reference_audio` ohne `torchaudio.load` (eine `codec_encode`-Session).
  **MOSS PyTorch: NICHT unterstützt** — `NanoTTSService.synthesize_stream` verlangt einen existierenden
  Pfad. **Live (GPU) noch ungetestet.**
- `src/asr.py` — `whisper_transcribe(audio, sr, lang)` für **`POST /asr-check`** (Audio rein, `{text,
  seconds}` raus, nichts gespeichert, ≤ 30 s). Nutzt F5s Whisper-Pipeline mit `{"raw", "sampling_rate"}`,
  **wenn f5-tts installiert ist** (eine Whisper im VRAM), sonst eine eigene `transformers`-Pipeline mit
  gleichem Modell/Argumenten (Lazy-Install: unter XTTS/MOSS/Qwen fehlt f5-tts, 2026-11-12)
  — NIE `F5TTSWorker.transcribe` (geht über die Temp-WAV). In der Engine injizierbar (`asr=`), im
  Server per `asyncio.Lock` serialisiert (Pipeline ist global, nicht threadsicher; asyncio statt
  Thread-Lock, damit Wartende keine Executor-Threads der TTS-Worker blockieren), 503 ab `max_queue`.
  **`_quiet_libraries()`** (2026-11-14): transformers + huggingface_hub nur während des Whisper-Laufs auf
  „nur Fehler“ (`chunk_length_s is very experimental`, Logits-Processor, `clean_up_tokenization_spaces`,
  `torch_dtype`, HF_TOKEN-Hinweis), danach altes Level zurück. Qwen ruft Whisper je Stimme OHNE `.txt`
  EINMAL (Sidecar wird geschrieben); die erste solche Zeile zahlt das Laden von Whisper (live ~10 s).
- **`POST /speaker-embedding`** (2026-11-15, Plan `echokraut/docs/plans/game-voice-reference.md` §11.3):
  Audio rein (wie `/asr-check`), L2-normierter Sprechervektor raus, nichts gespeichert. Worker-Protokoll
  optional: `speaker_encoder` (Name des Vektorraums) + `speaker_embedding(audio, sr)`. **Qwen**:
  `model.model.model.extract_speaker_embedding` (24 kHz, Name `<backend>-xvector`, 0.6B und 1.7B sind
  verschiedene Räume), **XTTS**: `get_speaker_embedding` (22,05 kHz, 512 Werte, `xtts-v2-speaker`).
  `Engine.speaker_encoder` ist alles-oder-nichts über den Pool UND verlangt denselben Namen je Worker;
  sonst 501. `Engine.speaker_embedding` **leiht sich einen freien Worker** (`_free.get()`) wie ein
  `/tts` — das Modell darf nicht zwei Dinge gleichzeitig tun —, wartet also hinter Generierungen.
  Server: eigener `asyncio.Lock` (nicht der Whisper-Lock), 503 ab `max_queue`, 1–30 s, Nullvektor → 500.
  `/health`: `speaker_embedding` (bool) + `speaker_encoder`. **Messung (deutsche FFXIV-Clips):** rohe
  Vektoren liegen alle bei ~0,98 Kosinus — erst Zentrieren auf den Mittelwert vieler Clips macht sie
  brauchbar; XTTS trennt Sprecherwechsel deutlich besser als Qwen (Details im Plugin-Plan §11.8).
  Tests: `tests/test_speaker_embedding.py` (Fake-Encoder über `make_engine(speaker=…)`).
- `src/server.py` — `create_app(config, engine=None)`: lifespan builds+starts engine, emits `ready`,
  parent-PID watchdog. Endpoints: `/tts` (StreamingResponse), `/samples`, `/languages`, `/cancel/{id}`,
  `/jobs/{id}`, `/health`, `/asr-check`, `/shutdown`, `/` (web UI). Optional Bearer `api_key`.
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
  - **Schickt `speed` UND `nfe_step`** (seit 2026-09-07). Das Feld war im `/tts`-Body längst da, aber
    nur über die HTTP-API erreichbar — weshalb der Hörvergleich 32 gegen 16 nie gemacht wurde.
    `nfe_step` ist der billigste Tempo-Hebel: halbe Schrittzahl ≈ halbe GPU-Arbeit (5090, gleicher
    Satz: rtf 0,32 bei 32 gegen 0,17 bei 16). Es ist **F5-only** — XTTS klont aus dem Audio und
    ignoriert den Wert, das Feld wird dort also deaktiviert statt still wirkungslos zu sein. Die
    Statuszeile nennt den benutzten Wert neben dem rtf, damit ein A/B-Lauf sich selbst dokumentiert.
  - **`server.TtsRequest` begrenzt seither `speed` (0 < s ≤ 3) und `nfe_step` (4…128).** Beide gehen
    unverändert in die Engine, und beide sind jetzt Formularfelder: `speed = 0` ist nicht langsam
    sondern undefiniert, und ein riesiger `nfe_step` belegt einen Worker minutenlang (der Pool gibt
    genau einen je Anfrage aus) — auf einer geteilten Instanz ein billiger Stillstand.
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
- `src/wav.py` — `wav_header()` / `wrap_pcm()`: the 44-byte canonical RIFF header, hand-written and
  unit-tested against stdlib `wave` (a parser accepting it is the actual contract).
- `src/response_cache.py` — **Antwort-Cache + geteilte Generierung (2026-11-13).** `ResponseCache`
  (`response_cache_minutes` Default 5, 0 = aus inkl. Teilen; `response_cache_max_mb` 128, LRU).
  Eine Generierung = `SharedGeneration`: läuft als asyncio-Task über `eng.stream(producer_job, …)`,
  puffert alle PCM-Teile; Clients lesen per `subscribe()` erst den Puffer, dann live (Warten in
  50-ms-Scheiben, weil `/cancel` ein threading.Event setzt, das keinen asyncio-Waiter weckt).
  **Jeder Client hat einen EIGENEN Job** (`X-Job-Id`); die Generierung hat einen internen
  Producer-Job, der nie herausgegeben wird. `detach()` (finally von `subscribe`, also auch bei
  Verbindungsabbruch) zählt herunter; erst bei 0 wird der Producer-Job gecancelt. Gespeichert wird
  nur `complete` (Producer-Job `DONE`, kein Fehler); ein laufender Eintrag, dessen Producer schon
  gecancelt ist, wird bei `lookup` NICHT mehr geteilt (der Nachzügler bekäme sonst einen
  abgeschnittenen Clip). `no_cache` überspringt `lookup`, löscht den alten Eintrag, markiert eine noch
  laufende alte Generierung `storable=False` (ihre Clients bekommen sie trotzdem fertig).
  **Schlüssel** (`server._cache_key`): Backend, aufgelöste Sprache, Text, Stimme
  (`SampleService.voice_key`: Ordnername für Stimmordner, sonst Dateistamm — gleiche Stimme trifft
  trotz Zufallsclip; Inline: `voice_id` + sha256 von `ref_text`), Request-`ref_text`, `speed`,
  `nfe_step`. Prozess-Optionen gehören nicht hinein (Cache stirbt mit dem Prozess). Nur eine NEUE
  Generierung ruft `eng.admit()` (Queue-Platz); Treffer/Teilen kosten keinen Worker. Der Rate-Limiter
  zählt weiterhin jede Anfrage. Header `X-Cache` (miss/shared/hit/bypass/off), `/health
  response_cache`. Die Web-UI schickt immer `no_cache: true`. Tests: `tests/test_response_cache.py`.
