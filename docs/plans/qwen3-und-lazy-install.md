# Qwen3-TTS als Engine + Lazy-Install (nur aktive Engine installieren)

Stand 2026-11-12 · Status: **alle Fragen entschieden (§6); Phasen 1–4 umgesetzt (Wrapper), Phase 5 (Plugin/Installer, Spieltest) offen**

Zwei WORKLOG-Doings, bewusst zusammen geplant, weil Qwen ohne die Lazy-Install-Mechanik nicht
einbaubar ist (eigenes venv) und die Lazy-Install-Mechanik ohne eine zweite venv-Gruppe nur halb
gebaut wäre.

---

## 1. Ausgangslage (gemessen, nicht geschätzt)

**Benchmark-Umgebung** `E:\TTSBench\venv_fqwen` auf dem Gaming-PC (RTX 5090, Windows):
Python 3.12.13, `torch==2.7.0+cu128`, `torchaudio==2.7.0+cu128`, `transformers==5.15.1`,
`faster-qwen3-tts==0.5.4` (MIT), `qwen-tts-hf==0.1.1.post1`, `huggingface_hub==1.33.0`.
→ **Derselbe Torch-Pin wie der Rest des Wrappers**, nur transformers ist anders.

`faster-qwen3-tts 0.5.4` deklariert: `transformers>=5.15.1,<6`, `huggingface-hub>=1.5,<2`,
`qwen-tts-hf<0.2`, `torch>=2.5.1`. coqui-tts/XTTS braucht `transformers<5` → **ein gemeinsames venv ist
unmöglich**, nicht nur unbequem.

**Größen** (Windows, gemessen):

| Was | Größe |
|---|---|
| venv (davon torch 5,5 GB) | 6,2 GB (Qwen) / 6,6 GB (heutiges `.venv`, alle drei Engines) |
| heutiges `models/` (alle Engines, alle F5-Sprachen) | **11,6 GB** |
| Qwen3-TTS 0.6B Base | 2,34 GB |
| XTTS-v2 | 1,75 GB |
| Qwen3-TTS 1.7B Base | ~4,5 GB (geschätzt aus Parameterzahl; nicht gemessen) |

Heute belegt eine Installation also ~18 GB, egal welche Engine genutzt wird. Mit Lazy-Install:
venv 6,6 GB + Modell der aktiven Engine/Sprache (1–2,5 GB) ≈ **8–9 GB**.

**API-Befunde aus dem Quelltext von faster-qwen3-tts 0.5.4:**
- `generate_voice_clone_streaming(text, language, ref_audio, ref_text, chunk_size, temperature,
  xvec_only, non_streaming_mode, voice_clone_prompt=…)` → Generator von `(chunk, sr, info)`.
- Es gibt einen **eingebauten** `_voice_prompt_cache`, Schlüssel `(ref_audio, ref_text)` —
  unbegrenzt und pfadbasiert. Für Inline-Referenzen (Bytes im RAM) unbrauchbar.
- `voice_clone_prompt=` nimmt einen vorab erzeugten Prompt → **wir cachen selbst** (LRU) und
  umgehen den eingebauten Cache. Damit gilt eine Regel für Datei- und Inline-Referenzen.
- `_load_ref_audio_with_silence` hängt 0,5 s Stille an die Referenz (verhindert, dass das letzte
  Phonem in den Satzanfang blutet) — das müssen wir beim Selbst-Kodieren nachbauen.

**Plugin-Befund:** `TtsEngineCatalog` (Echokraut) kennt das Feld `description` bereits und
**verwirft jede Datei mit `version != 1`** (`TtsEngineCatalogService.ExpectedVersion = 1`).
→ `engines.json` bekommt `description`, **die Version bleibt 1**. Ein Hochzählen hätte das Plugin
auf die eingebettete Kopie zurückfallen lassen, ohne Qwen.

---

## 2. Feste Entscheidungen (User, 2026-11-12)

- Backend über **faster-qwen3-tts** (CUDA-Graphs). `parity_mode` nicht (7× langsamer).
- **Zwei Engine-Einträge, ein Backend:** Qwen 0.6B (`Qwen/Qwen3-TTS-12Hz-0.6B-Base`) und Qwen 1.7B
  (`Qwen/Qwen3-TTS-12Hz-1.7B-Base`, ~4,9 GB VRAM; Beschreibung nennt die Zielgruppe).
- Default `non_streaming_mode=True` (Volltext vorab; Audio streamt trotzdem: erster Ton 0,28 s,
  13 Teile je langem Satz). Optionen: `xvec_only`, Temperatur 0,7.
- ICL mit `ref_text`; Sprache als Name (`"German"`).
- Streaming ist Grundbedingung. Vertrag: float32 @ 24 kHz je Chunk.
- `voice_clone_prompt` je Stimme cachen (erste Anfrage je Stimme sonst bis 5 s).
- Wrapper installiert künftig **nur die aktive Engine + das Modell der aktiven Sprache**; ein
  Wechsel lädt nach (Nutzerbeschwerde: Ordner zu groß).

---

## 3. Entwurf

### 3.1 Engine-Register (eine Quelle für alles, was heute verstreut ist)

Neues Modul `wrapper/src/engines.py`, reine Daten, ohne Torch-Import:

```python
@dataclass(frozen=True)
class EngineSpec:
    id: str              # == --tts-backend, == engines.json id
    env: str             # venv-Gruppe: "classic" | "qwen"
    model_module: str    # "src.models" | "src.xtts_backend" | … (python -m … lädt Gewichte)
    per_language: bool   # F5: ein Modell je Sprache

ENGINES = {
  "f5":        EngineSpec("f5",        "classic", "src.models",       True),
  "xtts":      EngineSpec("xtts",      "classic", "src.xtts_backend", False),
  "moss":      EngineSpec("moss",      "classic", "src.moss_backend", False),
  "qwen-0.6b": EngineSpec("qwen-0.6b", "qwen",    "src.qwen_backend", False),
  "qwen-1.7b": EngineSpec("qwen-1.7b", "qwen",    "src.qwen_backend", False),
}
```

- `engine.KNOWN_BACKENDS` wird daraus abgeleitet (heute eine Handliste, die mit `engines.json`
  übereinstimmen muss). Ein Test liest `../engines.json` und prüft Mengengleichheit — damit ist die
  Regel „beide Listen gleich" nicht mehr nur Doku.
- Installationsfunktionen bleiben im Bootstrap (dort lebt `uv`); das Register sagt nur WAS.

### 3.2 venv-Gruppen statt eines venvs

| Gruppe | Pfad | Engines | transformers |
|---|---|---|---|
| `classic` | `.venv` (**unverändert**) | f5, xtts, moss | `>=4.57,<5` |
| `qwen` | `.venv-qwen` | qwen-0.6b, qwen-1.7b | `==5.15.1` |

Warum nicht ein venv **je Engine**: jedes venv trägt 5,5 GB torch. f5/xtts/moss vertragen sich
nachweislich in einem venv; ein Wechsel zwischen ihnen soll ein paar hundert MB kosten, nicht 6 GB.
Und das bestehende `.venv` aller Bestandsnutzer bleibt gültig → **keine Migration, kein Neudownload**.
Eine künftige Engine mit eigenem Abhängigkeitsbaum (Pocket?) wird eine dritte Gruppe, kein Umbau.

**Der Server läuft im venv der aktiven Engine.** Kein Subprozess-Worker, keine IPC: der Wrapper ist
ohnehin „eine Engine je Prozess", also wählt `step_serve` einfach den passenden Interpreter.
`src/` muss dafür in beiden venvs importierbar sein → Kern-Abhängigkeiten (fastapi, uvicorn,
pydantic, numpy, soundfile, huggingface_hub) in beide.

### 3.3 Lazy-Install im Bootstrap

- **`pyproject.toml`:** `f5-tts` wandert aus `dependencies` in ein Extra `[f5]`. Kern bleibt
  engine-frei. (Heute zieht `pip install <wrapper_root>` immer F5 mit.)
- **Marker je Gruppe und je Engine/Sprache** statt `deps.done`/`model.done`:
  - `deps-<gruppe>.done` — venv gebaut, torch/transformers verifiziert.
  - `engine-<id>.done` — Engine-Paket im venv installiert und importierbar (`_verify_f5` usw.).
  - `model-<id>.done` bzw. `model-f5-<sprache>.done` — Gewichte da.
- **`step_deps(config, det)`** → nur Gruppe der aktiven Engine; fehlt das Engine-Paket, wird es
  **in das vorhandene venv nachinstalliert** (der heutige Reparaturpfad `_install_engines`, auf eine
  Engine verkleinert). Reihenfolge bleibt tragend: Engine VOR Torch-Re-Pin, danach torchcodec raus,
  dann verifizieren, dann Marker.
- **`step_model`** → nur `spec.model_module` der aktiven Engine; F5 nur die aktive Sprache
  (`src.models` bekommt ein `--language`, `download_all` bleibt für Docker).
- **Bestandsinstallationen:** alte `deps.done`/`model.done` werden beim ersten Lauf in die neuen
  Marker übersetzt (alles, was der alte Marker bedeutete, ist ja da) — aber erst nach erfolgreicher
  Verifikation, gemäß der Regel „ein Marker heißt fertig installiert, nicht funktioniert noch".
- **`_existing_venv_problem`/`_verify_venv`** prüfen nur noch das, was die aktive Engine braucht.
  Heute würde ein venv ohne XTTS als „reparaturbedürftig" gelten — mit Lazy-Install ist das der
  Normalzustand.
- **`_stop_stale_processes`** fegt auch `.venv-qwen`.
- **GPU-Gate für Qwen:** CUDA-Graphs brauchen CUDA. Ist `det.backend` keine NVIDIA-Variante, bricht
  der Bootstrap mit klarer Meldung ab („Qwen braucht eine NVIDIA-Grafikkarte — wähle XTTS oder MOSS"),
  bevor 6 GB geladen werden. Siehe Offene Frage 3 zu ROCm.
- **Ein Engine-Wechsel ist weiterhin nur ein Neustart** mit anderem `--tts-backend`; der Bootstrap
  merkt, was fehlt, und lädt es mit den üblichen Progress-Events nach. Das Plugin muss dafür nichts
  wissen, sollte aber eine längere Startzeit erwarten (Timeout!) — siehe 3.8.

### 3.4 ASR von F5 entkoppeln

`src/asr.py` nutzt heute F5s Whisper-Pipeline (`f5_tts.infer.utils_infer.transcribe`) und begründet
das ausdrücklich damit, dass „der Bootstrap jede Engine installiert". Das stimmt nach 3.3 nicht mehr:
unter XTTS, MOSS und Qwen wäre `/asr-check` und der ASR-Fallback für fehlende `ref_text` kaputt.

→ `asr.py` baut die Pipeline selbst: `transformers.pipeline("automatic-speech-recognition",
"openai/whisper-large-v3-turbo")`, lazy, einmal je Prozess, Modell ins `models/`-Cache. Läuft unter
transformers 4.57 und 5.15. Für Qwen wird das wichtiger als für die anderen, weil ICL ohne `ref_text`
nicht geht. Whisper (~1,6 GB) lädt erst beim ersten Bedarf, nicht bei der Installation.

### 3.5 `src/qwen_backend.py` — `QwenWorker`

- `supports_streaming = True`, `supports_inline_reference = True`, `sample_rate = 24000`.
- **Laden:** `FasterQwen3TTS.from_pretrained(<lokaler Snapshot>)` + `warmup(prefill_len=100)` im
  Konstruktor (CUDA-Graph-Capture; die Ladezeit fällt beim Start an, nicht bei der ersten Zeile).
  Modell-Repo kommt aus `config.qwen_models[config.tts_backend]`.
- **`infer_stream`:** `generate_voice_clone_streaming(text, language=QWEN_LANG[code],
  voice_clone_prompt=<gecacht>, ref_text, chunk_size=config.qwen_chunk_size,
  temperature=config.qwen_temperature, non_streaming_mode=config.qwen_non_streaming_mode,
  xvec_only=…)`; jeder Chunk → flaches float32. Liefert die Engine ein `sr != 24000`, wird
  resampelt (torchaudio, wie `moss_audio`) — gemessen war 24 kHz, aber stillschweigend falsche Rate
  wäre halbe Geschwindigkeit.
- **`infer`:** zieht `infer_stream` leer (ein Erzeugungspfad, wie bei MOSS).
- **Prompt-Cache:** `inline_ref.ConditioningCache` (LRU, `inline_ref_cache_size`). Schlüssel:
  Datei-Referenz = (Pfad, mtime, ref_text, xvec_only); Inline = `voice_id` + xvec_only. Erzeugt über
  die Prompt-API des Basismodells mit `(array, sr)` und angehängten 0,5 s Stille (wie
  `_load_ref_audio_with_silence`). **Spike 2026-11-13 (0.5.4, RTX 5090):**
  `model.model.create_voice_clone_prompt(ref_audio=(array, sr), ref_text=…)` → Liste von
  `VoiceClonePromptItem`, als `voice_clone_prompt=` übergeben → **bitgleich** zum Pfad-Weg; Kodieren
  0,03 s; erster Ton 0,31 s statt 1,26 s beim ersten Pfad-Aufruf. xvec: `ref_text=""`,
  `x_vector_only_mode=True`, **ohne** angehängte Stille (die Bibliothek hängt dort keine an).
  Default-Temperatur der Bibliothek: 0,9. Beim Import meldet das Paket `sox` „SoX could not be
  found!“ auf stdout/stderr — reines Rauschen (wir laden über soundfile), im Worker unterdrücken.
- **Fehlender `ref_text`** (kein Sidecar, keiner im Request, ASR aus oder leer): Rückfall auf
  `xvec_only` für diese Anfrage + `log_once`-Warnung — statt eines 500. Klingt schlechter, ist aber
  hörbar statt stumm.
- **Sprachen:** `QWEN_LANGUAGES` = en/de/fr/ja/zh/ko/es/it/pt/ru → Namen (`"German"` …), in
  `server._request_languages` eingetragen (per Anfrage, kein Reload — wie XTTS).
- `speed`, `nfe_step`: nicht unterstützt → `warn_unsupported_speed` (aus `moss_common` in einen
  gemeinsamen Ort heben, statt es ein drittes Mal zu schreiben); nfe wird ignoriert.
- `transcribe()` → `""` (ASR macht 3.4), `self_test()` → kurzer Satz.
- Fatale CUDA-Fehler laufen über den bestehenden Exit-75-Pfad, nichts Neues.
- `python -m src.qwen_backend` → `snapshot_download` des aktiven Repos (Progress über
  `src.progress`). **Vorher prüfen:** `hfcache.serialize_symlink_probe` patcht ein Attribut von
  `huggingface_hub.file_download` — in hub 1.x muss es existieren, sonst sauber überspringen.

### 3.6 Config (`config.py` + `config.json`)

| Feld | Default | Bedeutung |
|---|---|---|
| `qwen_models` | `{"qwen-0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "qwen-1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"}` | Engine-ID → HF-Repo |
| `qwen_non_streaming_mode` | `true` | Volltext vorab (User-Default) |
| ~~`qwen_xvec_only`~~ | — | **Entfernt 2026-11-15** (User: klingt nur schlechter); xvec bleibt allein der Rückfall ohne `ref_text` |
| `qwen_temperature` | Bibliotheks-Default (in Phase 3 auslesen) | Option 0,7 |
| `qwen_chunk_size` | `12` | wie im Benchmark |
| `qwen_install` | `{"packages": ["faster-qwen3-tts==0.5.4"], "transformers": "transformers==5.15.1"}` | Pins als Daten |

Bool-Felder in `_coerce` eintragen (sonst bleiben sie Strings). `_server_env` reicht alles durch.

### 3.7 Drumherum im Repo

- `engines.json` (Version **bleibt 1**): `qwen-0.6b` „Qwen 0.6B", `qwen-1.7b` „Qwen 1.7B", je mit
  englischer `description` (z. B. 1.7B: „Best voice match, needs an NVIDIA GPU with ~5 GB free VRAM
  next to the game"), `minVersion` = nächster Release-Tag (**0.0.2.0** — 0.0.1.9 ging am 2026-11-12 ohne Qwen raus).
- Starter `start-qwen.bat`/`.sh` (0.6B; 1.7B per durchgereichtem `--tts-backend qwen-1.7b`) — oder
  zwei Paare, Offene Frage 5.
- `tools/windows-test.sh` kennt `qwen-0.6b`/`qwen-1.7b`.
- `THIRD_PARTY_LICENSES.md`: faster-qwen3-tts (MIT), qwen-tts-hf (Lizenz prüfen), Qwen3-TTS-Gewichte
  (Apache-2.0 laut Modellkarte — prüfen; wäre neben MOSS die zweite Engine ohne NC-Beschränkung).
- README (Backends, Größen, „nur aktive Engine wird installiert"), CLAUDE.md.
- **Docker bleibt wie es ist** (alle klassischen Engines im Image, kein Qwen) — siehe Nicht-Ziele.

### 3.8 Außerhalb dieses Repos (eigene Sessions)

- **Echokraut-Plugin:** Optionen `xvec_only` + Temperatur für Qwen (als `--qwen-xvec-only` /
  `--qwen-temperature` an den Wrapper), eingebaute Beschreibungstexte + Lokalisierung,
  **Start-Timeout prüfen** (ein Engine-Wechsel lädt jetzt 2–6 GB nach), Hinweis „nur NVIDIA".
- **EchokrautLocalInstaller:** Args durchreichen; Restart-Policy (Exit 75) gilt unverändert.
- **Spieltest GPU-Last** (93 % beim Generieren gegen XTTS 26 %): FPS-Einbruch im Spiel messen —
  Ergebnis entscheidet, wie deutlich die Beschreibung warnen muss.

---

## 4. Phasen (je Phase: Tests grün, `node .claude/verify.js`, Commit nach Rückfrage)

1. **Register + ASR entkoppeln** — `src/engines.py`, `KNOWN_BACKENDS` abgeleitet, Test gegen
   `engines.json`; `asr.py` auf eigene Pipeline. Verhalten sonst unverändert.
2. **Lazy-Install (nur `classic`)** — pyproject-Extras, Marker je Engine/Sprache, Übersetzung alter
   Marker, `step_deps`/`step_model` auf aktive Engine, `src.models --language`. Live:
   `tools/windows-test.sh xtts` mit `--cold` (frische Installation → Größe messen) und Wechsel
   xtts → f5 → moss ohne `--cold` (Nachinstallation).
3. **Qwen** — Spike Prompt-API am PC, dann venv-Gruppe `qwen`, `QwenWorker`, Config, GPU-Gate,
   `python -m src.qwen_backend`. Live: beide Qwen-Engines über `windows-test.sh`, `/tts` streamend
   (`parts > 1`, `first` < 0,5 s nach Warmup), zweite Anfrage gleiche Stimme ohne Kodier-Pause,
   Inline-Referenz.
4. **Drumherum** — engines.json, Starter, Lizenzen, README, CLAUDE.md (CLAUDE.md ist mit 73k Zeichen
   ohnehin über Budget — Aufteilen ist eigenes Doing, hier nur neue Abschnitte in die richtige Datei).
5. Plugin/Installer — eigene Session(s), siehe 3.8.

---

## 5. Tests (Unit, ohne torch)

- Register: `KNOWN_BACKENDS == ids(engines.json)`; jede Engine hat eine bekannte Gruppe.
- Bootstrap: Marker-Logik je Engine/Sprache (fehlendes Engine-Paket → nachinstallieren, nicht
  neu bauen; fehlende Sprache → nur dieses Modell); Übersetzung alter Marker; Qwen auf
  Nicht-NVIDIA → `FatalError` ohne Download; Server-Interpreter = venv der aktiven Gruppe;
  abgesetzte uv-Befehle je Gruppe (wie `test_bootstrap_repair.py`, Befehle prüfen, nicht Module).
- `QwenWorker` gegen Fake-`FasterQwen3TTS`: Chunk-Vertrag (float32, flach, 24 kHz, Resample bei
  anderer Rate), Prompt-Cache-Treffer (zweiter Aufruf kodiert nicht neu), Schlüssel enthält
  `xvec_only`, ref_text fehlt → xvec-Rückfall + Warnung, Sprach-Mapping, Inline-Referenz.
- Protokoll-Vollständigkeit wie in `test_moss_onnx_backend.py`.
- Config: neue Bools über ENV/CLI als echte Bools.

---

## 6. Offene Fragen / Entscheidungen

**Entschieden (User, 2026-11-12):**
- (2) Aufräumen alter Modelle **nur von Hand durch den User** — kein Automatismus, kein Knopf.
- (4) Engine-IDs `qwen-0.6b` / `qwen-1.7b`.
- (5) Zwei Starter-Paare (je Modell eins).
- (6) **Erst Lazy-Install (Phasen 1–2), dann Qwen.**

**Offen:**
1. ~~Braucht Qwen ein eigenes venv?~~ **Entschieden 2026-11-12: ja, `.venv-qwen` mit 0.5.4.**
   faster-qwen3-tts 0.3.2 (`qwen-tts`, transformers 4.57.3) hätte ins classic-venv gepasst. Gemessen
   am PC, 48 Zeilen je Version, gleiche Seeds: rtf 0,314 (0.3.2) / 0,361 (0.5.4), erster Ton
   0,37 / 0,48 s, WER 0,2 / 0,2 %, Sprecherähnlichkeit 0,728 / 0,755, VRAM 2,89 / 2,90 GB.
   **Hörurteil User: 0.5.4 hörbar besser** → eigenes venv. (Testvenv `E:\TTSBench\venv_fq032`,
   Ausgaben `qcmp_032`/`qcmp_054`.)
3. ~~AMD~~ **Entschieden 2026-11-13: AMD zulassen** (ROCm Linux + `rocm_win`), als experimentell
   markiert, **ohne CPU-Rückfall** — scheitert der Worker-Bau/Selbsttest, bricht der Start mit klarer
   Meldung ab. Ein Bekannter des Users testet auf AMD-Hardware. DirectML/CPU/Intel: abgelehnt.

## 7. Risiken

- **Verschobene Auflösung beim Nachinstallieren** in `classic`: ein später ergänztes coqui-tts kann
  geteilte Pakete verschieben, die F5 gebraucht hat. Abwehr: dieselben Constraints (transformers,
  datasets) bei JEDER Nachinstallation, Torch-Re-Pin, und Verifikation ALLER bereits installierten
  Engines dieser Gruppe vor dem Marker.
- **`qwen-tts-hf` zieht gradio & Co.** (im Benchmark-venv vorhanden). Falls unnötig groß: `--no-deps`
  mit kuratierter Liste wie bei MOSS — erst messen, dann entscheiden.
- **CUDA-Graph-Capture beim Start** kostet Zeit und VRAM; mit `max_workers=1` unkritisch, bei mehr
  Workern je Worker eigene Graphen.
- **Windows-Pfadlänge/Symlinks** beim HF-Download unter hub 1.x — durch den Windows-Test abgedeckt.

## 8. Nicht-Ziele

- Pocket TTS (eigenes Doing; wird voraussichtlich eine dritte venv-Gruppe — der Entwurf hält das offen).
- Docker-Image mit Qwen (bräuchte ein zweites venv im Image oder eine eigene Variante).
- Automatisches Löschen von Gewichten (Offene Frage 2).
- Plugin-/Installer-Code (3.8).
