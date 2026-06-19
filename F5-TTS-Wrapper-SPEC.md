# F5-TTS Wrapper — Implementierungs-Spezifikation

> Vorlage für Claude Code. Ziel: ein **lightweight Python-Wrapper um F5-TTS**, der von
> einer C#-Anwendung als separater Prozess installiert, gestartet, gestoppt und
> ausgelesen wird. Stellt eine REST-API für **streamendes TTS mit Voice-Cloning** bereit.

---

## 0. Kurzfassung der Entscheidungen

| Thema | Entscheidung |
|---|---|
| TTS-Engine | **F5-TTS** (`SWivid/F5-TTS`), Zero-Shot-Cloning, Flow-Matching |
| Ziel-OS | **Windows (x64) + Linux (x64)** — kein macOS |
| Lizenz Projekt | AGPL-3.0 (Wrapper-Code) |
| Lizenz Modell | F5-TTS-Weights bleiben **CC-BY-NC** — separat ausweisen, nicht als Teil des AGPL-Codes |
| Parallelität | **Mehrere Inferenzen parallel** über Worker-Pool + Queue, VRAM-bewusst |
| GPU-Backends | NVIDIA (inkl. Blackwell), AMD (Linux: ROCm / Windows: **DirectML**), Intel (XPU, best-effort), sonst CPU |
| Bootstrap | Plug & Play via `uv` — lädt Python + Dependencies selbst per HTTP, **kein git/System-Python nötig** |
| Transport | HTTP, `Transfer-Encoding: chunked`, Body = rohes PCM, Metadaten in Response-Headern |
| Voice-Samples | Request schickt **nur Dateinamen** → aufgelöst gegen `<wrapper>/samples/`; Basename-only, kein Path-Traversal |
| Samples-Liste | `GET /samples` gibt alle nutzbaren Sample-**Namen** zurück |
| Netzwerk | **Fester Port + Host aus der Config** (kein dynamischer Port) → remote nutzbar; optionaler API-Key |
| Install-UX | Jeder Schritt wird als klares Fortschritts-Event (`index/total` + Klartext) gemeldet |
| C#-Anbindung | stdout = NDJSON-Events (Fortschritt, ready, errors), Prozess start/stop, `/shutdown` + Kill-Fallback |
| Crash-Isolation | Wrapper = **eigener Prozess**; ein Wrapper-Fehler darf das injizierte Spiel **nie** crashen (§13) |

---

## 1. Architektur

```
┌──────────────────────┐     start/stop (Process)      ┌───────────────────────────┐
│   C#-Anwendung        │ ────────────────────────────▶ │  Python-Wrapper (subproc)  │
│                       │                                │                            │
│  - Prozess-Mgmt       │ ◀──── stdout: NDJSON-Events ── │  bootstrap → uv/venv       │
│  - liest Console-Out  │       (progress, ready, errors)│  FastAPI + uvicorn         │
│  - HTTP-Client        │                                │  Worker-Pool (F5-TTS)      │
│                       │ ──── HTTP (REST + Stream) ───▶ │  Job-Registry / Queue      │
└──────────────────────┘                                └───────────────────────────┘
                                                              │
                                                              ▼
                                                    samples/  models/ (HF-Cache)
```

Der Wrapper ist ein eigenständiges Verzeichnis. Die C#-App entpackt nach der Installation
die Voice-Samples in `<wrapper>/samples/`, startet den Bootstrap (der beim ersten Lauf alles
installiert) und kommuniziert danach nur noch über HTTP + stdout.

---

## 2. Projektstruktur

```
wrapper/
├── bootstrap/
│   ├── bootstrap.py          # Plattform-unabhängiger Einstieg (von C# gestartet)
│   ├── install_win.ps1       # optionaler Windows-Starter (ruft bootstrap.py)
│   └── install_linux.sh      # optionaler Linux-Starter
├── src/
│   ├── server.py             # FastAPI-App, Endpoints, Lifespan
│   ├── engine.py             # F5-TTS-Worker-Pool + Inferenz
│   ├── streaming.py          # Satz-Chunking + Chunk-Generator
│   ├── jobs.py               # Job-Registry, Cancellation-Tokens
│   ├── samples.py            # Sample-Auflösung + Validierung + Transkript
│   ├── gpu_detect.py         # GPU-Erkennung + Backend-/Wheel-Auswahl
│   ├── ndjson.py             # strukturierte stdout-Events
│   └── config.py             # Konfiguration (CLI/ENV/JSON)
├── samples/                  # ← von C# befüllt (.wav + optional gleichnamige .txt)
├── models/                   # HF-Cache der F5-TTS-Weights
├── pyproject.toml            # Dependencies (ohne torch — das macht der Bootstrap)
└── config.json               # Default-Konfiguration
```

---

## 3. Bootstrap & Installation (Plug & Play)

**Aufruf durch C#:** `python bootstrap/bootstrap.py --start` — falls kein Python vorhanden ist,
startet C# stattdessen `install_win.ps1` / `install_linux.sh`, die zuerst `uv` ziehen und dann
`bootstrap.py` über das uv-eigene Python ausführen.

`bootstrap.py` ist **idempotent** und führt eine **feste, nummerierte Schrittsequenz** aus
(jeden Schritt überspringen, wenn bereits erledigt — Marker/Lock-Datei in `<wrapper>/.state`).
**Jeder Schritt meldet sich klar an und ab**, damit die C#-App dem User jederzeit „Schritt X von N:
…" anzeigen kann und niemand vor einem scheinbar eingefrorenen Fenster sitzt (Details §3.1):

1. **`uv` beschaffen** (falls nicht vorhanden): Standalone-Binary per HTTPS herunterladen
   (`https://astral.sh/uv/install.sh` bzw. die GitHub-Releases von `astral-sh/uv`),
   nach `<wrapper>/.uv/` entpacken. `uv` ist ein einzelnes Binary und bringt sein eigenes
   Python-Management mit — daher **kein System-Python und kein git nötig**.
2. **Python pinnen:** `uv python install 3.11` (Version pinnen, in `config.json` konfigurierbar).
3. **GPU erkennen:** `gpu_detect.py` ausführen → bestimmt Backend + passenden PyTorch-Wheel-Index
   (siehe §4). Ergebnis cachen.
4. **venv + Dependencies:** `uv venv` in `<wrapper>/.venv`, dann:
   - **torch/torchaudio** vom backend-spezifischen Index (`--index-url …`),
   - danach `uv pip install` der restlichen Deps aus `pyproject.toml`
     (`f5-tts`, `fastapi`, `uvicorn`, `soundfile`, `numpy`, ggf. `torch-directml`).
5. **Modell vorladen:** F5-TTS-Weights nach `<wrapper>/models/` cachen (HF-Download), damit der
   erste Request nicht blockiert. `HF_HOME`/`HF_HUB_CACHE` auf `<wrapper>/models/` setzen.
6. **Server starten:** `uvicorn` an **`host:port` aus der Config** binden (fester Port, kein
   `--port 0`) und `ready`-Event auf stdout ausgeben (siehe §8).

> **Wichtig für Claude Code:** PyTorch-Wheel-Index-URLs und das genaue F5-TTS-Paket
> (`pip install f5-tts` vs. Git-Tarball) zum Umsetzungszeitpunkt gegen
> `https://pytorch.org/get-started/locally/` und das F5-TTS-README verifizieren — die ändern
> sich häufig. Die Logik unten beschreibt die *Auswahl*, nicht fixe URLs.

### 3.1 Fortschrittsanzeige beim Install

Damit der User nie im Unklaren ist, sendet der Bootstrap pro Schritt **klare Events** (NDJSON, §8.1):

- Beim Betreten eines Schritts ein `progress`-Event mit `index` (1-basiert), `total` (Gesamtzahl
  Schritte), `step` (stabiler Key) und `message` (Klartext, z. B. „Lade PyTorch (CUDA 12.8) …").
- Bei langen Downloads (uv, torch, Modell) zusätzlich periodische `percent`-Updates (0–100),
  damit ein Balken animiert werden kann.
- Beim Abschluss eines Schritts ein `progress`-Event mit `done: true`.
- `skipped: true`, wenn ein Schritt dank Cache übersprungen wird (auch das anzeigen, sonst wirkt
  ein schneller Reinstall „kaputt").

So kann die C#-App durchgängig „**Schritt 4/6 – Installiere Modell … 37 %**" rendern. Die
Schrittzahl `total` ist fix (6 Schritte aus §3), reine Skip-Läufe eingeschlossen.

### Offline-/Proxy-Hinweise
- HuggingFace-Erreichbarkeit ist Voraussetzung (Modell-Download). Optional `HF_ENDPOINT`
  konfigurierbar machen für Mirror/Proxy-Umgebungen.
- Alle Downloads über HTTPS, mit Retry + Resume. Fehler als NDJSON-`error`-Event melden.

---

## 4. GPU-Erkennung & Backend-Auswahl (`gpu_detect.py`)

Reihenfolge der Erkennung; erste Übereinstimmung gewinnt. Gibt zurück:
`{ backend, device, torch_index_url, extra_packages[], max_workers_hint }`.

### 4.1 NVIDIA (CUDA)
- Erkennen: `nvidia-smi` vorhanden **und** liefert mind. eine GPU.
- **Compute Capability ermitteln** (z. B. via `nvidia-smi --query-gpu=compute_cap`):
  - **≥ 12.0 (Blackwell, RTX 50xx, sm_120)** → CUDA-12.8-Wheels (`cu128`).
  - sonst (Ada/Ampere/…) → aktueller stabiler CUDA-Wheel-Stand (z. B. `cu124`/`cu126`).
- `device = "cuda"`.

### 4.2 AMD (ROCm / DirectML)
- Erkennen: AMD-GPU via `rocminfo` / `/opt/rocm` (Linux) bzw. PCI-Vendor-ID `0x1002` (Windows).
- **Linux** → ROCm-Wheels (`rocmX.Y`), `device = "cuda"` (HIP maskiert sich als CUDA-Device).
- **Windows** → **DirectML**: CPU-Build von torch + `torch-directml`, `device = "dml"`.
  - ⚠️ DirectML hat eingeschränkte Op-Coverage. Beim Worker-Start einen kurzen Self-Test
    (Mini-Inferenz) fahren; schlägt er fehl → automatisch auf **CPU** zurückfallen und das
    per NDJSON melden.

### 4.3 Intel (XPU, best-effort)
- Erkennen: Intel-GPU (Arc) via PCI-Vendor-ID `0x8086` + dGPU.
- → `intel-extension-for-pytorch` (IPEX) + XPU-Wheels, `device = "xpu"`.
- Finicky → ebenfalls Self-Test, sonst CPU-Fallback.

### 4.4 CPU (Fallback)
- Keine GPU erkannt/verwertbar → CPU-Wheels, `device = "cpu"`, `max_workers_hint = 1`.
- F5-TTS läuft auf CPU, aber langsam — als funktionierender Fallback ok.

### 4.5 VRAM-Schätzung → `max_workers_hint`
- Freien VRAM ermitteln (`nvidia-smi`/Backend-API). F5-TTS belegt grob **~2–3 GB** pro
  Modellinstanz. Default: `max_workers = clamp(1, floor((free_vram_gb − reserve) / per_job_gb), cfg_max)`
  mit `reserve = 1.5 GB`, `per_job_gb = 3`. Auf CPU/DML: `max_workers = 1` (override per Config).

---

## 5. REST-API

Basis: `http://<host>:<port>` mit **festem Host und Port aus der Config** (§10). Für nur-lokalen
Betrieb `host = 127.0.0.1`; für **Remote-Zugriff** `host = 0.0.0.0` (lauscht auf allen Interfaces).

> **Sicherheit bei Remote:** Sobald an `0.0.0.0` gebunden wird, ist die API im Netz erreichbar.
> Daher optionaler **API-Key** (`api_key` in der Config): ist er gesetzt, müssen alle Requests
> `Authorization: Bearer <key>` mitschicken, sonst `401`. Zusätzlich empfiehlt sich, den Zugriff
> per Firewall/Reverse-Proxy abzusichern. Ist `api_key` null, bleibt die API offen (nur für
> vertrauenswürdige Netze/Loopback gedacht).

Alle Endpoints unten.

### 5.1 `POST /tts` — streamende Synthese
**Request (JSON):**
```json
{
  "sample": "anna_de.wav",        // Basename in samples/, Pflicht
  "language": "de",               // ISO-Code; steuert ggf. Modellvariante/Text-Normalisierung
  "text": "Der zu vertonende Text …",
  "ref_text": null,               // optional: Transkript des Samples; wenn null → s. §7
  "speed": 1.0,                   // optional
  "nfe_step": 32                  // optional: Qualität/Speed-Tradeoff (16=schnell, 32=Standard)
}
```
**Response:** `200`, **Streaming** (`Transfer-Encoding: chunked`), Body = **rohes PCM**.
Metadaten als Response-Header (kommen vor dem ersten Byte an, C# liest mit
`ResponseHeadersRead`):
```
Content-Type: audio/pcm
X-Job-Id: <uuid>
X-Sample-Rate: 24000        # F5-TTS-Output bei Umsetzung verifizieren
X-Channels: 1
X-Sample-Format: pcm_s16le
```
Body-Frames: nacheinander die PCM-Bytes je fertig generiertem Satz-Chunk (§6).
Bei Abbruch/Fehler mitten im Stream: Verbindung sauber schließen; Fehlerdetail zusätzlich als
NDJSON-`error`-Event auf stdout (der Stream selbst kann keine Trailer-Fehler garantieren).

### 5.2 `GET /samples` — verfügbare Voice-Samples fürs Cloning
Gibt alle nutzbaren Sample-**Namen** aus `samples/` zurück (gefiltert nach `allowed_sample_ext`).
Namen genügen:
```json
{ "samples": ["anna_de.wav", "tom_en.wav"] }
```
Optional erweiterbar via `?details=true` (liefert dann zusätzlich `has_ref_text` und `bytes` pro
Eintrag), aber der Default ist die reine Namensliste.

### 5.3 `POST /cancel/{job_id}` — Inferenz abbrechen
Setzt das Cancel-Token des Jobs. Antwort `200 {"cancelled": true}` bzw. `404`, wenn unbekannt.
Der Generator prüft das Token zwischen den Chunks und beendet den Stream.

### 5.4 `GET /jobs/{job_id}` — Inferenz-Fortschritt
Live-Stand eines laufenden Jobs, damit auch **während der Synthese** ein Fortschrittsbalken
möglich ist (der Audio-Stream selbst transportiert keinen Prozentwert):
```json
{ "state": "running", "sentences_total": 12, "sentences_done": 5, "percent": 42 }
```
`state`: `queued` | `running` | `done` | `cancelled` | `error`. `percent` ist `sentences_done /
sentences_total` (bzw. `null`, solange die Satzzahl noch nicht feststeht). C# pollt das parallel
zum Stream (~alle 250 ms); die `job_id` kommt aus dem `X-Job-Id`-Header von `/tts`. Funktioniert
sauber auch bei mehreren parallelen Jobs, da pro `job_id` getrennt.

### 5.5 `GET /health`
`200 {"status":"ok","backend":"cuda","device":"cuda:0","workers":2,"queue":0}`.

### 5.6 `POST /shutdown`
Graceful Shutdown (laufende Jobs canceln, Pool schließen, Prozess beenden). C# nutzt das primär;
als Fallback hartes Kill des Prozesses.

---

## 6. Streaming-Modell (`streaming.py`)

F5-TTS hat **kein** token-weises Streaming, sondern erzeugt pro Aufruf ein komplettes Stück.
Daher:

1. Eingabetext in **Sätze chunken** (Satzende-Erkennung; sehr lange Sätze hart nach Zeichen-/
   Tokenlimit weiter splitten, da F5-TTS pro Generierung nur begrenzte Länge stabil kann).
2. Chunks **sequenziell** durch den Worker generieren.
3. Jeden fertigen Chunk **sofort** als PCM in den HTTP-Body schreiben (yield) → erstes Audio
   kommt nach dem ersten Satz, nicht nach dem ersten Byte des Modells.
4. Optional kurze Crossfades/Trim am Chunk-Rand, um Klicks zu vermeiden.

Implementierung als `async generator`, der vom FastAPI-`StreamingResponse` konsumiert wird und
nach jedem Chunk das Cancel-Token (§7) prüft **und den Job-Fortschritt aktualisiert**
(`sentences_done`++), den `GET /jobs/{job_id}` (§5.4) ausliest.

---

## 7. Voice-Samples (`samples.py`)

- Request liefert **nur `sample` als Basename**. Auflösung:
  `path = (SAMPLES_DIR / Path(sample).name).resolve()` und **verifizieren**, dass `path`
  tatsächlich in `SAMPLES_DIR` liegt (kein `..`, keine absoluten Pfade, keine Symlinks raus).
  Sonst `400`.
- **Referenz-Transkript:** F5-TTS braucht idealerweise den gesprochenen Text der Referenz.
  - Liegt neben `anna_de.wav` eine `anna_de.txt` → diese als `ref_text` verwenden.
  - Sonst, wenn `ref_text` im Request fehlt → ASR-Transkription (Whisper o. Ä.) automatisch;
    das kostet zusätzlich VRAM/Zeit → einmalig pro Sample cachen (z. B. `.txt` schreiben, falls
    `samples/` beschreibbar, sonst In-Memory-Cache).
- `GET /samples` enumeriert nur Dateien mit erlaubter Endung (`allowed_sample_ext`, z. B.
  `*.wav`/`*.flac`/`*.mp3`); mit `?details=true` zusätzlich, ob ein Transkript vorhanden ist.

---

## 8. C#-Anbindung

### 8.1 stdout-Protokoll (NDJSON, eine JSON-Zeile pro Event)
```json
{"event":"starting"}
{"event":"progress","index":1,"total":6,"step":"uv","message":"Beschaffe uv …"}
{"event":"progress","index":1,"total":6,"step":"uv","done":true}
{"event":"progress","index":2,"total":6,"step":"python","message":"Installiere Python 3.11","skipped":true}
{"event":"progress","index":4,"total":6,"step":"deps","message":"Installiere PyTorch (CUDA 12.8) …","percent":37}
{"event":"progress","index":5,"total":6,"step":"model","message":"Lade F5-TTS-Modell …","percent":80}
{"event":"ready","host":"0.0.0.0","port":8765,"backend":"cuda","device":"cuda:0","workers":2}
{"event":"log","level":"info","message":"…"}
{"event":"error","message":"…","fatal":true}
{"event":"shutdown"}
```
- **Pflicht:** `PYTHONUNBUFFERED=1` und `PYTHONUTF8=1` setzen, nach jedem Event `flush()`,
  damit C# zeilenweise und ohne Codepage-Probleme mitliest.
- `progress`-Events tragen `index`/`total` (für „Schritt X/N"), `message` (Klartext für den User)
  und optional `percent` (Download-Balken), `done`/`skipped`. Siehe §3.1.
- Der **Port ist fix aus der Config**; das `ready`-Event bestätigt nur, dass der Server lauscht
  (und nennt Host/Port zur Sicherheit nochmal).

### 8.2 Prozess starten & Port abwarten (C#-Skizze)
```csharp
var psi = new ProcessStartInfo
{
    FileName = pythonOrBootstrapPath,
    Arguments = "bootstrap/bootstrap.py --start",
    WorkingDirectory = wrapperDir,
    RedirectStandardOutput = true,
    RedirectStandardError  = true,
    UseShellExecute = false,
    CreateNoWindow  = true,
};
psi.Environment["PYTHONUNBUFFERED"] = "1";
psi.Environment["PYTHONUTF8"] = "1";

var proc = Process.Start(psi)!;
var readyTcs = new TaskCompletionSource();
int port = config.Port; // fest aus der Config

proc.OutputDataReceived += (_, e) =>
{
    if (string.IsNullOrWhiteSpace(e.Data)) return;
    using var doc = JsonDocument.Parse(e.Data);
    var root = doc.RootElement;
    switch (root.GetProperty("event").GetString())
    {
        case "progress":
            int idx = root.GetProperty("index").GetInt32();
            int total = root.GetProperty("total").GetInt32();
            string msg = root.TryGetProperty("message", out var m) ? m.GetString()! : "";
            int? pct = root.TryGetProperty("percent", out var p) ? p.GetInt32() : null;
            // → UI: "Schritt {idx}/{total} – {msg}"  (+ optionaler Balken via pct)
            break;
        case "ready":
            readyTcs.TrySetResult();
            break;
        case "error" when root.TryGetProperty("fatal", out var f) && f.GetBoolean():
            readyTcs.TrySetException(new Exception(root.GetProperty("message").GetString()));
            break;
    }
};
proc.BeginOutputReadLine();
proc.BeginErrorReadLine();

await readyTcs.Task; // Server lauscht jetzt auf config.Host:port (ggf. mit Timeout absichern)
```

### 8.3 Stream lesen (C#-Skizze)
```csharp
var req = new HttpRequestMessage(HttpMethod.Post, $"http://127.0.0.1:{port}/tts")
{
    Content = JsonContent.Create(new { sample = "anna_de.wav", language = "de", text = txt })
};
using var resp = await http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
resp.EnsureSuccessStatusCode();

string jobId = resp.Headers.GetValues("X-Job-Id").First();
int sampleRate = int.Parse(resp.Headers.GetValues("X-Sample-Rate").First());

await using var s = await resp.Content.ReadAsStreamAsync(ct);
var buf = new byte[8192];
int n;
while ((n = await s.ReadAsync(buf, ct)) > 0)
{
    // buf[0..n] = PCM16-LE @ sampleRate, mono → an Player/Buffer geben
}

// Abbrechen aus anderem Thread:
// await http.PostAsync($"http://127.0.0.1:{port}/cancel/{jobId}", null);
```

### 8.4 Stoppen
- Zuerst `POST /shutdown` (graceful), auf `shutdown`-Event bzw. Prozessende warten (Timeout).
- Danach Fallback: `proc.Kill(entireProcessTree: true)`.

### 8.5 Kein sichtbares Konsolen-/PowerShell-Fenster (Windows)
- C# startet den Prozess mit `UseShellExecute=false`, `CreateNoWindow=true` und
  `RedirectStandardOutput/Error=true` → **es öffnet sich kein Fenster.** Alles läuft versteckt;
  die Ausgabe (Fortschritt, Prozente, Logs) landet als NDJSON in deiner C#-App, die daraus ihre
  **eigene** UI rendert. Der User sieht also kein Terminal, sondern nur dein Programmfenster.
- Falls ein `.ps1`-Starter genutzt wird, ebenfalls versteckt aufrufen:
  `powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File install_win.ps1`.
- **Wichtig (häufiger Stolperstein):** Der Bootstrap startet selbst Unterprozesse (uv, pip,
  `nvidia-smi`, …). Diese lassen unter Windows **kurz ein cmd-Fenster aufblitzen**, wenn man sie
  naiv startet. Daher in `bootstrap.py`/`engine.py` **alle** `subprocess`-Aufrufe mit
  `creationflags=subprocess.CREATE_NO_WINDOW` (bzw. `STARTUPINFO` + `SW_HIDE`) versehen, damit
  weder beim Install noch zur Laufzeit irgendetwas aufpoppt.

---

## 9. Parallelität & Queue (`engine.py`, `jobs.py`)

- **Worker-Pool** mit `max_workers` Instanzen (aus §4.5). Echte Parallelität = mehrere
  F5-TTS-Modellinstanzen, jede ~2–3 GB VRAM. Pool-Größe konfigurier-/überschreibbar.
- Eingehende `/tts`-Requests → **asyncio.Queue**; ein freier Worker zieht den nächsten Job.
  Übersteigt die Last den Pool, warten Requests (Backpressure). Optional `max_queue` mit `503`.
- **Job-Registry:** pro Job `{job_id, cancel_event, worker_id, state}`. `/cancel` setzt das
  `cancel_event`; der Chunk-Generator prüft es zwischen Sätzen.
- Modell-Instanzen einmalig beim Start laden (nicht pro Request), um Latenz/VRAM-Churn zu sparen.
- Saubere Fehlerisolierung: Crasht ein Worker, neu instanziieren statt ganzen Prozess zu killen.

---

## 10. Konfiguration (`config.json` / ENV / CLI)

```json
{
  "host": "127.0.0.1",          // für Remote: "0.0.0.0"
  "port": 8765,                 // fester Port (kein dynamischer Port)
  "parent_pid": null,           // PID des Spiel-/C#-Prozesses; Watchdog beendet sich, wenn weg (CLI: --parent-pid)
  "api_key": null,              // gesetzt → Bearer-Token Pflicht (empfohlen bei Remote)
  "python_version": "3.11",
  "model": "F5TTS_v1",
  "samples_dir": "samples",
  "models_dir": "models",
  "max_workers": null,          // null = aus VRAM ableiten
  "vram_reserve_gb": 1.5,
  "per_job_gb": 3.0,
  "max_queue": 64,
  "asr_for_missing_ref_text": true,
  "allowed_sample_ext": [".wav", ".flac", ".mp3"],
  "hf_endpoint": null,
  "log_level": "info"
}
```
ENV überschreibt JSON, CLI überschreibt ENV.

---

## 11. Fehlerbehandlung & Logging

- Alle nutzerrelevanten Ereignisse als **NDJSON auf stdout** (§8.1); ausführliche Tracebacks
  zusätzlich auf stderr.
- Fatale Bootstrap-Fehler (kein Netz, Wheel nicht verfügbar, Modell-Download scheitert) →
  `{"event":"error",…,"fatal":true}` und Exit-Code ≠ 0.
- Request-Fehler → passende HTTP-Codes (`400` ungültiger Sample/Body, `401` API-Key fehlt/falsch,
  `404` Sample/Job, `503` Queue voll, `500` Inferenzfehler).

---

## 12. Lizenz-Hinweise (für README/Distribution)

- Wrapper-Code: **AGPL-3.0**.
- **F5-TTS-Weights: CC-BY-NC** — getrennt halten, nicht als Bestandteil des AGPL-Codes deklarieren,
  Lizenztext beilegen/verlinken. Nutzung im nicht-kommerziellen FOSS-Projekt mit Spendenlink ist
  damit vereinbar; das Modell wird zur Laufzeit von HuggingFace geladen, nicht mitausgeliefert.

---

## 13. Isolation & Crash-Resilienz (kritisch: C# ist ins Spiel injiziert)

Da die C#-Komponente in den **Spielprozess injiziert** ist, gilt als oberste Regel: **Ein Fehler im
Wrapper darf den Spielprozess unter keinen Umständen beeinträchtigen.** Die Architektur trägt das
bereits — der Wrapper ist ein **eigener OS-Prozess**, also können Python-Exceptions, CUDA-OOM oder
native Segfaults im PyTorch/F5-TTS-Stack den Spielspeicher technisch nicht berühren. Entscheidend
sind die Regeln an der Naht zwischen beiden.

### 13.1 C#-Seite — hier sitzt das echte Crash-Risiko (läuft IM Spiel)
- **Nie einen Spiel-Thread blockieren.** Sämtliche Wrapper-Kommunikation (Prozess-Start,
  stdout-Lesen, HTTP) strikt async / auf Background-Threads; **kein** `.Result`/`.Wait()` auf einem
  Spiel-Thread. Jeder HTTP-Call mit Timeout + `CancellationToken`.
- **Alles in try/catch kapseln.** Keine Exception aus der Wrapper-Interaktion darf in Spielcode
  propagieren. Toter Wrapper = gefangener Fehler = „TTS momentan nicht verfügbar", Spiel läuft weiter.
- **Supervisor/Watchdog:** `Process.Exited` abonnieren; stirbt der Wrapper, automatisch mit Backoff
  neu starten (z. B. max. N Versuche, dann TTS still deaktivieren). stdout/stderr **immer**
  leerlesen, sonst kann der Kindprozess am vollen Pipe-Puffer hängen.
- **Feature-Degradation:** TTS ist fürs Spiel optional — fällt der Wrapper aus, ohne Audio
  weiterlaufen. Kein blockierender Dialog, kein Throw, kein Warten auf einem Render-Thread.

### 13.2 Wrapper-Seite — sauber sterben statt hängen
- **Top-Level-Exception-Handler:** unerwartete Fehler → `{"event":"error","fatal":true}` + definierter
  Exit-Code, statt stillem Hang.
- **Request-/Worker-Isolation:** Eine fehlgeschlagene Inferenz gibt HTTP 500 zurück und lässt Server
  und andere Jobs am Leben; ein gecrashter Worker wird neu instanziiert (§9), nicht der ganze Prozess.
- **CUDA-OOM/Inferenzfehler abfangen** → HTTP 500 (optional Worker-Zahl reduzieren), nie Prozess-Crash.
- **Parent-Watchdog:** C# übergibt seine PID per `--parent-pid`. Der Wrapper überwacht sie und
  **beendet sich selbst**, sobald der Elternprozess (Spiel) verschwindet — so bleibt kein verwaister
  Prozess zurück, der VRAM oder den Port hält.

### 13.3 GPU-Teilung mit dem Spiel
- Läuft der Wrapper auf **derselben GPU** wie das Spiel, konkurriert er um VRAM/Compute und kann
  Framedrops verursachen. Empfehlung: `max_workers = 1` und konservativer `vram_reserve_gb`, damit
  das Spiel nicht ausgehungert wird.
- Alternativ den **Remote-Modus** nutzen (fester Host/Port, §5/§10): Wrapper auf einem **zweiten
  Rechner** laufen lassen → die Spiel-GPU bleibt komplett unberührt. Genau dafür zahlt die
  Remote-Fähigkeit hier direkt ein.

---

## 14. Vor der Umsetzung zu verifizieren

1. Aktuelle **PyTorch-Wheel-Index-URLs** je Backend (cu128/cu124/rocm/xpu/cpu) + `torch-directml`-
   kompatible torch-Version.
2. **F5-TTS-API-Signatur** der installierten Version (`from f5_tts.api import F5TTS`; Methoden
   `infer(...)`, Parameter `ref_file/ref_text/gen_text/nfe_step/speed`) und **Output-Sample-Rate**.
3. Genaues **Modell-Tag/Variante** für Deutsch (Basis-Multilingual vs. deutscher Finetune) — für
   beste Qualität deutsche Referenz-Audios empfehlen.
4. DirectML/XPU-**Self-Test** real gegen F5-TTS testen; CPU-Fallback-Pfad sicher.
5. Satz-Chunk-Längenlimits empirisch gegen die installierte F5-TTS-Version bestimmen.
