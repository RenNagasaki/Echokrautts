# Werkzeuge (`tools/`)
Aus der Root-`CLAUDE.md` verschoben (2026-11-13). `tools/pocket-mirror.py`: siehe `docs/claude/engines.md` (Pocket).

## Windows-Test vom Mac (`tools/windows-test.sh` + `tools/windows-test.ps1`, 2026-11-08)
- Testet den **echten Windows-Installationsweg** (das kann die Linux-CI nicht): Arbeitskopie per
  `git ls-files -co | tar | ssh` auf den Gaming-PC (`E:/EchokrauttsTest`, C: hat zu wenig Platz) →
  je Engine `install_win.ps1 --start` → `/health` (CUDA-Erkennung, bei moss CPU) → `/samples` →
  `POST /tts` (WAV ≥ 0,5 s) → `/shutdown`. Ergebnisse (NDJSON-Log, `detection.json`, `/health`, WAV)
  per scp nach `windows-test-out/` (gitignored). Startbar vom User oder von Claude:
  `tools/windows-test.sh [xtts|f5|moss …] [--keep] [--cold] [--no-wake]`.
- **Bewusst nicht in GitLab** (User-Entscheidung 2026-11-08: kein Runner, kein Extra-Konto auf dem PC).
  Läuft per ssh unter dem normalen PC-Konto; ssh-Alias `gaming-pc` und Schlüssel kommen aus dem
  Repo `private/streaming-pc` (`mac/install-mac.sh`, `steps/20-openssh.ps1`, Standard-Shell PowerShell).
- **Voice-Pack:** die native Installation bringt absichtlich keine Stimmen (das Plugin liefert sie) — ist
  `samples/` leer, ruft der Test einmal `python -m src.voicepack` auf, wie es sonst das Plugin täte.
- **Erster grüner Lauf 2026-11-08** (RTX 5090, CUDA cu128 erkannt): xtts 4,1 s Audio in 18 s (inkl.
  Aufwärmen), f5 2,4 s in 17 s, moss (CPU/ONNX) 2,9 s in 2,1 s; frische Installation bis `/health`
  ~10 min, belegt ~18 GB auf E:.
- Ohne `--keep` jedes Mal eine frische Installation (Ordner wird gelöscht, alte Server-Prozesse aus dem
  Ordner beendet); die erste Engine installiert, die weiteren nutzen dieselbe Installation. Schläft der
  PC, weckt das Skript ihn per WoL (Broadcast aus dem Heimnetz).
- ⚠ Remote-Befehle laufen in `powershell -c`: ohne `exit 0` am Ende meldet ssh den Status des letzten
  Befehls — ein `Remove-Item -ErrorAction SilentlyContinue` ins Leere ergibt Exit 1. Binärdaten nie per
  `tar -cf -` über PowerShells stdout zurückholen (PS 5.1 kann umkodieren), deshalb scp.
- ⚠ **ssh-Sitzungen auf dem PC laufen erhöht (Admin)** → uvs Python unter `%APPDATA%\uv\python`
  (Junction `cpython-3.11-…`, vom normalen Konto angelegt) scheitert mit `os error 448` „nicht
  vertrauenswürdiger Bereitstellungspunkt“ (Windows lässt erhöhte Prozesse keiner Junction eines
  nicht erhöhten folgen). Dasselbe gilt für die Verknüpfung `%USERPROFILE%\.local\bin\python3.11.exe` im PATH. Der Test setzt deshalb `UV_PYTHON_INSTALL_DIR`/`UV_PYTHON_BIN_DIR` in den Testordner und `UV_PYTHON_PREFERENCE=only-managed`. Für echte
  Nutzer wohl irrelevant (Starter und Plugin laufen nicht erhöht) — **außer** jemand startet den
  Installer einmal normal und einmal „als Administrator“.
