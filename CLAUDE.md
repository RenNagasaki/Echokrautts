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
  httpx fastapi numpy`, then `.venv-test/Scripts/python -m pytest -q` → **643 passed, 3 skipped** (2026-11-15; auf macOS fallen 5 Tests in `test_bootstrap_rocm_windows.py` vorbestehend rot, weil der Bootstrap darwin ablehnt)
- **Prüfkommando: `node .claude/verify.js`** (gitignored, maschinen-lokal) — ruft die Suite im
  `.venv-test` auf und lässt auf macOS genau `test_bootstrap_rocm_windows.py` aus (Grund oben).
  (skip = symlink test, Windows privilege). Suite must stay green.
- **Runtime venv** (`wrapper/.venv`, gitignored): created by the bootstrap with torch+f5-tts.

## One-Click-Starter (Repo-Root)
- **Pro Backend ein Starter-Paar** im Repo-Root, manuelle 1-Klick-Launcher: `start-f5tts.bat`/
  `start-f5tts.sh` (F5-TTS), `start-xtts.bat`/`start-xtts.sh` (XTTS-v2) und `start-moss.bat`/
  `start-moss.sh` (MOSS-TTS-Nano), `start-pocket.*` (Pocket TTS), `start-qwen-0.6b.*` / `start-qwen-1.7b.*` (Qwen3-TTS). Der einzige Unterschied ist das durchgereichte `--tts-backend f5`
  bzw. `xtts` bzw. `moss` — der Bootstrap installiert seit 2026-11-12 NUR diese Engine (F5: nur die aktive Sprache) und lädt bei
  einem Wechsel nach (Umschalten = Neustart). Sie rufen die jeweiligen `wrapper/bootstrap/install_*`-
  Starter mit `--start --tts-backend <…>` auf (uv holen → installieren → servieren) und reichen
  Extra-Args durch (z.B. `--language en`). Die `.bat`-Varianten laufen in sichtbarem Fenster mit
  `pause` am Ende; die `.sh`-Varianten machen `exec` auf `install_linux.sh`. Unterschied zum C#-Host:
  der startet `bootstrap.py` direkt/versteckt — diese Starter sind für den Menschen am Rechner
  gedacht. (Früher: ein einzelnes `start.bat`/`start.sh` ohne Backend-Wahl.)

## Weitere Doku — vor Arbeit im Bereich dort lesen
Die Detail-Doku ist nach Bereichen aufgeteilt (2026-11-13). `CLAUDE.md` in Unterordnern lädt erst,
wenn eine Datei dort gelesen wird; die Dateien unter `docs/claude/` laden NIE von selbst.

| Bereich | Datei |
|---|---|
| Wrapper-Kern: config, engines.py, models, hfcache, ndjson, progress, samples, gpu_detect, engine, jobs, i18n (Nutzertexte) | `wrapper/src/CLAUDE.md` |
| HTTP-Schicht: server, Web-UI, ratelimit, wav, inline_ref, asr | `docs/claude/http-api.md` |
| Engines F5/XTTS/Qwen/Pocket, Custom-Modell, Sprachen | `docs/claude/engines.md` |
| MOSS (PyTorch + ONNX, Streaming, Knistern, Sprache der Referenz) | `docs/claude/moss.md` |
| Bootstrap: Installation, Lazy-Install, venv-Gruppen, Marker, Instanz-Lock, Download-Fortschritt | `wrapper/bootstrap/CLAUDE.md` |
| Docker-Image + GitHub-Actions, Voice-Pack-Download (`src/voicepack.py`) | `docker/CLAUDE.md` |
| Windows-Test vom Mac, Pocket-Mirror | `tools/CLAUDE.md` |
| Release-Testplan (Wrapper + Installer + Plugin, je Release) | Plugin-Repo `docs/pre-release-ingame-checklist.md` › „🎯 Release-Testplan“ |

## Release-Asset bauen (`build-release-zip.py`, Repo-Root)
- `python build-release-zip.py` schreibt **`wrapper/EchokrauTTS.zip`** (gitignored); Release danach
  von Hand auf GitHub anlegen und die Datei hochladen. `--list` zeigt nur, was hineinkäme.
- **`build-release-zip.bat` im Repo-Root ist der Doppelklick-Weg** (Argumente werden durchgereicht, also
  auch `build-release-zip.bat --list`). Sie sucht den Interpreter, statt `python` anzunehmen:
  Test-venv → `py -3` → `python` → `uv run --no-project python`. Grund: auf Windows ist `python` häufig
  nur der WindowsApps-Platzhalter, der den Store öffnet statt Python zu starten — ein Doppelklick-Skript
  darf daran nicht scheitern. Das Bau-Skript selbst braucht nur die Standardbibliothek, jedes Python 3
  genügt. Kein `.sh`-Gegenstück: unter Linux ist `python build-release-zip.py` der ganze Befehl.
- **Layout ist Vorgabe des C#-Hosts, keine Wahl:** er lädt das Asset, entpackt es nach
  `<installRoot>/echokrautts` und startet dort `bootstrap/bootstrap.py`. Im Archiv liegt deshalb der
  INHALT von `wrapper/` in der Wurzel, ohne `wrapper/`-Präfix — verifiziert gegen das echte
  0.0.0.4-Asset (identische Top-Level-Struktur, keine Datei verloren).
- **Was hineinkommt, entscheidet git**, nie eine Handliste: `git ls-files --cached --others
  --exclude-standard`. Bewusst NICHT nur `--cached` — ein Release, direkt nach dem Schreiben einer
  neuen Datei gebaut, hätte sonst genau den Code nicht drin, für den es gemacht wurde. Umgekehrt
  fallen `.venv`, `.state`, `models`, geladene `samples`, `__pycache__` und das Archiv selbst
  automatisch raus, weil `.gitignore` sie schon kennt.
- **`CLAUDE.md` unter `wrapper/` bleibt draußen** (`DEV_ONLY_NAMES`, seit der Doku-Aufteilung 2026-11-13);
  `.dockerignore` schließt `**/CLAUDE.md` aus dem Image aus.
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

## `engines.json` (Repo-Root) — die Engine-Liste, die das Plugin liest

**Wer hier eine Engine hinzufügt oder entfernt, pflegt `engines.json` im selben Arbeitsschritt.**
Das Echokraut-Plugin baut sein Engine-Dropdown aus dieser Datei, nicht mehr aus einer eigenen
Aufzählung — genau deshalb muss eine neue Engine hier eingetragen werden, sonst ist sie für die
Nutzer nicht auswählbar, egal wie fertig sie im Wrapper ist.

```jsonc
{
  "version": 1,                       // Schema-Version; ändert sich nur, wenn Felder dazukommen
  "engines": [
    { "id": "xtts", "name": "XTTS", "minVersion": "0.0.0.1" },
    { "id": "f5",   "name": "F5",   "minVersion": "0.0.0.1" },
    { "id": "moss", "name": "MOSS", "minVersion": "0.0.1.0" },
    { "id": "pocket", "name": "Pocket TTS", "description": "…", "minVersion": "0.0.2.0" },
    { "id": "qwen-0.6b", "name": "Qwen 0.6B", "description": "…", "minVersion": "0.0.2.0" },
    { "id": "qwen-1.7b", "name": "Qwen 1.7B", "description": "…", "minVersion": "0.0.2.0" }
  ]
}
```

- **Bewertungsfelder (optional, 2026-11-13):** `quality` 1–5 (relativ zu den anderen), `hardware`
  `gpu`/`cpu`, `vramGb`, `cores`, `load` `low`/`medium`/`high` → das Plugin zeigt
  „Qwen 1.7B (Quality 5/5, GPU ~5 GB, high load)“ direkt im Dropdown. **Reihenfolge der Einträge =
  Dropdown-Reihenfolge, beste zuerst** (Plugin-Test prüft das). Noten (User abgenickt): Qwen 1.7B 5,
  Qwen 0.6B 4, XTTS 4, F5 3 (deutsch nie gemessen), Pocket 3, MOSS 1 — Grundlage WER/ECAPA mit 7
  deutschen FFXIV-Referenzen (Qwen 0,4 %/0,73, XTTS 4,1 %/0,66, Pocket-24 1,7 %/0,60, MOSS 9,3 %/0,61).
- **Im Wrapper liest sie NIEMAND.** Sie ist ein reines Veröffentlichungs-Artefakt für das Plugin und
  liegt deshalb im Repo-Root, nicht unter `wrapper/`. Sie ist auch **nicht Teil des Release-Archivs**
  (`build-release-zip.py` packt nur `wrapper/`) — das Plugin holt sie über `raw.githubusercontent`.
- **`version` bleibt 1**, auch beim Hinzufügen von `description`: das Plugin
  (`TtsEngineCatalogService.ExpectedVersion = 1`) verwirft jede andere Version und fiele auf seine
  eingebettete Kopie zurück — ohne die neue Engine. `description` (englisch) kennt es bereits.
- **`id`** ist exakt der Wert von `--tts-backend`. Das Plugin speichert ihn wörtlich in seiner
  Konfiguration und reicht ihn durch — ein Tippfehler startet den Wrapper mit einer Engine, die es
  nicht gibt.
- **`minVersion`** ist der **Release-Tag, mit dem die Engine zum ersten Mal ausgeliefert wird**, nicht
  der aktuelle Stand des Branches. Eine noch nicht veröffentlichte Engine trägt also den Tag, den das
  nächste Release bekommen wird.
- **Das Plugin liest die Datei am Release-Tag** (`raw.githubusercontent.com/…/<tag>/engines.json`),
  Rückfall auf `main` und dann auf eine im Plugin eingebettete Kopie. Ein Tag ist eingefroren: was
  beim Release hier steht, sehen die Nutzer dieser Wrapper-Version für immer — die Datei gehört
  deshalb **vor** dem Taggen richtiggestellt.
- Wer eine ältere Wrapper-Version installiert hat, sieht eine neuere Engine trotzdem in der Liste,
  markiert mit „erst ab Version X", und kann sie erst nach einem Wrapper-Update wählen. Das ist der
  Zweck von `minVersion`.
- Eine entfernte Engine fliegt aus der Datei. Wer sie noch eingestellt hat, behält seine Einstellung
  (das Plugin zeigt weiter, was tatsächlich konfiguriert ist) — der Wrapper **lehnt eine unbekannte
  Engine seit 2026-09-09 beim Start ab**, mit der Liste der gültigen Namen (`engine.KNOWN_BACKENDS`).
  Vorher war F5 zugleich Default UND letzter Zweig, also wurde jeder unbekannte Wert stillschweigend
  zu F5: falsche Stimme, kein Fehler, und `/health` meldete weiter den angefragten Namen. Realistische
  Quelle ist ein Tippfehler im dokumentierten `F5W_TTS_BACKEND` (steht in allen drei Compose-Dateien)
  oder eine `id`, die ein Plugin vor einer Umbenennung gespeichert hat.
- **`KNOWN_BACKENDS` in `engine.py` und die `id`s hier müssen dieselbe Menge sein.** Eine Engine, die
  nur hier steht, lässt sich wählen und bricht dann beim Start ab; eine, die nur dort steht, ist für
  Plugin-Nutzer unsichtbar.

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
- **Remotes:** `origin` = GitLab (private GitLab, project `echotools/echokrautts`, primary).
  GitHub (`RenNagasaki/Echokrautts`) is a read-only snapshot mirror (CI, since 2026-11-23). Default branch `main`.

## CI (`.gitlab-ci.yml`)
- **Not the C# Dalamud pipeline** — no SonarQube/.NET here. Two stages:
  - `test` (every branch): `python:3.12-slim`, installs only dev deps (`pytest pytest-asyncio
    httpx fastapi numpy`) — NOT `pip install .[dev]` (that pulls f5-tts→torch, which tests mock),
    then `cd wrapper && python -m pytest -q`.
  - `privacy-gate` (`tree` mode) → `mirror-to-github` (`main` + tags): orphan snapshot commit of the
    pipeline tree, force-pushed to `main` of hardcoded `RenNagasaki/Echokrautts`; tag pipelines repoint
    the tag at the snapshot (non-force). **Never `--mirror`** (deletes GitHub-only release tags).
- CI/CD vars needed: `GITHUB_USER` (PAT owner / auth identity, ≠ repo owner) + `GITHUB_PAT`.
- **Snapshot mirror since 2026-11-23** (like Echokraut; before: full history, which the privacy gate
  blocked). Old tags `0.0.0.1`–`0.0.0.5` and other old refs on GitHub still point at the full history;
  removing them is a manual step. The release-triggered Docker build works either way (the workflow
  file travels in the tree, tags get repointed).
