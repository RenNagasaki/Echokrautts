# Bootstrap / Installation (`wrapper/bootstrap/`)
Aus der Root-`CLAUDE.md` verschoben (2026-11-13). Das Instanz-Lock wohnt in `src/instance.py`,
die Erkennung in `src/gpu_detect.py` (siehe `wrapper/src/CLAUDE.md`).

## Installation
- `src/instance.py` — **ein Installationsordner, ein Besitzer (2026-09-27, aus einem Tester-Log).**
  Der Installer wurde gestartet, während Lauf 1 noch torch installierte. Der zweite Bootstrap machte
  `uv venv --clear` auf ein venv, in das der erste noch schrieb (`Lib: directory is not empty`,
  os error 145). Danach hielten die Reste `.venv\Scripts` dauerhaft fest (os error 5, 30 min lang,
  über drei Spielneustarts). `InstanceLock` (`.state/bootstrap.lock`, msvcrt/fcntl; die PID steht in
  einer EIGENEN Datei `bootstrap.pid`, weil ein Windows-Byte-Lock die gesperrten Bytes für alle
  anderen unlesbar macht) wird als ERSTES in `main()` genommen, über `_claim_install_folder` und für
  die ganze Lebenszeit samt Serve gehalten. Das OS gibt ihn beim Tod frei, verwaiste Locks gibt es
  also nicht. **Der neue Lauf ÜBERNIMMT** (`take_over`): er beendet den Halter, dessen
  Kill-on-close-Job dann den ganzen Baum mitnimmt. Danach beendet `_stop_stale_processes` alles, was
  aus `.venv` oder `.uv` läuft, samt Nachfahren (ein venv-`python.exe` ist nur ein Launcher, der
  echte Interpreter ist sein KIND). Das läuft auch zwischen den Retries in `_create_venv`.
  Ausgenommen sind wir selbst und unsere Vorfahren, denn der Host startet uns über
  `.uv\uv.exe run`. **Bewusst NICHT der ganze Wrapper-Ordner**, dort liegt auch `.venv-test`.
  ⚠ Folge: zwei Starter im selben Ordner laufen nicht mehr parallel (auch nicht auf verschiedenen
  Ports). Der zweite beendet den ersten. Nie fatal: Ein fehlgeschlagener Sweep wird nur geloggt.
  Live geprüft: Übernahme eines echten Lock-Halters in 0,25 s.
- **Ein fehlendes ENGINE-Paket wird nachinstalliert, NICHT mit einem venv-Neubau beantwortet
  (2026-09-10, aus einem echten gescheiterten Upgrade).** `_existing_venv_problem` gibt seither
  `(Grund, reparierbar)` zurück: falsches torch/transformers ⇒ Neubau (die Grundlage lässt sich nicht
  flicken), fehlende Engine ⇒ `_install_engines` in das VORHANDENE venv, Marker bleibt. Der erste
  0.0.1.0-Upgrade scheiterte genau hier: das Alt-venv hatte kein MOSS, die Reparatur wollte neu bauen,
  und `uv venv --clear` starb an `Zugriff verweigert (os error 5)` auf `.venv\Scripts` — **mit bereits
  gelöschtem site-packages, also einer kaputten Installation.** Ein 14-MB-Paket hätte gereicht.
- **`_create_venv` wiederholt bei gesperrtem Verzeichnis** (5 Versuche, wachsende Pause). Unter Windows
  lässt sich ein Verzeichnis nicht entfernen, solange eine Datei darin offen ist — und die
  wahrscheinlichsten Halter sind der Server, der eine Sekunde vorher lief, und die
  Verifikations-Subprozesse dieses Schritts selbst. Bleibt es gesperrt, nennt die Meldung die Ursache
  und was zu tun ist, statt eine errno zu zitieren.
- **`_install_engines` ist die gemeinsame Installationsfolge** für Neubau UND Reparatur. Die
  Reihenfolge darin ist tragend (Engines VOR dem Torch-Re-Pin); eine zweite Kopie wäre beim ersten
  Anfassen auseinandergelaufen.
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

## Multi-backend (lazy install: only the active engine, 2026-11-12)
- **Lazy install** (Plan `docs/plans/qwen3-und-lazy-install.md`, Nutzerbeschwerde „Ordner zu groß“:
  vorher ~6,6 GB venv + 11,6 GB Gewichte für jeden): `--tts-backend <f5|xtts|moss>` wählt, was
  installiert UND geladen wird; ein Wechsel ist ein Neustart, der Fehlendes nachinstalliert.
  - **venv-Gruppen** (`EngineSpec.group`): `classic` = f5/xtts/moss im historischen `.venv` (alte
    Installationen bleiben gültig); andere Gruppen → `.venv-<gruppe>` (Qwen braucht transformers 5).
    Der Server läuft im venv der aktiven Engine (`step_serve`).
  - **Marker:** `deps-<gruppe>` (venv + Fundament torch/transformers), `engine-<id>` (Paket
    importierbar), `model-<id>` bzw. `model-f5-<sprache>`. `_migrate_legacy_markers` übersetzt das
    alte `deps.done`/`model.done` (alles installiert) beim ersten Start.
  - **`step_deps`:** Fundament kaputt → venv neu (alle Engine-Marker der Gruppe weg); sonst jede als
    installiert markierte Engine prüfen + aktive ergänzen, **in place** nachinstallieren
    (`_install_engines(…, engine_ids)`: torch → Wrapper+transformers-Pin → Engines → torch-Re-Pin →
    torchcodec raus → Extras), danach ALLE Engines der Gruppe verifizieren. Eine (neu)installierte
    Engine verliert ihren Modell-Marker (Download läuft idempotent erneut).
  - Je Engine `ENGINE_INSTALLERS`/`ENGINE_VERIFIERS` im Bootstrap (Test: Schlüssel == KNOWN_BACKENDS).
    `f5-tts` steht NICHT mehr in `pyproject.toml` (sonst zöge jede Installation F5) — das
    Dockerfile nennt es deshalb ausdrücklich. MOSS braucht transformers, deklariert es aber nicht
    (`--no-deps`) → der transformers-Pin wird im Basisschritt installiert, nicht nur als Constraint.
  - Aufräumen nicht mehr genutzter Gewichte macht **nur der User von Hand** (Entscheidung 2026-11-12).
- (Historisch) `--tts-backend` (config `tts_backend`, default `f5`) selects the engine. `_default_factory` in `engine.py` picks `F5TTSWorker` vs `XTTSWorker` (XTTS imported
  lazily). Bootstrap forwards the choice via `F5W_TTS_BACKEND`; `/health` reports `tts_backend`.

## Deps-Auflösung
- **Deps install (`step_deps`):** (bis 2026-11-12 EINE Auflösung für f5-tts + coqui-tts; seit dem
  Lazy-Install je Engine ein Aufruf, jeweils mit dem transformers-Pin.) The torch re-pin +
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
- **On the AllTalk checkout
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
