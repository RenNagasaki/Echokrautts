# MOSS-TTS-Nano (PyTorch + ONNX)
Aus der Root-`CLAUDE.md` verschoben (2026-11-13).

## Module
- `src/moss_backend.py` — **MOSS-TTS-Nano (OpenMOSS, 0.1B), das DRITTE Backend (2026-09-08).**
  `MossWorker`: gleicher Kontrakt (float32 @ 24000 Hz), klont aus Audio OHNE Transkript (wie XTTS),
  **`supports_streaming = True`** mit echtem `infer_stream`.
  - **Warum es die anderen schlägt, LIVE gemessen (RTX 5090, vor dem Bauen):** **CPU ist genauso
    schnell wie die GPU** (rtf 1,25 gegen 1,22) ⇒ **`moss_device` steht per Default auf `"cpu"`** und
    lässt die Grafikkarte beim Spiel, was das eigentliche Ziel war. `"auto"` als Default wäre falsch
    gewesen: es hätte still die GPU genommen, also genau das, wogegen dieses Backend antritt · **erster Ton nach 0,12 s (GPU) / 0,45 s (CPU)** gegen ~2,4 s
    bei XTTS · **Apache-2.0 für Code UND Gewichte** (einzige Engine ohne NC-Beschränkung) ·
    **312 MB** gegen F5 ~5 GB / XTTS ~2 GB · 19 Sprachen inkl. aller vier Client-Sprachen.
  - ⚠ **Klangurteil des Users (2026-09-08, nach dem ersten echten Einsatz): „Klappt gut. Klingt
    erwartbarerweise jetzt nicht so prickelnd, ist aber für die mit schlechten PCs besser als gar
    nichts.“** → **MOSS ist die Option für schwache Rechner, NICHT der Qualitätssieger.** Wer die GPU
    übrig hat, nimmt XTTS. Ein 0,1B-Modell klingt wie ein 0,1B-Modell; sein Wert ist, überhaupt
    benutzbar zu sein, wenn die Karte dem Spiel gehört. Steht auch im README, damit niemand MOSS als
    Standard vorschlägt.
  - ⚠ **Ehrliche Grenze: rtf ~1,25 ist LANGSAMER als Echtzeit.** Bei kurzen Spielzeilen verdeckt das
    Streaming es; ein langer Satz wird eingeholt. F5 (0,32) und XTTS (0,44) sind in der Summe
    schneller, aber langsamer zum ersten Ton.
  - **Verzeichnisnamen mit UNTERSTRICH, nicht Bindestrich** (`models/moss_tts_nano`): MOSS liefert
    seinen Modellcode mit (`trust_remote_code`), transformers macht aus dem Verzeichnisnamen einen
    Modulnamen, und ein Bindestrich wird wörtlich zu `_hyphen_` — der erzeugte Code importiert sich
    dann selbst unter einem Namen, den es nicht gibt. Live beim ersten Laden aufgeschlagen.
  - **`_use_models_cache()` läuft VOR dem Import der Runtime**, nicht danach: transformers legt seine
    Cache-Pfade beim IMPORT fest. Zu spät gesetzt landete der Modellcode im maschinenweiten Cache
    (live: `G:\cache`). Gesetzt werden `HF_HOME`, `HF_HUB_CACHE` **und `HF_MODULES_CACHE`** —
    letzteres ist der separate Cache für `trust_remote_code`-Code.
  - **Die Runtime schreibt JEDE Generierung als Datei** und bietet keinen Schalter dagegen. Der
    Worker legt sie deshalb in ein eigenes `models/moss-scratch/` und löscht sie nach jedem Request
    (`_cleanup`, nie fatal) — sonst wächst die Platte bei tausenden Spielzeilen unbegrenzt.
  - **⚠ DIE SPRACHE DER REFERENZ BESTIMMT DIE LAUTUNG — MOSS hat KEINE Sprachkonditionierung
    (2026-09-10, vom User gehört bestätigt).** Das Modell leitet die Sprache aus dem TEXT ab; es gibt
    kein Sprach-Token und kein sprachspezifisches Gewicht. Eine englische Referenzstimme mit deutschem
    Text ergibt deshalb deutschen Text mit **hartem englischem Akzent**. Das ist kein Fehler der Engine,
    sondern eine Eigenschaft des Zusammenspiels — und es trifft uns hart, weil **alle 260 Stimmen des
    Voice-Packs Englisch sind** (LibriTTS-R, siehe `samples/ATTRIBUTION.csv`).
    ⚠ **Dieses Repo verteilt KEINE Stimmen.** Das Voice-Pack gehört dem Echokraut-PLUGIN und liegt in
    dessen Releases (`src/voicepack.py` holt es beim Erststart nur, damit eine frische Installation
    nicht stumm ist). Ein deutsches Pack ist deshalb Arbeit im Plugin-/VoicePack-Repo, nicht hier —
    hier gehört nur die Erkenntnis hin, WARUM es gebraucht wird.
    **Gegenprobe gemacht:** derselbe deutsche Satz aus einer echten deutschen Referenz (HUI-Audio-Corpus)
    klingt „deutlich besser“ (Urteil des Users). F5 (deutsches Finetune) und XTTS (explizites Sprach-Token)
    haben eine deutsche Lautung im Modell und sind deshalb weniger betroffen — das ist die Erklärung, nicht
    gemessen. **Folge: ein deutsches Voice-Pack würde MOSS am meisten nützen.** Arbeit im VoicePack-Repo.
  - **`moss_stream` (Default `true`): MOSS streamt — den Rückstand auszugleichen ist Sache des
    KONSUMENTEN, nicht des Wrappers (User-Entscheidung 2026-09-10).** Anlass war ein Nutzerbericht:
    Web-UI (`format:"wav"`, komplett gepuffert) sauber, Stream ins Spiel knistert — bei **denselben
    Bytes**, denn der Wav-Pfad hängt nur einen RIFF-Header vor genau die Teile des PCM-Pfads. Der
    Unterschied ist allein die Stückelung.
    **Ursache, gemessen:** MOSS ist die einzige Engine hier mit **rtf > 1** (1,25–1,44 gegen F5 0,32
    und XTTS 0,44). Wer beim ersten Teil losspielt, verbraucht eine Sekunde Audio je Sekunde, während
    weniger als eine ankommt — der Rückstand wächst mit der Satzlänge (**2,80 s** bei einem
    6,4-s-Satz). Das Plugin startet mit 1 s Kissen (`StreamBufferPolicy.InitialCushionMs`).
    **Ein fester Vorlauf löst das NICHT** — nötig ist `L ≥ D·(r−1)`, das skaliert mit der Satzlänge.
    Gemessen an echten Ankunftsprofilen (`scratchpad/moss_headstart.py`): 0,5 s trägt bis ~40 Zeichen,
    bei 79 Zeichen 1–10 Unterläufe, bei 164 Zeichen 48–52.
    **Ein fester NACHFÜLL-Wert ist noch schlechter** (`scratchpad/moss_policies.py`, Last durch
    Strecken der Zeitstempel modelliert). 164 Zeichen / 15 s Audio bei rtf 1,5: fest 0,25 s → **15**
    Unterbrechungen, 0,5 s → 8, 1,0 s → 4, **Verdoppeln (= Plugin heute) → 2**. Grund: nach dem
    Nachfüllen von `L` spielt man noch `L·r/(r−1)` Sekunden. **Das vorhandene Verdoppeln ist also
    bereits die beste dieser Varianten — nicht durch einen Festwert ersetzen.**
    ⚠ **Der MECHANISMUS des Knisterns ist UNERKLÄRT.** Die naheliegende Vermutung — die 500-ms-Sperre
    (`MinRebufferIntervalMs`) lasse Unterläufe unbehandelt durch — ist von der Simulation widerlegt
    (mit und ohne Sperre dieselben 2 sauberen Pausen, null unbehandelte Lücken). Nächster Kandidat:
    ein Übergang ohne Ausblendung beim Fortsetzen (`FadeInMs` gibt es nur beim ERSTEN Start).
    Unbelegt, und Plugin-Seite.
    **`moss_stream=false`** liefert den Satz am Stück (live: `parts=1`, `first`=`generated`; mit
    `true` `parts=44`) — der Ausweg für einen Konsumenten, der nicht puffern kann, zum Preis der
    vollen Generierungszeit vor dem ersten Ton. `infer()` zieht `infer_stream()` ohnehin leer, es
    gibt also nur EINEN Erzeugungspfad; die Flagge entscheidet nur, wer wartet.
    ⚠ **MOSS ist nicht deterministisch** — zweimal derselbe Satz ergab 6,40 s und 5,28 s Audio. Ein
    Byte-Vergleich zweier Läufe ist kein gültiges Beweismittel.
  - **48 kHz stereo → 24 kHz mono** je Chunk in `_to_contract` (torchaudio, schon gepinnt).
    Durchreichen wäre nicht laut kaputt, sondern still falsch: Stereo als Mono klingt wie Müll,
    48 kHz als 24 kHz spielt mit halber Geschwindigkeit.
  - **Installation: gepinntes GitHub-ARCHIV, kein `git+https://`** (pip bräuchte sonst git auf dem
    Nutzerrechner) und **ein COMMIT, kein Branch**. `--no-deps`, weil MOSS `torch==2.7.0` deklariert
    und pip es sonst von PyPI holt und die CUDA-Build ersetzt. **`WeTextProcessing` wird bewusst
    weggelassen** — es hängt an `pynini` (keine Windows-Wheels), ist lazy importiert, optional, und
    normalisiert ohnehin nur zh/en. `_verify_moss` importiert `moss_tts_nano_runtime` (TOP-LEVEL-Modul,
    nicht das Paket!) vor `deps.done`.
- `src/moss_onnx_backend.py` — **MOSS auf ONNX Runtime, der DEFAULT seit 2026-09-10
  (`config.moss_runtime`, Werte `onnx` | `pytorch`).** Dieselben Gewichte, nur exportiert — und damit
  die einzige gemessene Beschleunigung, die KEINE Audioqualität kostet (int8, weniger Codebooks und
  bfloat16 verändern alle, was das Modell produziert).
  **Gemessen (Ryzen Zen4, gleicher Satz, fester Seed, ms je 80-ms-Frame — unter 80 = rtf < 1):**

  | Runtime | 1 Thread | 4 Threads | 8 Threads | 16 Threads |
  |---|---|---|---|---|
  | PyTorch fp32 | 161,5 | 115,4 | 100,3 | 95,5 |
  | ONNX | **63,3** | **41,4** | **35,9** | 95,7 ⚠ |

  **ONNX auf EINEM Kern schlägt PyTorch auf sechzehn.** End-to-end durch den echten Server bestätigt:
  rtf **0,52** gegen 1,15, `parts=19` (Streaming läuft), erster Ton 0,83 s.
  - ⚠ **`moss_onnx_threads` ist bewusst NIEDRIG (Default höchstens 4), nicht kernzahl-abhängig.**
    16 Threads waren so langsam wie EINER und langsamer als vier — onnxruntime koordiniert dann mehr
    als es rechnet. Wenige Threads sind zudem die höfliche Wahl neben einem laufenden Spiel, was der
    Daseinszweck dieser Engine ist.
  - **Streaming ist NACHGEBAUT, nicht benutzt:** der Vendor dekodiert inkrementell, gibt aber erst am
    Ende alles zurück. `_decode_streaming` treibt `generate_audio_frames(on_frame=…)` auf einem Thread
    und nimmt die Stücke über eine Queue ab — die kleinste Brücke von „ruft zurück“ zu „ist ein
    Generator“. Die Budget-Politik des Vendors bleibt unangetastet.
  - **Fehlt onnxruntime oder fehlen die Graphen, fällt `_default_factory` auf PyTorch zurück und SAGT ES**
    (Warnung). Anders als ein unbekanntes BACKEND — das die falsche Stimme wäre und deshalb hart
    scheitert — ist das dasselbe Modell in langsamer. `is_available()` unterscheidet die beiden
    Ursachen, weil sie verschiedene Reparaturen brauchen.
  - **Kosten: 763 MB** exportierte Graphen (gegen 312 MB Checkpoints), eigene HF-Repos
    (`MOSS-TTS-Nano-100M-ONNX`, `MOSS-Audio-Tokenizer-Nano-ONNX`), Download ist **nicht fatal**.
  - `enable_wetext=False` fest verdrahtet: `WeTextProcessing` hängt an `pynini` (keine Windows-Wheels)
    und normalisiert nur zh/en. ⚠ Der Vendor gibt die Wellenform **kanal-ZULETZT** `(N, 2)` zurück,
    der PyTorch-Weg `(2, N)`.
  - **Die GPU ist für MOSS gemessen LANGSAMER als die CPU** (115–123 gegen 95,5 ms/Frame, dtype fast
    egal) — die Zeit geht in Aufrufe, nicht in Rechnung. Der ONNX-Worker ist deshalb CPU-fest.
- `src/moss_common.py` — `MossWorkerBase`: was BEIDE MOSS-Laufzeiten teilen (`transcribe`, `infer`,
  `self_test`, `warn_unsupported_speed`, `DEFAULT_SAMPLE_RATE`). Eine Basisklasse und kein Hilfsmodul,
  weil es Protokoll-METHODEN sind, die die Engine am Worker-Objekt nachschlaegt.
  ⚠ **Die Duplikate haben bereits einen Fehler gekostet:** `MossOnnxWorker` ging ohne `self_test`
  raus. Die Engine ruft es auf dem fragilen Geraetepfad (dml/xpu/rocm_win) in einem `try` auf, also
  gab es keinen Absturz — nur die Meldung „worker init failed“ ueber einen Worker, der einwandfrei
  gebaut hatte, und einen unnoetigen Neuaufbau auf der CPU. Eine stille, irrefuehrende Verschlechterung,
  nach der niemand sucht. Ein Test in `test_moss_onnx_backend.py` haelt jetzt BEIDES fest: dass beide
  Klassen das ganze Protokoll erfuellen, und dass sie die geteilten Methoden nicht erneut definieren.
- `src/moss_audio.py` — `ContractResampler`: die 48-kHz-Stereo-zu-24-kHz-Mono-Umwandlung je Chunk, an
  EINER Stelle für beide MOSS-Laufzeiten. Enthält den Naht-Fix (`OVERLAP = 64`, siehe Knister-Eintrag).
  **Die Zielrate wird je Aufruf übergeben, nicht gespeichert** — `worker.sample_rate` ist der Vertrag,
  eine zweite Kopie davon wäre beim ersten Wechsel uneins geworden (ein Test hält genau das fest).
