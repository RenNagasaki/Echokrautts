"""User-facing texts of the NDJSON stream, in the language of the game client.

The host (EchokrautLocalInstaller) shows every NDJSON line in its console window
and log file, so install progress, warnings and errors are read by players — in
English, German, French or Japanese, the four FFXIV client languages. This module
is the ONE table of those texts.

* :func:`t` formats a message in the active language and falls back to English
  for a missing translation, an unknown language or a broken placeholder — a
  log line must never raise.
* It returns :class:`Text`, a ``str`` that remembers its key; ``ndjson`` writes
  that key as an extra ``"key"`` field, so a consumer can react to a message
  without matching a text that now changes with the language.
* The active language is chosen ONCE per process by :func:`use_config`:
  ``config.ui_language`` if set, otherwise ``config.language`` (the plugin puts
  the client language there), otherwise English.

Developer-only lines (request timing, orphan protection, verification
diagnostics with pip output) are deliberately NOT in here: they stay English
so a support log reads the same for everyone.

Every key MUST carry all four languages with the same placeholders; a test
enforces it.
"""

from __future__ import annotations

import string
from typing import Any

SUPPORTED = ("en", "de", "fr", "ja")
FALLBACK = "en"

_active = FALLBACK


class Text(str):
    """A localized message that remembers its catalog key."""

    key: str

    def __new__(cls, value: str, key: str) -> "Text":
        obj = super().__new__(cls, value)
        obj.key = key
        return obj


def normalize(code: Any) -> str | None:
    """``"de"`` / ``"DE"`` / ``"de-DE"`` → ``"de"``; None when unsupported."""
    if not isinstance(code, str):
        return None
    base = code.strip().lower().replace("_", "-").split("-")[0]
    return base if base in SUPPORTED else None


def resolve_language(config: Any) -> str:
    """The UI language for ``config``: ui_language > language > English."""
    for field in ("ui_language", "language"):
        code = normalize(getattr(config, field, None))
        if code:
            return code
    return FALLBACK


def set_language(code: Any) -> str:
    """Make ``code`` the active language (unsupported → English); returns it."""
    global _active
    _active = normalize(code) or FALLBACK
    return _active


def use_config(config: Any) -> str:
    """Pick the language from ``config`` — call once at every process entry."""
    return set_language(resolve_language(config))


def active() -> str:
    return _active


def t(key: str, **params: Any) -> Text:
    """``key`` in the active language, formatted with ``params``."""
    entry = MESSAGES.get(key)
    if entry is None:
        return Text(key, key)
    for lang in (_active, FALLBACK):
        template = entry.get(lang)
        if template is None:
            continue
        try:
            return Text(template.format(**params), key)
        except (KeyError, IndexError, ValueError):
            continue
    return Text(key, key)


def placeholders(template: str) -> set[str]:
    """Names used as ``{name}`` in ``template`` (tests only)."""
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


# key -> language -> template. Keep the four languages together per key so a
# missing translation is visible at a glance.
MESSAGES: dict[str, dict[str, str]] = {
    # ------------------------------------------------------------ bootstrap
    "boot.takeover_failed": {
        "en": "Another EchokrauTTS bootstrap (pid {pid}) is running in this folder and could not be stopped. End it in Task Manager and try again.",
        "de": "Ein anderer EchokrauTTS-Bootstrap (pid {pid}) läuft in diesem Ordner und ließ sich nicht beenden. Beende ihn im Task-Manager und versuche es erneut.",
        "fr": "Un autre bootstrap EchokrauTTS (pid {pid}) tourne dans ce dossier et n'a pas pu être arrêté. Terminez-le dans le Gestionnaire des tâches et réessayez.",
        "ja": "別の EchokrauTTS ブートストラップ（pid {pid}）がこのフォルダーで実行中で、停止できませんでした。タスクマネージャーで終了してから再試行してください。",
    },
    "boot.took_over": {
        "en": "Another bootstrap (pid {pid}) was still running in this folder — stopped it, this run takes over",
        "de": "Ein anderer Bootstrap (pid {pid}) lief noch in diesem Ordner — beendet, dieser Lauf übernimmt",
        "fr": "Un autre bootstrap (pid {pid}) tournait encore dans ce dossier — arrêté, cette exécution prend le relais",
        "ja": "別のブートストラップ（pid {pid}）がこのフォルダーでまだ実行中でした — 停止し、この実行が引き継ぎます",
    },
    "boot.stale_stopped": {
        "en": "Stopping {count} leftover process(es) of the installation (pid {pids})",
        "de": "Beende {count} verwaiste(n) Prozess(e) aus der Installation (pid {pids})",
        "fr": "Arrêt de {count} processus orphelin(s) de l'installation (pid {pids})",
        "ja": "インストールに残っていたプロセスを {count} 個終了します（pid {pids}）",
    },
    "boot.stale_failed": {
        "en": "Cleaning up leftover processes failed: {error}",
        "de": "Aufräumen verwaister Prozesse fehlgeschlagen: {error}",
        "fr": "Le nettoyage des processus orphelins a échoué : {error}",
        "ja": "残存プロセスの後片付けに失敗しました: {error}",
    },
    "boot.download_failed": {
        "en": "Download failed: {url} ({error})",
        "de": "Download fehlgeschlagen: {url} ({error})",
        "fr": "Échec du téléchargement : {url} ({error})",
        "ja": "ダウンロードに失敗しました: {url}（{error}）",
    },
    "boot.unsupported_platform": {
        "en": "Unsupported platform: {platform} (Windows/Linux only)",
        "de": "Nicht unterstützte Plattform: {platform} (nur Windows/Linux)",
        "fr": "Plateforme non prise en charge : {platform} (Windows/Linux uniquement)",
        "ja": "対応していないプラットフォームです: {platform}（Windows/Linux のみ）",
    },
    "boot.venv_locked": {
        "en": "The venv folder is locked and could not be replaced. Usually an EchokrauTTS server is still running or a virus scanner holds the files. Stop the running server (or restart the PC) and try again. Original error: {error}",
        "de": "Das venv-Verzeichnis ist gesperrt und konnte nicht ersetzt werden. Meist läuft noch ein EchokrauTTS-Server oder ein Virenscanner hält die Dateien. Beende den laufenden Server (oder starte den Rechner neu) und versuche es erneut. Ursprungsfehler: {error}",
        "fr": "Le dossier venv est verrouillé et n'a pas pu être remplacé. En général, un serveur EchokrauTTS tourne encore ou un antivirus bloque les fichiers. Arrêtez le serveur (ou redémarrez le PC) et réessayez. Erreur d'origine : {error}",
        "ja": "venv フォルダーがロックされていて置き換えられませんでした。多くの場合、EchokrauTTS サーバーがまだ動作中か、ウイルス対策ソフトがファイルを掴んでいます。サーバーを停止して（または PC を再起動して）再試行してください。元のエラー: {error}",
    },
    "boot.install_extend": {
        "en": "Adding to the installation ({names})",
        "de": "Installation wird ergänzt ({names})",
        "fr": "Complément de l'installation ({names})",
        "ja": "インストールを追加しています（{names}）",
    },
    "boot.install_rebuild": {
        "en": "Existing installation is unusable ({problem}) — rebuilding it",
        "de": "Vorhandene Installation unbrauchbar ({problem}) — wird neu gebaut",
        "fr": "L'installation existante est inutilisable ({problem}) — reconstruction",
        "ja": "既存のインストールは使用できません（{problem}）— 再構築します",
    },
    "boot.model_failed": {
        "en": "{name} model download failed (see stderr/log)",
        "de": "{name}-Modell-Download fehlgeschlagen (siehe stderr/Log)",
        "fr": "Échec du téléchargement du modèle {name} (voir stderr/journal)",
        "ja": "{name} モデルのダウンロードに失敗しました（stderr/ログを参照）",
    },
    "boot.extra_model_failed": {
        "en": "Additional download {module} failed — {name} keeps running, but slower",
        "de": "Zusatz-Download {module} fehlgeschlagen — {name} läuft langsamer weiter",
        "fr": "Le téléchargement supplémentaire {module} a échoué — {name} continue, mais plus lentement",
        "ja": "追加ダウンロード {module} に失敗しました — {name} は動作しますが遅くなります",
    },
    "boot.unexpected": {
        "en": "Unexpected bootstrap error: {error}",
        "de": "Unerwarteter Bootstrap-Fehler: {error}",
        "fr": "Erreur inattendue du bootstrap : {error}",
        "ja": "ブートストラップで予期しないエラーが発生しました: {error}",
    },
    "boot.unsupported_device": {
        "en": "{name} needs an NVIDIA graphics card (AMD with ROCm is experimental); detected '{backend}'. Choose XTTS, F5 or MOSS.",
        "de": "{name} braucht eine NVIDIA-Grafikkarte (AMD mit ROCm experimentell); erkannt wurde '{backend}'. Wähle XTTS, F5 oder MOSS.",
        "fr": "{name} nécessite une carte graphique NVIDIA (AMD avec ROCm est expérimental) ; détecté : '{backend}'. Choisissez XTTS, F5 ou MOSS.",
        "ja": "{name} には NVIDIA のグラフィックカードが必要です（AMD + ROCm は実験的）。検出結果: '{backend}'。XTTS、F5、MOSS のいずれかを選んでください。",
    },
    # ------------------------------------------------------- bootstrap steps
    "step.uv_present": {
        "en": "uv present", "de": "uv vorhanden", "fr": "uv présent", "ja": "uv は導入済み",
    },
    "step.uv_from_path": {
        "en": "uv taken from PATH", "de": "uv von PATH übernommen",
        "fr": "uv repris du PATH", "ja": "PATH 上の uv を使用",
    },
    "step.uv_fetching": {
        "en": "Getting uv …", "de": "Beschaffe uv …", "fr": "Récupération de uv …", "ja": "uv を取得中 …",
    },
    "step.uv_downloading": {
        "en": "Downloading uv …", "de": "Lade uv …", "fr": "Téléchargement de uv …", "ja": "uv をダウンロード中 …",
    },
    "step.uv_installed": {
        "en": "uv installed", "de": "uv installiert", "fr": "uv installé", "ja": "uv をインストールしました",
    },
    "step.python_present": {
        "en": "Python {version}", "de": "Python {version}", "fr": "Python {version}", "ja": "Python {version}",
    },
    "step.python_installing": {
        "en": "Installing Python {version} …", "de": "Installiere Python {version} …",
        "fr": "Installation de Python {version} …", "ja": "Python {version} をインストール中 …",
    },
    "step.python_ready": {
        "en": "Python {version} ready", "de": "Python {version} bereit",
        "fr": "Python {version} prêt", "ja": "Python {version} の準備完了",
    },
    "step.detecting": {
        "en": "Detecting GPU/backend …", "de": "Erkenne GPU/Backend …",
        "fr": "Détection du GPU/backend …", "ja": "GPU/バックエンドを検出中 …",
    },
    "step.venv_locked_wait": {
        "en": "venv folder locked, waiting {delay}s (attempt {attempt}/{attempts} failed) …",
        "de": "venv-Verzeichnis gesperrt, warte {delay}s (Versuch {attempt}/{attempts} fehlgeschlagen) …",
        "fr": "Dossier venv verrouillé, attente de {delay} s (tentative {attempt}/{attempts} échouée) …",
        "ja": "venv フォルダーがロック中、{delay} 秒待機します（試行 {attempt}/{attempts} 失敗）…",
    },
    "step.rocm_runtime": {
        "en": "Installing ROCm runtime (Windows) …", "de": "Installiere ROCm-Laufzeit (Windows) …",
        "fr": "Installation du runtime ROCm (Windows) …", "ja": "ROCm ランタイム（Windows）をインストール中 …",
    },
    "step.torch": {
        "en": "Installing PyTorch ({backend}) …", "de": "Installiere PyTorch ({backend}) …",
        "fr": "Installation de PyTorch ({backend}) …", "ja": "PyTorch（{backend}）をインストール中 …",
    },
    "step.wrapper_deps": {
        "en": "Installing wrapper dependencies …", "de": "Installiere Wrapper-Abhängigkeiten …",
        "fr": "Installation des dépendances du wrapper …", "ja": "ラッパーの依存関係をインストール中 …",
    },
    "step.installing": {
        "en": "Installing {name} …", "de": "Installiere {name} …",
        "fr": "Installation de {name} …", "ja": "{name} をインストール中 …",
    },
    "step.torchcodec_remove": {
        "en": "Removing unused torchcodec …", "de": "Entferne ungenutztes torchcodec …",
        "fr": "Suppression de torchcodec inutilisé …", "ja": "未使用の torchcodec を削除中 …",
    },
    "step.deps_present": {
        "en": "Dependencies present", "de": "Abhängigkeiten vorhanden",
        "fr": "Dépendances présentes", "ja": "依存関係は導入済み",
    },
    "step.deps_extended": {
        "en": "Dependencies added", "de": "Abhängigkeiten ergänzt",
        "fr": "Dépendances complétées", "ja": "依存関係を追加しました",
    },
    "step.venv_create": {
        "en": "Creating venv …", "de": "Erstelle venv …", "fr": "Création du venv …", "ja": "venv を作成中 …",
    },
    "step.deps_installed": {
        "en": "Dependencies installed", "de": "Abhängigkeiten installiert",
        "fr": "Dépendances installées", "ja": "依存関係をインストールしました",
    },
    "step.models_present": {
        "en": "Voice models present", "de": "Sprachmodelle vorhanden",
        "fr": "Modèles vocaux présents", "ja": "音声モデルは導入済み",
    },
    "step.model_loading": {
        "en": "Downloading {name} model …", "de": "Lade {name}-Modell …",
        "fr": "Téléchargement du modèle {name} …", "ja": "{name} モデルをダウンロード中 …",
    },
    "step.extra_model": {
        "en": "Downloading additional model ({module}) …", "de": "Lade Zusatzmodell ({module}) …",
        "fr": "Téléchargement du modèle supplémentaire ({module}) …", "ja": "追加モデル（{module}）をダウンロード中 …",
    },
    "step.models_loaded": {
        "en": "Voice models downloaded", "de": "Sprachmodelle geladen",
        "fr": "Modèles vocaux téléchargés", "ja": "音声モデルのダウンロード完了",
    },
    "step.server_start": {
        "en": "Starting server on {host}:{port} …", "de": "Starte Server auf {host}:{port} …",
        "fr": "Démarrage du serveur sur {host}:{port} …", "ja": "{host}:{port} でサーバーを起動中 …",
    },
    # ---------------------------------------------------------------- models
    "model.custom": {
        "en": "Using custom {engine} model: {name}", "de": "Nutze eigenes {engine}-Modell: {name}",
        "fr": "Utilisation du modèle {engine} personnalisé : {name}", "ja": "カスタム {engine} モデルを使用: {name}",
    },
    "model.loading": {
        "en": "Downloading {name} …", "de": "Lade {name} …",
        "fr": "Téléchargement de {name} …", "ja": "{name} をダウンロード中 …",
    },
    "model.ready": {
        "en": "{name} ready", "de": "{name} bereit", "fr": "{name} prêt", "ja": "{name} の準備完了",
    },
    "model.download_failed": {
        "en": "{name} model download failed: {error}", "de": "{name}-Modell-Download fehlgeschlagen: {error}",
        "fr": "Échec du téléchargement du modèle {name} : {error}", "ja": "{name} モデルのダウンロードに失敗しました: {error}",
    },
    # ---------------------------------------------------------------- engine
    "engine.loaded_on": {
        "en": "{name} loaded on {device}", "de": "{name} geladen auf {device}",
        "fr": "{name} chargé sur {device}", "ja": "{name} を {device} に読み込みました",
    },
    "engine.loaded_threads": {
        "en": "{name} loaded, {threads} threads", "de": "{name} geladen, {threads} Threads",
        "fr": "{name} chargé, {threads} threads", "ja": "{name} を読み込みました（{threads} スレッド）",
    },
    "engine.warmup_failed": {
        "en": "{name}: warm-up failed ({error}) — the first line will be slower",
        "de": "{name}: Aufwärmen fehlgeschlagen ({error}) — erste Zeile wird langsamer",
        "fr": "{name} : échec du préchauffage ({error}) — la première réplique sera plus lente",
        "ja": "{name}: ウォームアップに失敗しました（{error}）— 最初のセリフは遅くなります",
    },
    "engine.speed_ignored": {
        "en": "{name} has no speed parameter — the value is ignored",
        "de": "{name} kennt keinen speed-Parameter — der Wert wird ignoriert",
        "fr": "{name} n'a pas de paramètre speed — la valeur est ignorée",
        "ja": "{name} には speed パラメーターがありません — 値は無視されます",
    },
    "engine.device_fallback": {
        "en": "{device}/{backend}: {reason}; worker {worker} falls back to the CPU",
        "de": "{device}/{backend}: {reason}; Worker {worker} weicht auf die CPU aus",
        "fr": "{device}/{backend} : {reason} ; le worker {worker} se replie sur le CPU",
        "ja": "{device}/{backend}: {reason}。ワーカー {worker} は CPU に切り替えます",
    },
    "engine.gpu_only_failed": {
        "en": "{name} could not start on {device}/{backend} ({reason}). This engine does not run on the CPU — choose XTTS, F5 or MOSS.",
        "de": "{name} konnte auf {device}/{backend} nicht starten ({reason}). Diese Engine läuft nicht auf der CPU — wähle XTTS, F5 oder MOSS.",
        "fr": "{name} n'a pas pu démarrer sur {device}/{backend} ({reason}). Ce moteur ne fonctionne pas sur le CPU — choisissez XTTS, F5 ou MOSS.",
        "ja": "{name} を {device}/{backend} で起動できませんでした（{reason}）。このエンジンは CPU では動作しません — XTTS、F5、MOSS のいずれかを選んでください。",
    },
    "engine.runs_on": {
        "en": "{backend} runs on {effective} although {device} was detected",
        "de": "{backend} läuft auf {effective}, obwohl {device} erkannt wurde",
        "fr": "{backend} tourne sur {effective} alors que {device} a été détecté",
        "ja": "{device} が検出されましたが、{backend} は {effective} で動作します",
    },
    "engine.fatal_device": {
        "en": "Fatal device error, the GPU context of this process is unusable — exiting with code {code} so the host restarts the wrapper: {error}",
        "de": "Fataler Gerätefehler, der GPU-Kontext dieses Prozesses ist unbrauchbar — Ende mit Code {code}, damit der Host den Wrapper neu startet: {error}",
        "fr": "Erreur fatale du périphérique, le contexte GPU de ce processus est inutilisable — arrêt avec le code {code} pour que l'hôte redémarre le wrapper : {error}",
        "ja": "致命的なデバイスエラー: このプロセスの GPU コンテキストは使用できません — ホストがラッパーを再起動できるようコード {code} で終了します: {error}",
    },
    "engine.moss_onnx_fallback": {
        "en": "MOSS: ONNX not available ({reason}) — using PyTorch, which is about 2.5x slower",
        "de": "MOSS: ONNX nicht verfügbar ({reason}) — nutze PyTorch, das ist etwa 2,5x langsamer",
        "fr": "MOSS : ONNX indisponible ({reason}) — utilisation de PyTorch, environ 2,5x plus lent",
        "ja": "MOSS: ONNX が利用できません（{reason}）— PyTorch を使用します（約 2.5 倍遅くなります）",
    },
    "moss.scratch_not_removed": {
        "en": "MOSS: temporary file not removed ({error})",
        "de": "MOSS: Zwischendatei nicht entfernt ({error})",
        "fr": "MOSS : fichier temporaire non supprimé ({error})",
        "ja": "MOSS: 一時ファイルを削除できませんでした（{error}）",
    },
    "qwen.no_ref_text": {
        "en": "Qwen: reference without transcript — using only the speaker embedding (sounds less like the reference)",
        "de": "Qwen: Referenz ohne Transkript — nutze nur das Sprecher-Embedding (klingt weniger nach der Referenz)",
        "fr": "Qwen : référence sans transcription — seul l'embedding du locuteur est utilisé (ressemble moins à la référence)",
        "ja": "Qwen: 文字起こしのないリファレンス — 話者埋め込みのみを使用します（リファレンスに似にくくなります）",
    },
    "pocket.decode_steps_raised": {
        "en": "Pocket 6 layers: decode_steps 1 is not supported, using 4",
        "de": "Pocket 6 Layer: decode_steps 1 wird nicht unterstützt, nutze 4",
        "fr": "Pocket 6 couches : decode_steps 1 n'est pas pris en charge, utilisation de 4",
        "ja": "Pocket 6 レイヤー: decode_steps 1 は非対応のため 4 を使用します",
    },
    "pocket.layers_fallback": {
        "en": "Pocket TTS has no model with {wanted} layers for {code}, using {size} layers",
        "de": "Pocket TTS hat für {code} kein Modell mit {wanted} Layern, nutze {size} Layer",
        "fr": "Pocket TTS n'a pas de modèle à {wanted} couches pour {code}, utilisation de {size} couches",
        "ja": "Pocket TTS には {code} 用の {wanted} レイヤーモデルがないため、{size} レイヤーを使用します",
    },
    # ---------------------------------------------------------------- server
    "server.port_in_use": {
        "en": "Port {host}:{port} is already in use — is another wrapper instance still running?",
        "de": "Port {host}:{port} ist schon belegt — läuft noch eine andere Wrapper-Instanz?",
        "fr": "Le port {host}:{port} est déjà utilisé — une autre instance du wrapper tourne-t-elle encore ?",
        "ja": "ポート {host}:{port} は既に使用中です — 別のラッパーがまだ動作していませんか？",
    },
    # ------------------------------------------------------------- voicepack
    "voicepack.searching": {
        "en": "Looking for the current voice pack …", "de": "Suche aktuelles Voice-Pack …",
        "fr": "Recherche du pack de voix actuel …", "ja": "最新のボイスパックを検索中 …",
    },
    "voicepack.downloading": {
        "en": "Downloading voice pack …", "de": "Lade Voice-Pack …",
        "fr": "Téléchargement du pack de voix …", "ja": "ボイスパックをダウンロード中 …",
    },
    "voicepack.downloading_tag": {
        "en": "Downloading {tag} ({mb} MB) …", "de": "Lade {tag} ({mb} MB) …",
        "fr": "Téléchargement de {tag} ({mb} Mo) …", "ja": "{tag}（{mb} MB）をダウンロード中 …",
    },
    "voicepack.installed": {
        "en": "Voice pack {tag} installed: {count} files in {path}",
        "de": "Voice-Pack {tag} installiert: {count} Dateien in {path}",
        "fr": "Pack de voix {tag} installé : {count} fichiers dans {path}",
        "ja": "ボイスパック {tag} をインストールしました: {path} に {count} ファイル",
    },
    "voicepack.not_installed": {
        "en": "Voice pack not installed ({error}); put your own samples in {path} — synthesis needs at least one",
        "de": "Voice-Pack nicht installiert ({error}); lege eigene Stimmproben in {path} ab — die Synthese braucht mindestens eine",
        "fr": "Pack de voix non installé ({error}) ; placez vos propres échantillons dans {path} — la synthèse en nécessite au moins un",
        "ja": "ボイスパックはインストールされていません（{error}）。{path} に独自のサンプルを置いてください — 合成には最低 1 つ必要です",
    },
}
