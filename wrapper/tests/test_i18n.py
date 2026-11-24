"""User-facing NDJSON texts in the game client's language (src/i18n.py)."""

from __future__ import annotations

import json

import pytest

from src import i18n, ndjson
from src.config import Config, load_config


@pytest.fixture(autouse=True)
def _restore_language():
    before = i18n.active()
    yield
    i18n.set_language(before)


@pytest.mark.parametrize("key", sorted(i18n.MESSAGES))
def test_every_message_has_all_languages_with_the_same_placeholders(key):
    entry = i18n.MESSAGES[key]
    assert set(entry) == set(i18n.SUPPORTED), f"{key}: languages {sorted(entry)}"
    expected = i18n.placeholders(entry[i18n.FALLBACK])
    for lang, template in entry.items():
        assert template.strip(), f"{key}/{lang} is empty"
        assert i18n.placeholders(template) == expected, f"{key}/{lang}"


def test_text_follows_the_active_language():
    i18n.set_language("de")
    assert i18n.t("step.server_start", host="h", port=1) == "Starte Server auf h:1 …"
    i18n.set_language("en")
    assert i18n.t("step.server_start", host="h", port=1) == "Starting server on h:1 …"
    i18n.set_language("ja")
    assert "h:1" in i18n.t("step.server_start", host="h", port=1)


def test_unknown_language_falls_back_to_english():
    assert i18n.set_language("zh") == "en"
    assert i18n.t("step.venv_create") == "Creating venv …"


def test_broken_placeholder_falls_back_instead_of_raising(monkeypatch):
    monkeypatch.setitem(i18n.MESSAGES, "x.test", {"en": "ok {a}", "de": "kaputt {b}"})
    i18n.set_language("de")
    assert i18n.t("x.test", a=1) == "ok 1"
    # Even English broken: the key itself, never an exception.
    assert i18n.t("x.test") == "x.test"
    assert i18n.t("no.such.key") == "no.such.key"


def test_result_carries_its_key():
    text = i18n.t("step.uv_present")
    assert isinstance(text, str) and text.key == "step.uv_present"


@pytest.mark.parametrize("code, expected", [
    ("de", "de"), ("DE", "de"), ("fr-FR", "fr"), ("ja_JP", "ja"), ("zh", None), ("", None), (None, None),
])
def test_normalize(code, expected):
    assert i18n.normalize(code) == expected


def test_ui_language_wins_over_language():
    assert i18n.resolve_language(Config(language="de", ui_language="fr")) == "fr"


def test_empty_ui_language_follows_language():
    # The plugin passes the client language as `language` — that is the default.
    assert i18n.resolve_language(Config(language="ja", ui_language="")) == "ja"


def test_unsupported_everything_is_english():
    assert i18n.resolve_language(Config(language="xx", ui_language="yy")) == "en"


def test_load_config_selects_the_language():
    load_config(argv=["--language", "fr"], env={})
    assert i18n.active() == "fr"
    load_config(argv=["--ui-language", "en"], env={"F5W_LANGUAGE": "de"})
    assert i18n.active() == "en"
    load_config(argv=[], env={"F5W_UI_LANGUAGE": "ja"})
    assert i18n.active() == "ja"


def test_ndjson_writes_the_key_next_to_the_message(capsys):
    i18n.set_language("de")
    ndjson.log(i18n.t("step.deps_present"))
    ndjson.progress(1, 6, "deps", i18n.t("step.deps_installed"), done=True)
    ndjson.error(i18n.t("boot.unexpected", error="boom"), fatal=True)
    ndjson.log("plain developer line")
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert lines[0]["message"] == "Abhängigkeiten vorhanden"
    assert lines[0]["key"] == "step.deps_present"
    assert lines[1]["key"] == "step.deps_installed"
    assert lines[2]["key"] == "boot.unexpected" and "boom" in lines[2]["message"]
    assert "key" not in lines[3]


def test_speed_warning_is_localized(capsys):
    ndjson.reset_log_once()
    i18n.set_language("fr")
    ndjson.warn_ignored_speed("MOSS", 1.5)
    event = json.loads(capsys.readouterr().out)
    assert event["key"] == "engine.speed_ignored"
    assert "MOSS" in event["message"] and "ignorée" in event["message"]
    ndjson.reset_log_once()
