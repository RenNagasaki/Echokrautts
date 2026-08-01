import json

from src import ndjson


def test_every_event_carries_a_timestamp(capsys):
    ndjson.log("hello")
    ndjson.ready("127.0.0.1", 8765, "cpu", "cpu", 1)
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.strip()]

    assert len(lines) == 2
    for obj in lines:
        # UTC ISO-8601 with millisecond precision, e.g. 2026-07-04T12:34:56.789+00:00
        assert "ts" in obj and obj["ts"].endswith("+00:00")


def test_log_once_emits_a_repeated_message_only_once(capsys):
    ndjson.reset_log_once()
    for _ in range(4):  # e.g. one worker each, all loading the same model
        ndjson.log_once("using custom model")
    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]

    assert len(lines) == 1
    assert json.loads(lines[0])["message"] == "using custom model"


def test_log_once_distinguishes_message_and_level(capsys):
    ndjson.reset_log_once()
    ndjson.log_once("a")
    ndjson.log_once("b")
    ndjson.log_once("a", level="warning")
    ndjson.log_once("a")  # already seen at info level
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.strip()]

    assert [(o["message"], o["level"]) for o in lines] == [
        ("a", "info"),
        ("b", "info"),
        ("a", "warning"),
    ]


def test_reset_log_once_allows_the_message_again(capsys):
    ndjson.reset_log_once()
    ndjson.log_once("x")
    ndjson.reset_log_once()
    ndjson.log_once("x")

    assert len(capsys.readouterr().out.strip().splitlines()) == 2


def test_explicit_ts_is_not_overwritten(capsys):
    ndjson._write({"event": "custom", "ts": "sentinel"})
    obj = json.loads(capsys.readouterr().out.strip())
    assert obj["ts"] == "sentinel"
