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


def test_explicit_ts_is_not_overwritten(capsys):
    ndjson._write({"event": "custom", "ts": "sentinel"})
    obj = json.loads(capsys.readouterr().out.strip())
    assert obj["ts"] == "sentinel"
