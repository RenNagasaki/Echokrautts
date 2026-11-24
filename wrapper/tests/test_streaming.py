import pytest

from src.streaming import chunk_text


def test_empty_returns_empty():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_single_sentence():
    assert chunk_text("Hallo Welt.") == ["Hallo Welt."]


def test_merges_short_sentences():
    out = chunk_text("Eins. Zwei. Drei.", max_chars=100)
    assert out == ["Eins. Zwei. Drei."]


def test_splits_when_exceeding_max():
    out = chunk_text("Eins. Zwei. Drei.", max_chars=10)
    assert out == ["Eins.", "Zwei.", "Drei."]
    assert all(len(c) <= 10 for c in out)


def test_hard_split_overlong_sentence_prefers_clause_boundary():
    text = "Dies ist ein sehr langer Satz, der weit über das Limit hinausgeht ohne Punkt"
    out = chunk_text(text, max_chars=30)
    assert all(len(c) <= 30 for c in out)
    assert " ".join(out).replace("  ", " ") .startswith("Dies ist ein")


def test_no_punctuation_long_text_is_split():
    text = "wort " * 50
    out = chunk_text(text, max_chars=40)
    assert len(out) > 1
    assert all(len(c) <= 40 for c in out)


def test_unicode_german_punctuation():
    out = chunk_text("Wie geht's? Mir geht es gut! Und dir…", max_chars=15)
    assert out[0].startswith("Wie geht's?")
    assert all(len(c) <= 15 for c in out)


def test_invalid_max_chars():
    with pytest.raises(ValueError):
        chunk_text("x", max_chars=0)
