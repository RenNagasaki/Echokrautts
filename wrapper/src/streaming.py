"""Sentence chunking for streaming synthesis (SPEC §6).

F5-TTS has no token-wise streaming: each ``infer()`` call produces a complete
clip. To stream, we split the input text into sentence-sized chunks, synthesize
them sequentially, and yield each chunk's PCM as soon as it is ready (the server
does the yielding; this module only does the *splitting*).

Pure, synchronous, and model-free so it is fully unit-testable.
"""

from __future__ import annotations

import re

# Sentence-ending punctuation, including CJK/full-width forms. We split *after*
# the punctuation (and any trailing closing quotes/brackets) so the delimiter
# stays attached to its sentence.
_SENTENCE_END = re.compile(
    r"""(?<=[.!?。！？…])      # a sentence-ending mark
        ['")\]}»”’]*           # optional trailing closers
        \s+                    # whitespace separates the next sentence
    """,
    re.VERBOSE,
)


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Split an over-long sentence on soft boundaries, then hard if needed.

    Tries to break on the last comma/semicolon/space before ``max_chars`` to
    keep prosody sane; falls back to a hard character cut if no boundary exists.
    """
    out: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        # Prefer a clause boundary, then any whitespace, within the window.
        cut = max(
            window.rfind(", "),
            window.rfind("; "),
            window.rfind(": "),
            window.rfind(" — "),
        )
        if cut <= 0:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = max_chars  # no boundary at all → hard cut
        else:
            cut += 1  # keep the boundary char with the left part
        piece = remaining[:cut].strip()
        if piece:
            out.append(piece)
        remaining = remaining[cut:].strip()
    if remaining:
        out.append(remaining)
    return out


def chunk_text(text: str, max_chars: int = 250) -> list[str]:
    """Split ``text`` into synthesis chunks no longer than ``max_chars``.

    1. Normalize whitespace.
    2. Split on sentence boundaries.
    3. Greedily merge adjacent short sentences up to ``max_chars`` (fewer infer
       calls = less per-call overhead and fewer cross-fade seams).
    4. Hard-split any single sentence that still exceeds ``max_chars``.

    Returns an empty list for empty/whitespace-only input.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    sentences = [s.strip() for s in _SENTENCE_END.split(normalized) if s.strip()]
    if not sentences:
        sentences = [normalized]

    chunks: list[str] = []
    buffer = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            # Flush the buffer, then hard-split the long sentence on its own.
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(_hard_split(sentence, max_chars))
            continue

        if not buffer:
            buffer = sentence
        elif len(buffer) + 1 + len(sentence) <= max_chars:
            buffer = f"{buffer} {sentence}"
        else:
            chunks.append(buffer)
            buffer = sentence

    if buffer:
        chunks.append(buffer)
    return chunks
