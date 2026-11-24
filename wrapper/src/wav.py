"""Wrap raw PCM in a RIFF/WAVE container.

The engine produces headerless ``pcm_s16le`` because that is what the plugin
wants to stream. Browsers cannot play that, so the ``format: "wav"`` mode of
``/tts`` (and the web UI behind it) needs a header.

Written by hand rather than via :mod:`wave` + ``BytesIO``: the header is 44
fixed bytes, and doing it here keeps the sizes explicit and unit-testable
without a temp file.
"""

from __future__ import annotations

import struct

RIFF_HEADER_SIZE = 44


def wav_header(
    data_bytes: int, sample_rate: int, channels: int = 1, bits_per_sample: int = 16
) -> bytes:
    """Return the 44-byte canonical WAVE header for ``data_bytes`` of PCM."""
    if data_bytes < 0:
        raise ValueError("data_bytes must not be negative")
    block_align = channels * bits_per_sample // 8
    byte_rate = sample_rate * block_align
    return b"".join(
        (
            b"RIFF",
            # Everything after this field: 36 header bytes + the payload.
            struct.pack("<I", 36 + data_bytes),
            b"WAVEfmt ",
            struct.pack("<I", 16),  # PCM fmt chunk size
            struct.pack("<H", 1),  # format 1 = uncompressed PCM
            struct.pack("<H", channels),
            struct.pack("<I", sample_rate),
            struct.pack("<I", byte_rate),
            struct.pack("<H", block_align),
            struct.pack("<H", bits_per_sample),
            b"data",
            struct.pack("<I", data_bytes),
        )
    )


def wrap_pcm(
    pcm: bytes, sample_rate: int, channels: int = 1, bits_per_sample: int = 16
) -> bytes:
    """Prepend a WAVE header to raw little-endian PCM."""
    return wav_header(len(pcm), sample_rate, channels, bits_per_sample) + pcm
