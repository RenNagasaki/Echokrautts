"""The RIFF header used by the /tts wav mode (src/wav.py)."""

import io
import struct
import wave

import pytest

from src import wav


def test_header_is_44_bytes():
    assert len(wav.wav_header(0, 24000)) == wav.RIFF_HEADER_SIZE


def test_stdlib_wave_can_read_what_we_write():
    # The real contract: a WAVE parser must accept it. 1000 frames of silence.
    pcm = b"\x00\x00" * 1000
    blob = wav.wrap_pcm(pcm, sample_rate=24000)
    with wave.open(io.BytesIO(blob)) as fh:
        assert fh.getnchannels() == 1
        assert fh.getsampwidth() == 2
        assert fh.getframerate() == 24000
        assert fh.getnframes() == 1000
        assert fh.readframes(1000) == pcm


def test_sizes_are_consistent():
    pcm = b"\x01\x02" * 12
    blob = wav.wrap_pcm(pcm, sample_rate=24000)
    riff_size = struct.unpack("<I", blob[4:8])[0]
    data_size = struct.unpack("<I", blob[40:44])[0]
    assert data_size == len(pcm)
    # RIFF size counts everything after the size field itself.
    assert riff_size == len(blob) - 8


def test_byte_rate_and_block_align_follow_the_format():
    blob = wav.wrap_pcm(b"", sample_rate=48000, channels=2, bits_per_sample=16)
    byte_rate = struct.unpack("<I", blob[28:32])[0]
    block_align = struct.unpack("<H", blob[32:34])[0]
    assert block_align == 4  # 2 channels * 16 bit
    assert byte_rate == 48000 * 4


def test_empty_pcm_is_a_valid_header_only_file():
    blob = wav.wrap_pcm(b"", sample_rate=24000)
    assert len(blob) == wav.RIFF_HEADER_SIZE
    with wave.open(io.BytesIO(blob)) as fh:
        assert fh.getnframes() == 0


def test_negative_size_rejected():
    with pytest.raises(ValueError):
        wav.wav_header(-1, 24000)
