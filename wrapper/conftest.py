"""Pytest configuration and shared fixtures.

Ensures the wrapper root is importable (so ``import src...`` works) and provides
fake workers / engine builders so the suite runs without torch or f5-tts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

WRAPPER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WRAPPER_ROOT))

from src.config import Config  # noqa: E402
from src.engine import Engine  # noqa: E402
from src.gpu_detect import Detection  # noqa: E402
from src.jobs import JobRegistry  # noqa: E402


class FakeWorker:
    """A stand-in for F5TTSWorker that yields a fixed silent clip."""

    def __init__(self, sample_rate: int = 24000, fail: bool = False):
        self.sample_rate = sample_rate
        self.device = "cpu"
        self._fail = fail
        self.infer_calls = 0

    def infer(self, ref_file, ref_text, gen_text, nfe_step, speed):
        self.infer_calls += 1
        if self._fail:
            raise RuntimeError("simulated CUDA OOM")
        # 50 ms of silence, length proportional to text so output varies.
        n = max(1, int(0.05 * self.sample_rate))
        return np.zeros(n, dtype=np.float32)

    def transcribe(self, audio_path):
        return "fake reference transcript"

    def self_test(self):
        return True


def cpu_detection(max_workers_hint: int = 1) -> Detection:
    return Detection(
        backend="cpu",
        device="cpu",
        torch_index_url="https://download.pytorch.org/whl/cpu",
        max_workers_hint=max_workers_hint,
    )


@pytest.fixture
def samples_dir(tmp_path: Path) -> Path:
    d = tmp_path / "samples"
    d.mkdir()
    (d / "anna_de.wav").write_bytes(b"RIFFfake")
    return d


@pytest.fixture
def config(tmp_path: Path, samples_dir: Path) -> Config:
    return Config(
        samples_dir=str(samples_dir),
        models_dir=str(tmp_path / "models"),
        max_workers=1,
        max_queue=8,
    )


def make_engine(config: Config, *, fail: bool = False, workers: int = 1) -> Engine:
    jobs = JobRegistry()
    det = cpu_detection(max_workers_hint=workers)
    return Engine(
        config,
        det,
        jobs,
        worker_factory=lambda i, d: FakeWorker(fail=fail),
    )
