"""Job registry and cancellation tokens (SPEC §5.4, §9).

Each ``/tts`` request creates a :class:`Job` tracked here so that:
- ``GET /jobs/{id}`` can report live progress while synthesis runs,
- ``POST /cancel/{id}`` can flip a cancel token the chunk generator checks
  between sentences.

The registry is thread-safe because the worker pool mutates job state from
executor threads while the asyncio event loop reads it for HTTP responses.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Optional

# Valid lifecycle states (SPEC §5.4).
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
CANCELLED = "cancelled"
ERROR = "error"


@dataclass
class Job:
    job_id: str
    state: str = QUEUED
    sentences_total: Optional[int] = None
    sentences_done: int = 0
    worker_id: Optional[int] = None
    error: Optional[str] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    @property
    def percent(self) -> Optional[int]:
        if not self.sentences_total:
            return None
        return int(round(100 * self.sentences_done / self.sentences_total))

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "sentences_total": self.sentences_total,
            "sentences_done": self.sentences_done,
            "percent": self.percent,
        }


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        job = Job(job_id=str(uuid.uuid4()))
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Set the cancel token. Returns False if the job is unknown (→ 404)."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return False
        job.cancel_event.set()
        return True

    def cancel_all(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            job.cancel_event.set()

    def set_total(self, job: Job, total: int) -> None:
        with self._lock:
            job.sentences_total = total

    def mark_running(self, job: Job, worker_id: int) -> None:
        with self._lock:
            job.state = RUNNING
            job.worker_id = worker_id

    def advance(self, job: Job) -> None:
        with self._lock:
            job.sentences_done += 1

    def finish(self, job: Job, state: str, error: Optional[str] = None) -> None:
        with self._lock:
            job.state = state
            if error:
                job.error = error

    def remove(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)
