from src.jobs import CANCELLED, DONE, RUNNING, JobRegistry


def test_create_and_get():
    reg = JobRegistry()
    job = reg.create()
    assert reg.get(job.job_id) is job
    assert reg.get("nope") is None


def test_cancel_unknown_returns_false():
    reg = JobRegistry()
    assert reg.cancel("missing") is False


def test_cancel_sets_event():
    reg = JobRegistry()
    job = reg.create()
    assert job.cancelled is False
    assert reg.cancel(job.job_id) is True
    assert job.cancelled is True


def test_progress_and_percent():
    reg = JobRegistry()
    job = reg.create()
    assert job.percent is None  # total unknown
    reg.set_total(job, 4)
    reg.mark_running(job, worker_id=123)
    assert job.state == RUNNING
    assert job.worker_id == 123
    reg.advance(job)
    reg.advance(job)
    assert job.sentences_done == 2
    assert job.percent == 50
    snap = job.snapshot()
    assert snap == {
        "state": RUNNING,
        "sentences_total": 4,
        "sentences_done": 2,
        "percent": 50,
    }


def test_finish_states():
    reg = JobRegistry()
    job = reg.create()
    reg.finish(job, DONE)
    assert job.state == DONE
    job2 = reg.create()
    reg.finish(job2, CANCELLED)
    assert job2.state == CANCELLED


def test_cancel_all():
    reg = JobRegistry()
    a, b = reg.create(), reg.create()
    reg.cancel_all()
    assert a.cancelled and b.cancelled
