import threading

import pytest

from clipper.pipeline import Steps
from clipper.run import Run
from clipper.web.jobs import JobBusy, JobRunner

SETTINGS = {"device": "auto", "model": "small", "diarize": False, "prompt": "funny bits",
            "top_n": 5}
MAKE_SETTINGS = {"style": {"layout": "fit"}}


def recorder_steps(log, fail_at=None, gate=None):
    """Fake steps that record (name, args) and optionally fail or block."""
    def step(name, value=None):
        def fn(*args):
            if gate is not None and name in ("ingest", "style"):
                gate.wait(5)
            log.append((name, args))
            if name == fail_at:
                raise RuntimeError(f"{name} exploded")
            return value
        return fn

    return Steps(
        ingest=step("ingest"),
        transcribe=step("transcribe"),
        segment=step("segment", {"content_type": "tutorial", "ai": {"provider": "anthropic"},
                                 "ai_error": None, "candidates": [1, 2, 3]}),
        rate=step("rate", {"scored": 3, "failed": 0, "device": "xpu"}),
        action=step("action", {"silent_seconds": 90.0, "spans": 2, "candidates": 1}),
        select=step("select", {"clips": [{"include": True}, {"include": False}]}),
        style=step("style", {"layout": "fit"}),
        fillin=step("fillin", {"c0": {"title": "T"}}),
        edit=step("edit", [{"id": "c0", "final": "clips/01-t/final.mp4"}]),
    )


@pytest.fixture
def run(tmp_path):
    r = Run.create(tmp_path, "ep")
    r.path("video.mp4").write_bytes(b"x")
    return r


def _analyze(runner, run, **settings):
    runner.start(run, "analyze", {**SETTINGS, **settings}, video=run.path("video.mp4"))
    runner.wait(5)
    return runner.status()


def test_a_failure_stops_later_steps_and_records_the_error(run):
    log = []
    status = _analyze(JobRunner(recorder_steps(log, fail_at="transcribe")), run)
    assert status["state"] == "failed" and status["failed_stage"] == "transcribe"
    assert status["stages"]["transcribe"] == "failed"
    assert status["stages"]["segment"] == "pending"
    assert "transcribe exploded" in status["error"]
    assert "rate" not in [name for name, _ in log]
    assert run.read_json("job.json")["failed_stage"] == "transcribe"


def test_make_needs_an_analyzed_run(run):
    with pytest.raises(ValueError, match="Analyze"):
        JobRunner(recorder_steps([])).start(run, "make", MAKE_SETTINGS)


def test_the_job_is_busy_while_running(run):
    gate = threading.Event()
    runner = JobRunner(recorder_steps([], gate=gate))
    runner.start(run, "analyze", SETTINGS, video=run.path("video.mp4"))
    try:
        assert runner.busy() is True
        assert runner.status()["stages"]["upload"] == "done"
        with pytest.raises(JobBusy):
            runner.start(run, "analyze", SETTINGS, video=run.path("video.mp4"))
    finally:
        gate.set()
        runner.wait(5)
    assert runner.busy() is False
    assert _analyze(runner, run)["state"] == "done"


def test_a_persist_failure_while_finishing_does_not_leave_the_runner_stuck(run, monkeypatch):
    real_write_json = run.write_json

    def flaky(name, data):
        if name == "job.json" and data.get("state") == "done":
            raise OSError("disk full")
        real_write_json(name, data)

    monkeypatch.setattr(run, "write_json", flaky)
    runner = JobRunner(recorder_steps([]))
    status = _analyze(runner, run)
    assert status["state"] == "failed" and "disk full" in status["error"]
    assert runner.busy() is False


def test_a_failure_building_the_real_steps_ends_the_job_failed(run, monkeypatch):
    import clipper.web.jobs as jobs_module

    def boom():
        raise ImportError("torch missing")

    monkeypatch.setattr(jobs_module, "default_steps", boom)
    status = _analyze(JobRunner(), run)
    assert status["state"] == "failed" and "torch missing" in status["error"]
    assert status["failed_stage"] is None


def test_a_start_time_write_failure_reraises_and_leaves_the_runner_idle(run, monkeypatch):
    monkeypatch.setattr(run, "write_json", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    runner = JobRunner(recorder_steps([]))
    with pytest.raises(OSError):
        runner.start(run, "analyze", SETTINGS, video=run.path("video.mp4"))
    assert runner.status() == {"state": "idle"}


def test_status_reports_progress_while_a_step_runs(run):
    seen = {}
    steps = recorder_steps([])

    def rate(r, settings, progress):
        progress(2, 5)
        seen["status"] = runner.status()
        return {"scored": 5, "failed": 0, "device": "cpu"}

    steps.rate = rate
    runner = JobRunner(steps)
    _analyze(runner, run)
    assert seen["status"]["progress"] == {"stage": "rate", "done": 2, "total": 5}
