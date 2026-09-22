import threading

import pytest

from clipper.run import Run
from clipper.web.jobs import STAGES, JobBusy, JobRunner, Stages

SETTINGS = {"profile": "podcast", "device": "auto", "model": "small",
            "diarize": False, "vertical": True, "prompt": "funny bits"}
AGENT = ("AGENT", {"device": "xpu"})


def recorder_stages(log, fail_at=None, gate=None, pick="stream"):
    """Fake stages that record (name, args) and optionally fail or block."""
    def step(name, value=None):
        def fn(*args):
            if gate is not None and name == "ingest":
                gate.wait(5)
            log.append((name, args))
            if name == fail_at:
                raise RuntimeError(f"{name} exploded")
            return value
        return fn

    return Stages(
        ingest=step("ingest"),
        transcribe=step("transcribe"),
        windows=step("windows"),
        load_agent=step("load_agent", AGENT),
        choose_profile=step("choose_profile", {"profile": pick, "votes": {pick: 3},
                                               "sampled": 3, "fallback": False}),
        score=step("score", {"scored": 4, "failed": 0, "candidates": 2,
                             "uncertain": 1, "device": "xpu"}),
    )


@pytest.fixture
def run(tmp_path):
    r = Run.create(tmp_path, "ep")
    r.path("video.mp4").write_bytes(b"x")
    return r


def _go(runner, run, **settings):
    runner.start(run, run.path("video.mp4"), {**SETTINGS, **settings})
    runner.wait(5)
    return runner.status()


def test_idle_before_any_job():
    assert JobRunner(recorder_stages([])).status() == {"state": "idle"}


def test_stages_run_in_order_and_finish(run):
    log = []
    status = _go(JobRunner(recorder_stages(log)), run)
    assert [name for name, _ in log] == ["ingest", "transcribe", "windows", "score"]
    assert status["state"] == "done"
    assert list(status["stages"]) == list(STAGES)
    assert all(v == "done" for v in status["stages"].values())
    assert status["result"]["candidates"] == 2
    assert status["profile"] == "podcast"
    assert status["run"] == "ep"


def test_an_explicit_profile_leaves_agent_loading_to_the_score_stage(run):
    log = []
    _go(JobRunner(recorder_stages(log)), run)
    _, profile, _, agent = dict(log)["score"]
    assert profile == "podcast"
    assert agent is None


def test_auto_loads_one_agent_and_reuses_it_for_scoring(run):
    log = []
    status = _go(JobRunner(recorder_stages(log)), run, profile="auto")
    assert [name for name, _ in log] == ["ingest", "transcribe", "windows",
                                         "load_agent", "choose_profile", "score"]
    _, profile, _, agent = dict(log)["score"]
    assert profile == "stream"
    assert agent == AGENT
    assert status["profile"] == "stream"
    assert status["votes"] == {"stream": 3}
    assert status["fallback"] is False


def test_a_failure_stops_later_stages_and_records_the_error(run):
    log = []
    status = _go(JobRunner(recorder_stages(log, fail_at="transcribe")), run)
    assert status["state"] == "failed"
    assert status["failed_stage"] == "transcribe"
    assert status["stages"]["transcribe"] == "failed"
    assert status["stages"]["windows"] == "pending"
    assert status["stages"]["score"] == "pending"
    assert "transcribe exploded" in status["error"]
    assert "score" not in [name for name, _ in log]


def test_settings_and_job_state_are_written_to_the_run(run):
    _go(JobRunner(recorder_stages([])), run, profile="auto")
    settings = run.read_json("settings.json")
    assert settings["prompt"] == "funny bits"
    assert settings["profile"] == "auto"
    assert settings["profile_chosen"] == "stream"
    assert settings["profile_votes"] == {"stream": 3}
    assert settings["profile_fallback"] is False
    assert run.read_json("job.json")["state"] == "done"


def test_an_explicit_profile_is_recorded_as_chosen(run):
    _go(JobRunner(recorder_stages([])), run)
    assert run.read_json("settings.json")["profile_chosen"] == "podcast"


def test_upload_is_already_done_and_the_job_is_busy_while_running(run):
    gate = threading.Event()
    runner = JobRunner(recorder_stages([], gate=gate))
    runner.start(run, run.path("video.mp4"), SETTINGS)
    try:
        assert runner.busy() is True
        assert runner.status()["stages"]["upload"] == "done"
        with pytest.raises(JobBusy):
            runner.start(run, run.path("video.mp4"), SETTINGS)
    finally:
        gate.set()
        runner.wait(5)
    assert runner.busy() is False
    assert runner.status()["state"] == "done"


def test_a_new_job_may_start_after_the_last_one_finished(run):
    runner = JobRunner(recorder_stages([]))
    _go(runner, run)
    assert _go(runner, run)["state"] == "done"
