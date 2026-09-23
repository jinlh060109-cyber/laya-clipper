import json

import pytest

from clipper.run import Run
from clipper.stages.score import run_score


class FakeAgent:
    def __init__(self):
        self.calls = 0

    def system_one(self, state, questions):
        self.calls += 1
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "score":
                answers[qid] = {"type": "score", "score": 3.0, "confidence": 0.3,
                                "legend": {}, "probabilities": {}}
            elif q["type"] == "choice":
                key = next(iter(q["criteria"]))
                answers[qid] = {"type": "choice", "choice": key, "confidence": 0.3,
                                "probabilities": {}}
            else:
                answers[qid] = {"type": "noul", "noul": 0.9, "confidence": 0.9,
                                "probabilities": {}}
        return {"model": "laya-rl-agent", "answers": answers, "usage": {}}


META = {"package": "0.3.5", "repo": "convaiinnovations/laya", "checkpoint": "root",
        "revision": "abc123", "device": "xpu", "device_requested": "xpu",
        "dtype": "torch.bfloat16", "confidence_calibrated": True,
        "uncalibrated_buckets": []}


@pytest.fixture
def run_with_windows(tmp_path):
    run = Run.create(tmp_path, "ep")
    run.write_json("transcript.json", {"language": "en", "model": "large-v3",
                                       "diarized": True, "segments": []})
    run.write_json("windows.json", {"windows": [
        {"id": i, "start": i * 10.0, "end": i * 10.0 + 30.0,
         "text": f"SPEAKER_01: line {i}", "preceding": "", "position": i / 10,
         "energy_mean": 0.4, "energy_peak": 0.8, "energy_peak_offset": 0.5}
        for i in range(4)]})
    return run


def test_score_writes_both_artifacts(run_with_windows, monkeypatch):
    monkeypatch.setattr("clipper.stages.score.load_agent",
                        lambda *a, **k: (FakeAgent(), META))
    run_score(run_with_windows, "podcast")
    assert run_with_windows.exists("scores.json")
    assert run_with_windows.exists("candidates.json")


def test_provenance_is_written_to_scores_and_carried_to_candidates(run_with_windows,
                                                                   monkeypatch):
    monkeypatch.setattr("clipper.stages.score.load_agent",
                        lambda *a, **k: (FakeAgent(), META))
    run_score(run_with_windows, "podcast")
    scores = run_with_windows.read_json("scores.json")
    candidates = run_with_windows.read_json("candidates.json")
    assert scores["laya_model"]["revision"] == "abc123"
    assert candidates["laya_model"] == scores["laya_model"]


def test_the_agent_is_loaded_once_and_called_once_per_window(run_with_windows,
                                                             monkeypatch):
    agent = FakeAgent()
    loads = []

    def fake_load(*a, **k):
        loads.append(1)
        return agent, META

    monkeypatch.setattr("clipper.stages.score.load_agent", fake_load)
    run_score(run_with_windows, "podcast")
    assert len(loads) == 1
    assert agent.calls == 4


def test_the_transcript_language_selects_the_checkpoint(run_with_windows, monkeypatch):
    seen = {}

    def fake_load(language, profile, device=None, **k):
        seen["language"] = language
        return FakeAgent(), META

    monkeypatch.setattr("clipper.stages.score.load_agent", fake_load)
    run_with_windows.write_json("transcript.json", {"language": "de", "segments": []})
    run_score(run_with_windows, "podcast")
    assert seen["language"] == "de"


def test_raw_answers_are_retained_for_recalibration(run_with_windows, monkeypatch):
    """Spec 13: a weighting change must be a recomputation, not a re-run."""
    monkeypatch.setattr("clipper.stages.score.load_agent",
                        lambda *a, **k: (FakeAgent(), META))
    run_score(run_with_windows, "podcast")
    first = run_with_windows.read_json("scores.json")["windows"][0]
    assert first["answers"]["clipworthy"]["value"] == 3.0
    assert first["answers"]["clipworthy"]["normalized"] == pytest.approx(0.75)
    assert "probabilities" in first["answers"]["clipworthy"]


def test_an_injected_agent_is_used_instead_of_loading(run_with_windows):
    agent = FakeAgent()
    run_score(run_with_windows, "podcast", agent=(agent, META))
    assert agent.calls == 4


def test_missing_windows_artifact_names_the_stage_that_makes_it(tmp_path):
    run = Run.create(tmp_path, "ep")
    run.write_json("transcript.json", {"language": "en", "segments": []})
    with pytest.raises(Exception, match="window"):
        run_score(run, "podcast")


def test_the_returned_summary_reports_counts(run_with_windows, monkeypatch):
    monkeypatch.setattr("clipper.stages.score.load_agent",
                        lambda *a, **k: (FakeAgent(), META))
    summary = run_score(run_with_windows, "podcast")
    assert summary["scored"] == 4
    assert summary["failed"] == 0
    assert "candidates" in summary


def test_zero_windows_skip_laya_and_write_empty_candidates(tmp_path, monkeypatch):
    import clipper.stages.score as stage

    run = Run.create(tmp_path, "silent")
    run.write_json("windows.json", {"windows": []})
    run.write_json("transcript.json", {"language": "en", "segments": []})
    monkeypatch.setattr(stage, "load_agent",
                        lambda *a, **k: pytest.fail("Laya must not load for zero windows"))
    result = run_score(run, "stream")
    assert result == {"scored": 0, "failed": 0, "candidates": 0,
                      "uncertain": 0, "device": "none"}
    candidates = run.read_json("candidates.json")
    assert candidates["laya_model"] is None
    assert candidates["candidates"] == [] and candidates["uncertain"] == []
