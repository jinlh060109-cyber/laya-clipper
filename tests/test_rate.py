import pytest

from clipper import rate
from clipper.questions import BUILTIN, combined_weights
from clipper.run import Run

AI_Q = {"useful": {"type": "score", "instructions": "How useful?", "criteria": ["no", "some", "very"]},
        "kind": {"type": "choice", "instructions": "Kind?", "criteria": {"tip": "t", "joke": "j"}}}
QUESTIONS = {**BUILTIN, **AI_Q}
WEIGHTS = combined_weights({"useful": 1}, AI_Q)


def _answers(conf=0.5, noul_conf=0.9):
    return {"answers": {
        "clipworthy": {"type": "score", "score": 3.0, "confidence": conf, "probabilities": {}},
        "hook_strength": {"type": "score", "score": 2.0, "confidence": conf, "probabilities": {}},
        "self_contained": {"type": "noul", "noul": 0.8, "confidence": noul_conf, "probabilities": {}},
        "ends_cleanly": {"type": "noul", "noul": 0.6, "confidence": noul_conf, "probabilities": {}},
        "useful": {"type": "score", "score": 2.0, "confidence": conf, "probabilities": {}},
        "kind": {"type": "choice", "choice": "tip", "confidence": 0.9, "probabilities": {}},
    }}


class FakeAgent:
    def __init__(self, response=None, fail_on=None):
        self.response = response or _answers()
        self.fail_on = fail_on
        self.states = []

    def system_one(self, state, questions):
        self.states.append(state)
        if self.fail_on is not None and self.fail_on in state["clip"]["text"]:
            raise RuntimeError("inference blew up")
        return self.response


CLIP = {"id": 0, "start": 10.0, "end": 40.0, "duration": 30.0, "text": "Here is the trick.",
        "category": "tip"}


def test_score_is_the_weighted_mean_of_normalized_answers():
    rated = rate.rate_one(CLIP, "tutorial", QUESTIONS, WEIGHTS, FakeAgent())
    expected = (0.40 * 3 / 4 + 0.25 * 2 / 4 + 0.10 * 0.8 + 0.05 * 0.6 + 0.20 * 2 / 2)
    assert rated["score"] == pytest.approx(expected)
    assert rated["answers"]["kind"]["value"] == "tip"
    assert rated["failed"] is False


def test_confidence_is_the_mean_over_weighted_questions():
    rated = rate.rate_one(CLIP, "tutorial", QUESTIONS, WEIGHTS, FakeAgent())
    assert rated["confidence"] == pytest.approx((0.5 * 3 + 0.9 * 2) / 5)


def test_low_confidence_marks_the_clip_uncertain():
    sure = rate.rate_one(CLIP, "t", QUESTIONS, WEIGHTS, FakeAgent(_answers(conf=0.5)))
    unsure = rate.rate_one(CLIP, "t", QUESTIONS, WEIGHTS, FakeAgent(_answers(conf=0.05)))
    shaky_yes_no = rate.rate_one(CLIP, "t", QUESTIONS, WEIGHTS, FakeAgent(_answers(noul_conf=0.52)))
    assert (sure["uncertain"], unsure["uncertain"], shaky_yes_no["uncertain"]) == (False, True, True)


def _run(tmp_path, candidates):
    run = Run.create(tmp_path, "r")
    run.write_json("transcript.json", {"language": "en", "segments": []})
    run.write_json("segments.json", {"content_type": "tutorial", "questions": QUESTIONS,
                                     "weights": WEIGHTS, "candidates": candidates})
    return run


def test_a_failing_clip_is_marked_and_the_rest_are_scored(tmp_path):
    other = {**CLIP, "id": 1, "text": "boom here"}
    run = _run(tmp_path, [CLIP, other])
    calls = []
    summary = rate.run_rate(run, agent=(FakeAgent(fail_on="boom"), {"device": "xpu"}),
                            progress=lambda d, t: calls.append((d, t)))
    saved = run.read_json("scored.json")
    assert [c["failed"] for c in saved["candidates"]] == [False, True]
    assert "inference blew up" in saved["candidates"][1]["error"]
    assert summary == {"scored": 1, "failed": 1, "device": "xpu"}
    assert calls == [(1, 2), (2, 2)]
    assert saved["laya_model"] == {"device": "xpu"}


def test_no_candidates_means_nothing_to_load(tmp_path):
    run = _run(tmp_path, [])
    summary = rate.run_rate(run, agent=None)
    assert summary == {"scored": 0, "failed": 0, "device": "none"}
    assert run.read_json("scored.json")["candidates"] == []
