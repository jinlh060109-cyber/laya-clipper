import json
import pytest

from clipper import rate
from clipper.questions import FIXED as BUILTIN, combined_weights
from clipper.run import Run

AI_Q = {"useful": {"type": "score", "instructions": "How useful?", "criteria": ["no", "some", "very"]},
        "kind": {"type": "choice", "instructions": "Kind?", "criteria": {"tip": "t", "joke": "j"}}}
QUESTIONS = {**BUILTIN, **AI_Q}
WEIGHTS = combined_weights({"useful": 1}, AI_Q)


def _yes_no(p_yes):
    """How Laya answers a yes/no now that it is asked as an A/B choice."""
    return {"type": "choice", "choice": "A" if p_yes >= 0.5 else "B",
            "confidence": 0.3, "probabilities": {"A": p_yes, "B": round(1 - p_yes, 6)}}


def _answers(conf=0.5, noul_conf=0.9):
    return {"answers": {
        "clipworthy": {"type": "score", "score": 3.0, "confidence": conf, "probabilities": {}},
        "hook_strength": {"type": "score", "score": 2.0, "confidence": conf, "probabilities": {}},
        "has_start": _yes_no(0.8 if noul_conf > 0.6 else noul_conf),
        "has_end": _yes_no(0.6 if noul_conf > 0.6 else noul_conf),
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
        if self.fail_on is not None and self.fail_on in json.dumps(state):
            raise RuntimeError("inference blew up")
        return self.response


CLIP = {"id": 0, "start": 10.0, "end": 40.0, "duration": 30.0, "text": "Here is the trick.",
        "category": "tip"}


def test_score_is_the_weighted_mean_of_normalized_answers():
    rated = rate.rate_one(CLIP, "tutorial", QUESTIONS, WEIGHTS, FakeAgent())
    expected = (0.30 * 3 / 4 + 0.20 * 2 / 4 + 0.15 * 0.8 + 0.15 * 0.6 + 0.20 * 2 / 2)
    assert rated["score"] == pytest.approx(expected)
    assert rated["answers"]["kind"]["value"] == "tip"
    assert rated["failed"] is False


def test_confidence_is_the_mean_over_weighted_questions():
    rated = rate.rate_one(CLIP, "tutorial", QUESTIONS, WEIGHTS, FakeAgent())
    # a yes/no's confidence is max(p, 1-p): 0.8 and 0.6 here
    assert rated["confidence"] == pytest.approx((0.5 * 3 + 0.8 + 0.6) / 5)


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


def test_laya_is_asked_yes_no_questions_as_a_neutral_choice():
    seen = []

    class Spy(FakeAgent):
        def system_one(self, state, questions):
            seen.append(questions)
            return super().system_one(state, questions)

    rated = rate.rate_one(CLIP, "t", QUESTIONS, WEIGHTS, Spy())
    asked = seen[-1]["has_end"]
    assert asked["type"] == "choice" and set(asked["criteria"]) == {"A", "B"}
    assert asked["criteria"]["A"].startswith("Yes.")
    answer = rated["answers"]["has_end"]
    assert answer["type"] == "noul" and answer["value"] == pytest.approx(0.6)
    assert answer["probabilities"] == {"false": pytest.approx(0.4), "true": pytest.approx(0.6)}


def test_edge_questions_read_only_the_opening_and_closing_sentence():
    from clipper.questions import FIXED
    agent = FakeAgent()
    candidate = {"id": "c1", "start": 0.0, "end": 30.0, "duration": 30.0,
                 "text": "So how much XP? The answer is a lot. That is the next factor."}
    rate.rate_one(candidate, "gaming", FIXED, {q: 0.25 for q in FIXED}, agent)
    clip_state, edges_state = agent.states
    assert "clip" in clip_state and "opening_sentence" not in clip_state
    assert edges_state == {"content_type": "gaming", "opening_sentence": "So how much XP?",
                           "closing_sentence": "That is the next factor."}


def test_a_low_start_or_end_answer_flags_the_clip():
    from clipper.questions import FIXED
    response = {"answers": {
        "clipworthy": {"score": 3, "probabilities": {"0": 0, "1": 0, "2": 0, "3": 1, "4": 0}, "confidence": 0.9},
        "hook_strength": {"score": 2, "probabilities": {"0": 0, "1": 0, "2": 1, "3": 0, "4": 0}, "confidence": 0.9},
        "has_start": _yes_no(0.2), "has_end": _yes_no(0.9)}}
    rated = rate.rate_one({"id": "c1", "duration": 30.0, "text": "A. B."}, "gaming", FIXED,
                          {q: 0.25 for q in FIXED}, FakeAgent(response))
    assert rated["flags"] == ["no start"]
