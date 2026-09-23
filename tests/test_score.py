from clipper.profiles.loader import load_profile
from clipper.score import (answers_to_dict, build_questions, normalize_answer,
                           score_windows, window_state)


class FakeAgent:
    """Stands in for laya.Agent. Records calls; returns canned answers."""

    def __init__(self, answers=None, fail_on=None):
        self.calls = []
        self._answers = answers or {}
        self._fail_on = fail_on or set()

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        if len(self.calls) - 1 in self._fail_on:
            raise RuntimeError("simulated inference failure")
        answers = {}
        for qid, q in questions.items():
            if qid in self._answers:
                answers[qid] = self._answers[qid]
            elif q["type"] == "score":
                answers[qid] = {"type": "score", "score": 2.0, "confidence": 0.15,
                                "legend": {"0": "a"}, "probabilities": {"0": 0.2}}
            elif q["type"] == "choice":
                key = next(iter(q["criteria"]))
                answers[qid] = {"type": "choice", "choice": key, "confidence": 0.2,
                                "probabilities": {key: 0.5}}
            else:
                answers[qid] = {"type": "noul", "noul": 0.7, "confidence": 0.7,
                                "probabilities": {}}
        return {"model": "laya-rl-agent", "answers": answers,
                "usage": {"input_tokens": 226, "output_tokens": 0}}


def win(i=0, start=0.0, end=30.0, text="hello world"):
    return {"id": i, "start": start, "end": end, "text": text, "preceding": "before",
            "position": 0.4, "energy_mean": 0.44, "energy_peak": 0.86,
            "energy_peak_offset": 0.71}


def test_build_questions_passes_profile_yaml_through_unchanged():
    p = load_profile("core")
    qs = build_questions(p)
    assert qs["clipworthy"]["type"] == "score"
    assert qs["clipworthy"]["instructions"] == p.questions["clipworthy"]["instructions"]
    assert qs["clipworthy"]["criteria"] == p.questions["clipworthy"]["criteria"]
    assert set(qs) == set(p.questions)


def test_window_state_puts_window_text_first():
    """Laya truncates state from the right, so the judged text must lead."""
    keys = list(window_state(win(), "podcast").keys())
    assert keys.index("window") < keys.index("preceding")
    assert keys[-1] == "preceding"


def test_window_state_carries_energy_and_position():
    s = window_state(win(), "podcast")
    assert s["profile"] == "podcast"
    assert s["window"]["start"] == 0.0
    assert s["energy_peak_offset"] == 0.71


def test_normalize_score_divides_by_levels_minus_one():
    qdef = {"type": "score", "criteria": ["a", "b", "c", "d", "e"]}
    assert normalize_answer({"type": "score", "score": 4.0}, qdef) == 1.0
    assert normalize_answer({"type": "score", "score": 0.0}, qdef) == 0.0
    assert normalize_answer({"type": "score", "score": 2.0}, qdef) == 0.5


def test_normalize_score_with_three_levels():
    qdef = {"type": "score", "criteria": ["a", "b", "c"]}
    assert normalize_answer({"type": "score", "score": 1.0}, qdef) == 0.5


def test_normalize_noul_passes_probability_through():
    assert normalize_answer({"type": "noul", "noul": 0.73}, {"type": "noul"}) == 0.73


def test_normalize_choice_is_none_because_choice_is_never_weighted():
    qdef = {"type": "choice", "criteria": {"a": "x", "b": "y"}}
    assert normalize_answer({"type": "choice", "choice": "a"}, qdef) is None


def test_normalize_score_is_clamped_to_unit_range():
    qdef = {"type": "score", "criteria": ["a", "b"]}
    assert normalize_answer({"type": "score", "score": 1.4}, qdef) == 1.0
    assert normalize_answer({"type": "score", "score": -0.2}, qdef) == 0.0


def test_answers_to_dict_keeps_raw_value_and_normalized():
    p = load_profile("core")
    resp = FakeAgent().system_one({}, build_questions(p))
    out = answers_to_dict(resp, p)
    assert out["clipworthy"]["value"] == 2.0          # raw, as Laya returned it
    assert out["clipworthy"]["normalized"] == 0.5     # rebased for the composite
    assert out["open_loop"]["value"] == 0.7
    assert out["open_loop"]["normalized"] == 0.7
    assert out["hook_type"]["normalized"] is None


def test_answers_to_dict_carries_confidence_per_answer():
    p = load_profile("core")
    out = answers_to_dict(FakeAgent().system_one({}, build_questions(p)), p)
    assert out["clipworthy"]["confidence"] == 0.15
    assert out["open_loop"]["confidence"] == 0.7


def test_score_windows_calls_the_agent_once_per_window():
    p = load_profile("core")
    agent = FakeAgent()
    records = score_windows([win(0), win(1), win(2)], p, agent)
    assert len(agent.calls) == 3
    assert [r["id"] for r in records] == [0, 1, 2]
    assert all(r["failed"] is False for r in records)


def test_score_windows_sends_the_whole_bundle_in_one_call():
    """One call per window, not one per question."""
    p = load_profile("core")
    agent = FakeAgent()
    score_windows([win(0)], p, agent)
    _, questions = agent.calls[0]
    assert len(questions) == len(p.questions)


def test_a_failing_window_is_marked_and_the_run_continues():
    p = load_profile("core")
    agent = FakeAgent(fail_on={1})
    records = score_windows([win(0), win(1), win(2)], p, agent)
    assert [r["failed"] for r in records] == [False, True, False]
    assert records[1]["answers"] == {}
    assert "simulated inference failure" in records[1]["error"]


def test_build_questions_deep_copies_nested_structures():
    """Mutating the returned dict must not affect the original Profile."""
    p = load_profile("core")
    original_criteria = p.questions["clipworthy"]["criteria"]
    original_length = len(original_criteria)
    qs = build_questions(p)
    # Mutate the returned criteria list
    qs["clipworthy"]["criteria"].append("mutated")
    # Original Profile must be unchanged
    assert len(p.questions["clipworthy"]["criteria"]) == original_length
    assert p.questions["clipworthy"]["criteria"] is original_criteria


class MalformedAgent:
    """Returns structurally incorrect responses without raising."""

    def __init__(self, response):
        self.response = response

    def system_one(self, state, questions):
        return self.response


def test_malformed_response_from_agent_is_caught_and_window_marked_failed():
    """A response that is malformed but doesn't raise should mark the window failed."""
    p = load_profile("core")
    # Agent returns answers as a string instead of a dict
    malformed_agent = MalformedAgent({"answers": "not-a-mapping"})
    records = score_windows([win(0), win(1), win(2)], p, malformed_agent)
    # All windows should have records
    assert len(records) == 3
    assert [r["id"] for r in records] == [0, 1, 2]
    # First window should be marked failed
    assert records[0]["failed"] is True
    assert "error" in records[0]
    assert records[0]["answers"] == {}
    # Remaining windows should also be attempted (agent called 3 times)
    assert records[1]["failed"] is True
    assert records[2]["failed"] is True


def test_score_windows_reports_progress_after_every_window_including_failures():
    from clipper.score import score_windows
    seen = []
    score_windows([win(0), win(1, 10.0, 40.0), win(2, 20.0, 50.0)], load_profile("podcast"),
                  FakeAgent(fail_on={1}), progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 3), (2, 3), (3, 3)]
