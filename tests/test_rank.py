import pytest

from clipper.profiles.loader import Profile, load_profile
from clipper.rank import composite, is_uncertain, passes_gate, rank


def ans(normalized=None, confidence=1.0, qtype="noul", value=None):
    return {"type": qtype, "value": value, "normalized": normalized,
            "confidence": confidence, "probabilities": {}, "legend": None}


def tiny_profile(**kw):
    base = dict(
        name="tiny",
        questions={"a": {"type": "score", "criteria": ["0", "1", "2", "3", "4"],
                         "instructions": "x"},
                   "b": {"type": "noul", "instructions": "x"},
                   "g": {"type": "noul", "instructions": "x"},
                   "c": {"type": "choice", "criteria": {"x": "1", "y": "2"},
                         "instructions": "x"}},
        weights={"a": 0.6, "b": 0.4},
        penalties={},
        gates={},
        thresholds={"candidate": 0.55},
        uncertain_confidence={"score": 0.10, "choice": 0.15, "noul": 0.55},
        merge={}, window={},
    )
    base.update(kw)
    return Profile(**base)


def test_composite_uses_normalized_not_raw_values():
    """A raw 0..4 score must never reach weights calibrated for 0..1."""
    p = tiny_profile()
    answers = {"a": ans(normalized=1.0, value=4.0, qtype="score"),
               "b": ans(normalized=1.0)}
    assert composite(answers, p) == pytest.approx(1.0)


def test_composite_is_a_weighted_sum():
    p = tiny_profile()
    answers = {"a": ans(normalized=0.5, qtype="score"), "b": ans(normalized=0.25)}
    assert composite(answers, p) == pytest.approx(0.6 * 0.5 + 0.4 * 0.25)


def test_penalties_subtract():
    p = tiny_profile(penalties={"g": 0.2})
    answers = {"a": ans(normalized=1.0, qtype="score"), "b": ans(normalized=1.0),
               "g": ans(normalized=1.0)}
    assert composite(answers, p) == pytest.approx(0.8)


def test_composite_is_clamped_to_unit_range():
    p = tiny_profile(penalties={"g": 2.0})
    answers = {"a": ans(normalized=1.0, qtype="score"), "b": ans(normalized=1.0),
               "g": ans(normalized=1.0)}
    assert composite(answers, p) == 0.0


def test_a_missing_answer_contributes_zero_rather_than_raising():
    p = tiny_profile()
    assert composite({"a": ans(normalized=1.0, qtype="score")}, p) == pytest.approx(0.6)


def test_a_choice_answer_never_contributes_to_the_composite():
    p = tiny_profile(weights={"a": 0.6, "b": 0.4, "c": 0.5})
    answers = {"a": ans(normalized=0.0, qtype="score"), "b": ans(normalized=0.0),
               "c": ans(normalized=None, qtype="choice", value="x")}
    assert composite(answers, p) == 0.0


def test_gate_passes_at_or_above_the_floor():
    p = tiny_profile(gates={"g": 0.35})
    assert passes_gate({"g": ans(normalized=0.35)}, p) is True
    assert passes_gate({"g": ans(normalized=0.36)}, p) is True


def test_gate_fails_below_the_floor():
    p = tiny_profile(gates={"g": 0.35})
    assert passes_gate({"g": ans(normalized=0.34)}, p) is False


def test_a_missing_gate_answer_fails_closed():
    p = tiny_profile(gates={"g": 0.35})
    assert passes_gate({}, p) is False


def test_score_confidence_is_judged_against_the_score_threshold():
    """0.12 is confident FOR A SCORE; a single 0.55 threshold would reject it."""
    p = tiny_profile()
    answers = {"a": ans(normalized=0.5, confidence=0.12, qtype="score"),
               "b": ans(normalized=0.5, confidence=0.9)}
    assert is_uncertain(answers, p) is False


def test_score_confidence_below_its_own_threshold_is_uncertain():
    p = tiny_profile()
    answers = {"a": ans(normalized=0.5, confidence=0.05, qtype="score"),
               "b": ans(normalized=0.5, confidence=0.9)}
    assert is_uncertain(answers, p) is True


def test_noul_confidence_is_judged_against_the_noul_threshold():
    """noul confidence is floored at 0.5, so 0.52 is genuinely a coin flip."""
    p = tiny_profile()
    answers = {"a": ans(normalized=0.5, confidence=0.9, qtype="score"),
               "b": ans(normalized=0.5, confidence=0.52)}
    assert is_uncertain(answers, p) is True


def test_only_weighted_questions_drive_uncertainty():
    """An unweighted question's confidence does not route the window."""
    p = tiny_profile()
    answers = {"a": ans(normalized=0.5, confidence=0.9, qtype="score"),
               "b": ans(normalized=0.5, confidence=0.9),
               "g": ans(normalized=0.5, confidence=0.01)}
    assert is_uncertain(answers, p) is False


def test_rank_annotates_every_record():
    p = tiny_profile(gates={"g": 0.35})
    scores = [{"id": 0, "failed": False,
               "answers": {"a": ans(normalized=1.0, qtype="score"),
                           "b": ans(normalized=1.0), "g": ans(normalized=1.0)}}]
    out = rank(scores, p)
    assert out[0]["composite"] == pytest.approx(1.0)
    assert out[0]["gated"] is True
    assert out[0]["uncertain"] is False


def test_rank_passes_failed_records_through_untouched():
    p = tiny_profile()
    out = rank([{"id": 3, "failed": True, "answers": {}, "error": "boom"}], p)
    assert out[0]["failed"] is True
    assert "composite" not in out[0]


def test_real_core_profile_ranks_a_plausible_window():
    p = load_profile("core")
    answers = {k: ans(normalized=0.8, confidence=0.9,
                      qtype=p.questions[k]["type"]) for k in p.questions}
    assert 0.0 <= composite(answers, p) <= 1.0
    assert passes_gate(answers, p) is True
