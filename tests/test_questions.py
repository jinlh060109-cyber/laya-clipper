import pytest

from clipper import questions as q


def test_builtin_questions_are_valid_laya_questions():
    assert set(q.BUILTIN) == {"clipworthy", "hook_strength", "self_contained", "ends_cleanly"}
    assert q.problems(q.BUILTIN) == []


def test_ai_questions_of_each_type_become_laya_questions():
    got, problems = q.from_ai([
        {"id": "Useful Tip", "type": "score", "instructions": "How useful is the tip?",
         "levels": ["None", "Some", "Clear", "Great"], "choices": [], "true_means": "",
         "false_means": ""},
        {"id": "kind", "type": "choice", "instructions": "What kind of moment?",
         "levels": [], "choices": [{"key": "tip", "description": "A tip"},
                                   {"key": "joke", "description": "A joke"},
                                   {"key": "story", "description": "A story"}],
         "true_means": "", "false_means": ""},
        {"id": "numbers", "type": "noul", "instructions": "Gives concrete numbers.",
         "levels": [], "choices": [], "true_means": "Yes, numbers.", "false_means": "No numbers."},
    ])
    assert problems == []
    assert got["useful_tip"] == {"type": "score", "instructions": "How useful is the tip?",
                                 "criteria": ["None", "Some", "Clear", "Great"]}
    assert got["kind"]["criteria"] == {"tip": "A tip", "joke": "A joke", "story": "A story"}
    assert got["numbers"]["criteria"] == {"true": "Yes, numbers.", "false": "No numbers."}


def test_broken_ai_questions_are_dropped_with_a_reason():
    got, problems = q.from_ai([
        {"id": "a", "type": "score", "instructions": "x", "levels": ["only one"]},
        {"id": "b", "type": "vibe", "instructions": "x"},
        {"id": "c", "type": "noul", "instructions": ""},
        {"id": "d", "type": "choice", "instructions": "x", "choices": [{"key": "one", "description": "1"}]},
    ])
    assert got == {}
    assert len(problems) == 4


def test_ai_question_ids_never_replace_builtins_or_each_other():
    got, _ = q.from_ai([
        {"id": "clipworthy", "type": "noul", "instructions": "a"},
        {"id": "x", "type": "noul", "instructions": "b"},
        {"id": "X", "type": "noul", "instructions": "c"},
    ])
    assert set(got) == {"ai_clipworthy", "x", "x_2"}


def test_at_most_six_ai_questions_are_kept():
    items = [{"id": f"q{i}", "type": "noul", "instructions": "i"} for i in range(9)]
    got, problems = q.from_ai(items)
    assert len(got) == 6 and any("six" in p for p in problems)


def test_weights_sum_to_one_and_split_builtins_and_ai():
    ai = {"tip": {"type": "score", "instructions": "i", "criteria": ["a", "b", "c"]},
          "numbers": {"type": "noul", "instructions": "i", "criteria": {"true": "y", "false": "n"}},
          "kind": {"type": "choice", "instructions": "i", "criteria": {"a": "a", "b": "b"}}}
    weights = q.combined_weights({"tip": 3, "numbers": 1, "kind": 5, "ghost": 2}, ai)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert "kind" not in weights and "ghost" not in weights
    assert weights["tip"] == pytest.approx(0.15) and weights["numbers"] == pytest.approx(0.05)
    assert weights["clipworthy"] == pytest.approx(0.40)


def test_without_ai_questions_the_builtins_carry_all_the_weight():
    weights = q.combined_weights({}, {})
    assert set(weights) == set(q.BUILTIN_WEIGHTS)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["clipworthy"] == pytest.approx(0.40 / 0.80)


def test_ai_questions_without_weights_share_equally():
    ai = {"a": {"type": "noul", "instructions": "i", "criteria": {"true": "y", "false": "n"}},
          "b": {"type": "noul", "instructions": "i", "criteria": {"true": "y", "false": "n"}}}
    weights = q.combined_weights({}, ai)
    assert weights["a"] == pytest.approx(0.10) and weights["b"] == pytest.approx(0.10)
