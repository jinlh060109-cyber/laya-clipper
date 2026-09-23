import pytest

from clipper import selection as sel
from clipper.run import Run


def _cand(i, score, conf=0.4, failed=False, uncertain=False, **extra):
    return {"id": i, "start": 10.0 * i, "end": 10.0 * i + 20, "duration": 20.0,
            "category": "tip", "reason": "because", "hook_line": f"Line {i}.",
            "text": f"Line {i}. More words.", "score": score, "confidence": conf,
            "uncertain": uncertain, "failed": failed, **extra}


ACTION = [{"id": 0, "start": 100.0, "end": 115.0, "peak_score": 0.7},
          {"id": 1, "start": 200.0, "end": 205.0, "peak_score": 0.9},   # under 10 s
          {"id": 2, "start": 300.0, "end": 320.0, "peak_score": 0.6},
          {"id": 3, "start": 400.0, "end": 420.0, "peak_score": 0.5}]


def test_the_top_scores_are_kept_best_first():
    rows = sel.select([_cand(0, 0.5), _cand(1, 0.9), _cand(2, 0.7)], [], top_n=2)
    kept = [r for r in rows if r["include"]]
    assert [r["id"] for r in kept] == ["c1", "c2"]
    assert rows[-1]["id"] == "c0" and rows[-1]["include"] is False


def test_ties_are_broken_by_confidence():
    rows = sel.select([_cand(0, 0.5, conf=0.2), _cand(1, 0.5, conf=0.6)], [], top_n=1)
    assert rows[0]["id"] == "c1"


def test_clips_under_the_minimum_score_are_listed_but_not_kept():
    rows = sel.select([_cand(0, 0.2)], [], top_n=5, min_score=0.3)
    assert rows[0]["include"] is False


def test_failed_clips_are_never_offered():
    rows = sel.select([_cand(0, 0.9, failed=True), _cand(1, 0.4)], [])
    assert [r["id"] for r in rows] == ["c1"]


def test_the_best_action_moments_are_appended():
    rows = sel.select([_cand(0, 0.5)], ACTION, action_n=2)
    action = [r for r in rows if r["source"] == "action"]
    assert [r["id"] for r in action] == ["a0", "a2", "a3"]
    assert [r["include"] for r in action] == [True, True, False]
    assert action[0]["category"] == "action" and action[0]["title_hint"].startswith("Action at 1:40")


def test_row_fields_for_the_preview():
    row = sel.select([_cand(3, 0.8, uncertain=True)], [])[0]
    assert row == {"id": "c3", "start": 30.0, "end": 50.0, "duration": 20.0,
                   "category": "tip", "score": 0.8, "confidence": 0.4, "uncertain": True,
                   "title_hint": "Line 3.", "reason": "because", "source": "ai",
                   "include": True, "fill_in": False}


def test_chunk_titles_come_from_their_first_sentence():
    row = sel.select([_cand(0, 0.8, category="chunk", hook_line="")], [])[0]
    assert row["source"] == "chunks" and row["title_hint"] == "Line 0."


def _run(tmp_path):
    run = Run.create(tmp_path, "r")
    run.write_json("scored.json", {"laya_model": None,
                                   "candidates": [_cand(0, 0.6), _cand(1, 0.4)]})
    run.write_json("action.json", {"candidates": ACTION[:1]})
    return run


def test_run_select_writes_the_selection(tmp_path):
    run = _run(tmp_path)
    out = sel.run_select(run, top_n=1)
    assert run.read_json("selection.json") == out
    assert out["rule"] == {"top_n": 1, "min_score": 0.30, "action_n": 2}
    assert [c["id"] for c in out["clips"] if c["include"]] == ["c0", "a0"]


def test_choices_from_the_preview_are_applied(tmp_path):
    run = _run(tmp_path)
    sel.run_select(run)
    out = sel.apply_choices(run, [{"id": "c1", "include": True, "fill_in": True},
                                  {"id": "a0", "include": False, "fill_in": False}])
    by_id = {c["id"]: c for c in out["clips"]}
    assert by_id["c1"]["include"] and by_id["c1"]["fill_in"]
    assert not by_id["a0"]["include"]


def test_an_unknown_clip_id_is_refused(tmp_path):
    run = _run(tmp_path)
    sel.run_select(run)
    with pytest.raises(ValueError, match="c9"):
        sel.apply_choices(run, [{"id": "c9", "include": True}])


def test_a_run_without_action_moments_still_selects(tmp_path):
    run = Run.create(tmp_path, "r")
    run.write_json("scored.json", {"laya_model": None, "candidates": [_cand(0, 0.6)]})
    assert [c["id"] for c in sel.run_select(run)["clips"]] == ["c0"]
