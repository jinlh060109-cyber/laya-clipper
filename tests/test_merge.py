import pytest
from clipper.merge import merge_candidates
from clipper.profiles.loader import load_profile

PROFILE = load_profile("podcast")

def _window(i: int, start: float, offset: float = 0.2, peak: float = 0.3) -> dict:
    return {"id": i, "start": start, "end": start + 30.0, "text": "t", "preceding": "",
            "position": 0.1, "energy_mean": 0.3, "energy_peak": peak,
            "energy_peak_offset": offset}

def _ranked(i: int, composite: float, gated: bool = True,
            uncertain: bool = False, fmt: str = "hot_take") -> dict:
    return {"id": i, "failed": False, "composite": composite, "gated": gated,
            "uncertain": uncertain,
            "answers": {"clip_format": {"type": "choice", "value": fmt, "normalized": None,
                                        "confidence": 0.8},
                        "buried_lede": {"type": "noul", "value": 0.1, "normalized": 0.1,
                                        "confidence": 0.9},
                        "opens_with_windup": {"type": "noul", "value": 0.1, "normalized": 0.1,
                                              "confidence": 0.9}}}

def test_adjacent_hot_windows_merge_into_one_candidate():
    windows = [_window(0, 0.0), _window(1, 10.0), _window(2, 20.0)]
    ranked = [_ranked(0, 0.8), _ranked(1, 0.8), _ranked(2, 0.8)]
    out = merge_candidates(windows, ranked, PROFILE)
    assert len(out["candidates"]) == 1
    assert out["candidates"][0]["window_ids"] == [0, 1, 2]

def test_cold_windows_split_candidates():
    windows = [_window(i, i * 10.0) for i in range(5)]
    ranked = [_ranked(0, 0.8), _ranked(1, 0.1), _ranked(2, 0.1),
              _ranked(3, 0.1), _ranked(4, 0.8)]
    assert len(merge_candidates(windows, ranked, PROFILE)["candidates"]) == 2

def test_below_threshold_windows_are_excluded():
    windows = [_window(0, 0.0)]
    assert merge_candidates(windows, [_ranked(0, 0.2)], PROFILE)["candidates"] == []

def test_gated_windows_are_excluded_regardless_of_score():
    windows = [_window(0, 0.0)]
    ranked = [_ranked(0, 0.95, gated=False)]
    assert merge_candidates(windows, ranked, PROFILE)["candidates"] == []

def test_candidate_is_capped_at_max_seconds():
    windows = [_window(i, i * 10.0) for i in range(12)]
    ranked = [_ranked(i, 0.8) for i in range(12)]
    out = merge_candidates(windows, ranked, PROFILE)
    assert all(c["end"] - c["start"] <= 60.0 + 1e-6 for c in out["candidates"])

def test_late_energy_peak_extends_candidate_backwards():
    """The laugh is not the clip; the line that caused it is. Extend backwards."""
    windows = [_window(0, 100.0, offset=0.95, peak=0.9)]
    out = merge_candidates(windows, [_ranked(0, 0.8)], PROFILE)
    candidate = out["candidates"][0]
    assert candidate["start"] < 100.0
    assert candidate["backward_extended"] is True

def test_early_energy_peak_does_not_extend():
    windows = [_window(0, 100.0, offset=0.1, peak=0.9)]
    out = merge_candidates(windows, [_ranked(0, 0.8)], PROFILE)
    assert out["candidates"][0]["backward_extended"] is False

def test_backward_extension_never_goes_below_zero():
    windows = [_window(0, 2.0, offset=0.95, peak=0.9)]
    out = merge_candidates(windows, [_ranked(0, 0.8)], PROFILE)
    assert out["candidates"][0]["start"] >= 0.0

def test_energy_alone_never_promotes_a_candidate():
    """Loud but empty. Without a transcript signal there is no clip."""
    windows = [_window(0, 0.0, offset=0.9, peak=1.0)]
    out = merge_candidates(windows, [_ranked(0, 0.15)], PROFILE)
    assert out["candidates"] == []

def test_uncertain_windows_go_to_their_own_bucket():
    windows = [_window(0, 0.0)]
    ranked = [_ranked(0, 0.8, uncertain=True)]
    out = merge_candidates(windows, ranked, PROFILE)
    assert out["candidates"] == []
    assert out["uncertain"][0]["window_ids"] == [0]

def test_candidate_reports_peak_and_curve():
    windows = [_window(0, 0.0), _window(1, 10.0)]
    out = merge_candidates(windows, [_ranked(0, 0.6), _ranked(1, 0.9)], PROFILE)
    candidate = out["candidates"][0]
    assert candidate["peak_composite"] == pytest.approx(0.9)
    assert candidate["curve"] == [0.6, 0.9]

def test_candidates_are_sorted_by_peak_descending():
    windows = [_window(i, i * 40.0) for i in range(3)]
    ranked = [_ranked(0, 0.6), _ranked(1, 0.9), _ranked(2, 0.75)]
    out = merge_candidates(windows, ranked, PROFILE)
    assert [c["peak_composite"] for c in out["candidates"]] == [0.9, 0.75, 0.6]

def test_failed_windows_are_ignored():
    windows = [_window(0, 0.0), _window(1, 10.0)]
    ranked = [{"id": 0, "failed": True, "error": "429"}, _ranked(1, 0.8)]
    out = merge_candidates(windows, ranked, PROFILE)
    assert out["candidates"][0]["window_ids"] == [1]

def test_signals_read_normalized_values_not_raw():
    """Raw Laya values are 0..k-1 for scores; signals must be on the 0..1 scale."""
    windows = [_window(0, 0.0)]
    record = _ranked(0, 0.8)
    record["answers"]["ends_cleanly"] = {"type": "score", "value": 3.0, "normalized": 0.75,
                                         "confidence": 0.2}
    signals = merge_candidates(windows, [record], PROFILE)["candidates"][0]["signals"]
    assert signals["ends_cleanly"] == pytest.approx(0.75)
    assert signals["buried_lede"] == pytest.approx(0.1)

def test_clip_format_keeps_the_raw_choice_value():
    windows = [_window(0, 0.0)]
    out = merge_candidates(windows, [_ranked(0, 0.8, fmt="story")], PROFILE)
    assert out["candidates"][0]["clip_format"] == "story"
