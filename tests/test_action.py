import pytest

from clipper.action import THRESHOLD, action_windows, choose_action


def arrays(n, energy=0.5, motion=0.5, raw=3.0, cuts=0):
    return [energy] * n, [motion] * n, [raw] * n, [cuts] * n


def test_a_long_span_is_cut_into_20s_windows_every_10s_ending_flush():
    wins = action_windows([[0.0, 45.0]], *arrays(60))
    assert [(w["start"], w["end"]) for w in wins] == [(0, 20), (10, 30), (20, 40), (25, 45)]


def test_a_short_span_is_one_window_and_under_8s_none():
    assert [(w["start"], w["end"]) for w in action_windows([[5.0, 17.0]], *arrays(30))] == [(5, 17)]
    assert action_windows([[0.0, 7.0]], *arrays(30)) == []


def test_the_score_weighs_peak_loudness_motion_and_cuts():
    energy, motion, raw, cuts = arrays(20, energy=0.0, motion=1.0)
    energy[4] = energy[9] = energy[14] = 0.9
    cuts[3] = 8
    (w,) = action_windows([[0.0, 20.0]], energy, motion, raw, cuts)
    assert w["signals"] == {"loudness": 0.9, "motion": 1.0, "cut_rate": 1.0}
    assert w["score"] == round(0.45 * 0.9 + 0.35 + 0.20, 4)
    assert w["peak_time"] == 4.5 and w["static"] is False


def test_a_frozen_picture_is_static():
    (w,) = action_windows([[0.0, 20.0]], *arrays(20, raw=0.2))
    assert w["static"] is True


def win(start, end, score, static=False):
    return {"start": start, "end": end, "score": score, "peak_time": start + 1,
            "static": static, "signals": {}}


def test_hot_windows_that_touch_merge_and_split_at_60s():
    wins = [win(t, t + 20, 0.8) for t in range(0, 100, 10)]
    chosen = sorted(choose_action(wins), key=lambda c: c["start"])
    assert [(c["start"], c["end"]) for c in chosen] == [(0, 60), (50, 110)]


def test_static_windows_are_never_chosen_even_to_fill():
    assert choose_action([win(0, 20, 0.95, static=True)]) == []


def test_top_15_by_peak_score():
    wins = [win(t * 100, t * 100 + 20, 0.6 + t / 100) for t in range(20)]
    chosen = choose_action(wins)
    assert len(chosen) == 15
    assert chosen[0]["peak_score"] == pytest.approx(0.79)
    assert [c["id"] for c in chosen] == list(range(15))


def test_fewer_than_5_hot_are_filled_with_the_best_non_overlapping_rest():
    wins = [win(0, 20, 0.9), win(10, 30, 0.5), win(100, 120, 0.4), win(200, 220, 0.3),
            win(300, 320, 0.2), win(400, 420, 0.1)]
    chosen = choose_action(wins)
    assert [(c["start"], c["peak_score"]) for c in chosen] == [
        (0, 0.9), (100, 0.4), (200, 0.3), (300, 0.2), (400, 0.1)]


def test_threshold_is_the_spec_value():
    assert THRESHOLD == 0.6
