from clipper.silence import silent_spans


def words(*spans):
    return {"segments": [{"words": [{"word": "w", "start": s, "end": e} for s, e in spans]}]}


def talk(start, end, step=0.5):
    """Continuous speech: a word every `step` seconds."""
    out, t = [], start
    while t < end:
        out.append((t, min(end, t + step)))
        t += step
    return out


def test_no_words_means_the_whole_video_is_silent():
    assert silent_spans({"segments": []}, 120.0) == [[0.0, 120.0]]


def test_gaps_of_at_least_min_gap_are_silent_and_shorter_ones_are_not():
    t = words(*talk(0, 10), *talk(15, 30), *talk(45, 60))
    assert silent_spans(t, 60.0) == [[30.0, 45.0]]


def test_lead_in_and_tail_count():
    t = words(*talk(20, 40))
    assert silent_spans(t, 100.0) == [[0.0, 20.0], [40.0, 100.0]]


def test_a_short_isolated_utterance_does_not_split_the_silence():
    t = words(*talk(0, 10), (50.0, 50.4), (50.5, 51.0), *talk(90, 100))
    assert silent_spans(t, 100.0) == [[10.0, 90.0]]


def test_a_long_isolated_utterance_does_split_it():
    t = words(*talk(0, 10), *talk(50, 55), *talk(90, 100))
    assert silent_spans(t, 100.0) == [[10.0, 50.0], [55.0, 90.0]]


def test_a_video_shorter_than_min_gap_has_no_spans():
    assert silent_spans({"segments": []}, 5.0) == []


def test_words_past_the_end_are_clamped():
    t = words(*talk(0, 10), *talk(98, 104))
    assert silent_spans(t, 100.0) == [[10.0, 98.0]]
