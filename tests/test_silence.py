from clipper.silence import silent_spans, speech_only


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


def scored(*spans, score=0.9):
    return [{"word": "w", "start": s, "end": e, "score": score} for s, e in spans]


def test_sparse_utterances_do_not_count_as_speech():
    """Whisper over game audio: one 'word' stretched over 29 s (the real run)."""
    t = {"segments": [{"words": scored(*talk(0, 10))},
                      {"words": scored((40.0, 69.0))},
                      {"words": scored(*talk(120, 130))}]}
    assert silent_spans(t, 130.0) == [[10.0, 120.0]]


def test_the_real_runs_invented_sentence_does_not_count_as_speech():
    """6 invented words over 11 s, timings only interpolated (score 0.0)."""
    invented = [(69.0 + k * 11 / 6, 69.0 + (k + 1) * 11 / 6) for k in range(6)]
    t = {"segments": [{"words": scored(*talk(0, 10))},
                      {"words": scored(*invented, score=0.0)},
                      {"words": scored(*talk(120, 130))}]}
    assert silent_spans(t, 130.0) == [[10.0, 120.0]]


def test_dense_speech_in_a_language_without_an_aligner_still_counts():
    """No aligner means every word scores 0.0; it is still someone talking."""
    t = {"segments": [{"words": scored(*talk(0, 10))},
                      {"words": scored(*talk(60, 70), score=0.0)},
                      {"words": scored(*talk(120, 130))}]}
    assert silent_spans(t, 130.0) == [[10.0, 60.0], [70.0, 120.0]]


def test_speech_only_keeps_real_talk_and_drops_the_rest():
    t = {"segments": [{"words": scored(*talk(0, 10))},
                      {"words": scored((40.0, 69.0))}]}
    kept = speech_only(t)
    assert [w["start"] for s in kept["segments"] for w in s["words"]] == [
        w["start"] for w in scored(*talk(0, 10))]
    assert t["segments"][1]["words"]  # the input is not modified
