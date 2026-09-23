import pytest
from clipper.window import build_windows

def _transcript(n_words: int = 300, start: float = 0.0, wps: float = 3.0) -> dict:
    words = [{"word": f"w{i}", "start": start + i / wps, "end": start + (i + 0.9) / wps,
              "score": 0.9, "speaker": "SPEAKER_00" if i % 60 < 30 else "SPEAKER_01"}
             for i in range(n_words)]
    return {"language": "en", "model": "large-v3", "diarized": True,
            "segments": [{"start": words[0]["start"], "end": words[-1]["end"],
                          "speaker": "SPEAKER_00", "text": " ".join(w["word"] for w in words),
                          "words": words}]}

def test_windows_step_by_step_seconds():
    w = build_windows(_transcript(), [0.5] * 100, duration=100.0)
    assert w[1]["start"] - w[0]["start"] == pytest.approx(10.0, abs=1.5)

def test_windows_are_about_window_seconds_long():
    w = build_windows(_transcript(), [0.5] * 100, duration=100.0)
    assert all(20.0 <= x["end"] - x["start"] <= 34.0 for x in w)

def test_window_never_starts_mid_word():
    t = _transcript()
    starts = {round(word["start"], 3)
              for word in t["segments"][0]["words"]}
    w = build_windows(t, [0.5] * 100, duration=100.0)
    assert all(round(x["start"], 3) in starts for x in w)

def test_preceding_context_is_populated_after_the_first_window():
    w = build_windows(_transcript(), [0.5] * 100, duration=100.0)
    assert w[0]["preceding"] == ""
    assert len(w[3]["preceding"]) > 0

def test_text_carries_speaker_labels():
    w = build_windows(_transcript(), [0.5] * 100, duration=100.0)
    assert "SPEAKER_00:" in w[0]["text"]

def test_energy_peak_offset_locates_the_spike():
    """A spike late in the window must report an offset near 1.0, so merging
    can extend the candidate backwards to the line that caused the reaction."""
    energy = [0.1] * 100
    energy[28] = 0.99
    w = build_windows(_transcript(), energy, duration=100.0)
    first = w[0]
    assert first["energy_peak"] == pytest.approx(0.99)
    assert first["energy_peak_offset"] > 0.85

def test_position_is_fractional_through_the_source():
    w = build_windows(_transcript(n_words=900), [0.5] * 300, duration=300.0)
    assert w[0]["position"] < 0.1
    assert w[-1]["position"] > 0.7

def test_ids_are_stable_and_sequential():
    w = build_windows(_transcript(), [0.5] * 100, duration=100.0)
    assert [x["id"] for x in w] == list(range(len(w)))

def test_short_source_yields_one_window():
    t = _transcript(n_words=20)
    assert len(build_windows(t, [0.5] * 10, duration=7.0)) == 1


def _fractional_window_transcript() -> dict:
    """A single window with start=12.4, end=29.9 -- boundaries that fall mid-second,
    which is the normal case for every real window after the first (word starts
    essentially never land on an exact integer second)."""
    words = [
        {"word": "a", "start": 12.4, "end": 12.9, "score": 0.9, "speaker": "SPEAKER_00"},
        {"word": "b", "start": 20.0, "end": 20.5, "score": 0.9, "speaker": "SPEAKER_00"},
        {"word": "c", "start": 29.0, "end": 29.9, "score": 0.9, "speaker": "SPEAKER_00"},
    ]
    return {"language": "en", "model": "large-v3", "diarized": True,
            "segments": [{"start": words[0]["start"], "end": words[-1]["end"],
                          "speaker": "SPEAKER_00", "text": " ".join(w["word"] for w in words),
                          "words": words}]}


def test_energy_peak_offset_fractional_start_early_peak_is_not_clamped():
    """Regression for the int()-truncation defect: a fractional window start
    (12.4) must not make an early, genuinely-in-window peak masquerade as an
    out-of-window one that gets clamped to 0.0."""
    t = _fractional_window_transcript()
    energy = [0.1] * 35
    energy[13] = 0.99  # first fully-contained bin (window: [12.4, 29.9))
    w = build_windows(t, energy, duration=40.0)
    first = w[0]
    assert first["start"] == pytest.approx(12.4)
    expected = ((13 + 0.5) - 12.4) / (29.9 - 12.4)
    assert first["energy_peak_offset"] == pytest.approx(expected)
    assert first["energy_peak_offset"] > 0.0


def test_energy_peak_offset_fractional_start_late_peak():
    t = _fractional_window_transcript()
    energy = [0.1] * 35
    energy[28] = 0.99  # last fully-contained bin (window: [12.4, 29.9))
    w = build_windows(t, energy, duration=40.0)
    first = w[0]
    expected = ((28 + 0.5) - 12.4) / (29.9 - 12.4)
    assert first["energy_peak_offset"] == pytest.approx(expected)
    assert first["energy_peak_offset"] > 0.85


def test_energy_no_pre_window_leak():
    """A spike in the bin straddling the window's start (mostly BEFORE 12.4)
    must not be picked up as this window's peak."""
    t = _fractional_window_transcript()
    energy = [0.1] * 35
    energy[12] = 0.99  # bin [12, 13) -- only [12.4, 13) of it is inside the window
    w = build_windows(t, energy, duration=40.0)
    first = w[0]
    assert first["energy_peak"] == pytest.approx(0.1)


def test_energy_in_window_tail_not_dropped():
    """A spike in the final bin fully inside the window ([28, 29) subset of
    [12.4, 29.9)) must still be counted, not dropped at the tail."""
    t = _fractional_window_transcript()
    energy = [0.1] * 35
    energy[28] = 0.99
    w = build_windows(t, energy, duration=40.0)
    first = w[0]
    assert first["energy_peak"] == pytest.approx(0.99)


def test_energy_degenerate_short_window_falls_back_without_raising():
    """A window shorter than the distance to the next bin boundary contains no
    fully-enclosed bin; it must fall back to the midpoint bin with offset 0.5
    instead of raising or deriving a masked/clamped value."""
    words = [{"word": "a", "start": 12.4, "end": 12.6, "score": 0.9, "speaker": "SPEAKER_00"}]
    t = {"language": "en", "model": "large-v3", "diarized": True,
         "segments": [{"start": 12.4, "end": 12.6, "speaker": "SPEAKER_00",
                       "text": "a", "words": words}]}
    energy = [0.1] * 20
    energy[12] = 0.9
    w = build_windows(t, energy, duration=13.0)
    assert len(w) == 1
    first = w[0]
    assert first["energy_peak"] == pytest.approx(0.9)
    assert first["energy_peak_offset"] == pytest.approx(0.5)


def test_window_text_joins_cjk_without_spaces():
    words = [{"word": ch, "start": i * 0.3, "end": i * 0.3 + 0.25, "score": 0.9,
              "speaker": "SPEAKER_00"} for i, ch in enumerate("今天我们聊一聊露营" * 20)]
    transcript = {"language": "zh", "model": "m", "diarized": False,
                  "segments": [{"start": 0.0, "end": words[-1]["end"], "speaker": "SPEAKER_00",
                                "text": "", "words": words}]}
    windows = build_windows(transcript, [0.5] * 60, 60.0)
    assert "今天我们聊一聊露营" in windows[0]["text"]


def test_the_windows_stage_ignores_what_is_not_really_speech(tmp_path):
    """Whisper's inventions over game audio (the real run) must not reach Laya."""
    from clipper.run import Run
    from clipper.window import write_windows

    run = Run.create(tmp_path, "game")
    run.write_json("source.json", {"duration": 160.0, "energy": [0.5] * 160})
    words = [{"word": "dd-d-d-d-d-d", "start": 40.0, "end": 69.0, "score": 0.0,
              "speaker": "SPEAKER_00"}]
    words += [{"word": "meddwl", "start": 69.0 + k * 11 / 6, "end": 69.0 + (k + 1) * 11 / 6,
               "score": 0.0, "speaker": "SPEAKER_00"} for k in range(6)]
    run.write_json("transcript.json", {"language": "cy", "segments": [
        {"start": 40.0, "end": 80.0, "speaker": "SPEAKER_00", "text": "", "words": words}]})
    assert write_windows(run) == []
