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
