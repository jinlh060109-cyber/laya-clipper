import json
from pathlib import Path
import pytest
from clipper.transcribe import normalize_transcript

FIXTURE = Path(__file__).parent / "fixtures" / "whisperx_result.json"

def _result():
    return json.loads(FIXTURE.read_text())

def test_normalize_sets_metadata():
    out = normalize_transcript(_result(), model="large-v3", diarized=True, language="en")
    assert out["model"] == "large-v3"
    assert out["diarized"] is True
    assert out["language"] == "en"

def test_normalize_strips_leading_segment_whitespace():
    out = normalize_transcript(_result(), "large-v3", True, "en")
    assert out["segments"][0]["text"] == "So the thing nobody says"

def test_unaligned_words_are_interpolated_not_dropped():
    """WhisperX omits timings for words it cannot align. Dropping them would
    corrupt caption text; leaving them untimed would crash the ASS renderer."""
    out = normalize_transcript(_result(), "large-v3", True, "en")
    words = out["segments"][0]["words"]
    assert [w["word"] for w in words] == ["So", "the", "thing", "nobody", "says"]
    thing = words[2]
    assert 872.72 <= thing["start"] <= thing["end"] <= 873.10
    assert thing["score"] == 0.0

def test_missing_speaker_defaults_to_speaker_zero():
    result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hi",
                            "words": [{"word": "hi", "start": 0.0, "end": 1.0, "score": 0.9}]}]}
    out = normalize_transcript(result, "large-v3", diarized=False, language="en")
    assert out["segments"][0]["speaker"] == "SPEAKER_00"
    assert out["segments"][0]["words"][0]["speaker"] == "SPEAKER_00"

def test_empty_segments_raise():
    with pytest.raises(ValueError, match="no speech"):
        normalize_transcript({"segments": []}, "large-v3", False, "en")


def test_two_word_unaligned_run_splits_evenly_without_shrinking_gap():
    """Regression for a bug where `gap` was recomputed against mutating state,
    so filling the first unaligned word shrank the denominator for the next
    one in the same run, making it engulf its neighbour."""
    result = {"segments": [{"start": 9.0, "end": 14.0, "text": "a x y b",
                            "words": [
                                {"word": "a", "start": 9.5, "end": 10.0, "score": 0.9},
                                {"word": "x"},
                                {"word": "y"},
                                {"word": "b", "start": 13.0, "end": 13.5, "score": 0.9},
                            ]}]}
    out = normalize_transcript(result, "large-v3", False, "en")
    words = out["segments"][0]["words"]
    x, y = words[1], words[2]
    assert x["start"] == pytest.approx(10.0)
    assert x["end"] == pytest.approx(11.5)
    assert y["start"] == pytest.approx(11.5)
    assert y["end"] == pytest.approx(13.0)
    assert x["score"] == 0.0 and y["score"] == 0.0


def test_three_word_unaligned_run_divides_evenly():
    result = {"segments": [{"start": 19.0, "end": 24.0, "text": "l p q r h",
                            "words": [
                                {"word": "l", "start": 19.5, "end": 20.0, "score": 0.9},
                                {"word": "p"},
                                {"word": "q"},
                                {"word": "r"},
                                {"word": "h", "start": 23.0, "end": 23.5, "score": 0.9},
                            ]}]}
    out = normalize_transcript(result, "large-v3", False, "en")
    p, q, r = out["segments"][0]["words"][1:4]
    assert (p["start"], p["end"]) == pytest.approx((20.0, 21.0))
    assert (q["start"], q["end"]) == pytest.approx((21.0, 22.0))
    assert (r["start"], r["end"]) == pytest.approx((22.0, 23.0))


def test_mixed_run_words_stay_monotonic_and_fully_timed():
    result = {"segments": [{"start": 0.0, "end": 20.0, "text": "a x y b p q r c",
                            "words": [
                                {"word": "a", "start": 1.0, "end": 1.5, "score": 0.9},
                                {"word": "x"},
                                {"word": "y"},
                                {"word": "b", "start": 5.0, "end": 5.5, "score": 0.9},
                                {"word": "p"},
                                {"word": "q"},
                                {"word": "r"},
                                {"word": "c", "start": 12.0, "end": 12.5, "score": 0.9},
                            ]}]}
    out = normalize_transcript(result, "large-v3", False, "en")
    words = out["segments"][0]["words"]
    for w in words:
        assert isinstance(w["start"], float)
        assert isinstance(w["end"], float)
        assert w["start"] <= w["end"]
    for prev_w, next_w in zip(words, words[1:]):
        assert prev_w["end"] <= next_w["start"] + 1e-9


def test_leading_and_trailing_unaligned_runs_anchor_to_segment_bounds():
    result = {"segments": [{"start": 5.0, "end": 15.0, "text": "c d e f g",
                            "words": [
                                {"word": "c"},
                                {"word": "d"},
                                {"word": "e", "start": 8.0, "end": 8.5, "score": 0.9},
                                {"word": "f"},
                                {"word": "g"},
                            ]}]}
    out = normalize_transcript(result, "large-v3", False, "en")
    c, d, e, f, g = out["segments"][0]["words"]
    assert (c["start"], c["end"]) == pytest.approx((5.0, 6.5))
    assert (d["start"], d["end"]) == pytest.approx((6.5, 8.0))
    assert (f["start"], f["end"]) == pytest.approx((8.5, 11.75))
    assert (g["start"], g["end"]) == pytest.approx((11.75, 15.0))
    words = out["segments"][0]["words"]
    for prev_w, next_w in zip(words, words[1:]):
        assert prev_w["end"] <= next_w["start"] + 1e-9


def _fake_whisperx(monkeypatch, diarize_calls):
    import sys
    import types

    result = json.loads(FIXTURE.read_text(encoding="utf-8"))
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    wx = types.ModuleType("whisperx")
    wx.load_audio = lambda path: "audio"
    wx.load_model = lambda *a, **k: types.SimpleNamespace(
        transcribe=lambda audio, batch_size: {"language": "en", "segments": result["segments"]})
    wx.load_align_model = lambda **k: ("align", {})
    wx.align = lambda segments, *a, **k: {"segments": segments}
    wx.assign_word_speakers = lambda diarized, result: result
    diarize = types.ModuleType("whisperx.diarize")

    class DiarizationPipeline:
        def __init__(self, use_auth_token, device):
            diarize_calls.append(use_auth_token)

        def __call__(self, audio):
            return "segments"

    diarize.DiarizationPipeline = DiarizationPipeline
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "whisperx", wx)
    monkeypatch.setitem(sys.modules, "whisperx.diarize", diarize)


def test_no_token_means_no_diarization_even_when_hf_token_is_in_the_environment(
        tmp_path, monkeypatch):
    """Callers pass hf_token=None to switch diarization off (--no-diarize, the web toggle)."""
    from clipper.run import Run
    from clipper.transcribe import transcribe

    calls = []
    _fake_whisperx(monkeypatch, calls)
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    out = transcribe(tmp_path / "audio.wav", Run.create(tmp_path, "r"), hf_token=None)
    assert calls == []
    assert out["diarized"] is False


def test_a_passed_token_diarizes(tmp_path, monkeypatch):
    from clipper.run import Run
    from clipper.transcribe import transcribe

    calls = []
    _fake_whisperx(monkeypatch, calls)
    out = transcribe(tmp_path / "audio.wav", Run.create(tmp_path, "r"), hf_token="hf_x")
    assert calls == ["hf_x"]
    assert out["diarized"] is True
