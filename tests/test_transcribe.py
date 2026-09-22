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
