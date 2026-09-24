from clipper.captions import (
    Cue, format_ass_time, render_ass,
)

def _cue(start=0.0, n=3, dur=0.4, speaker="SPEAKER_00"):
    words = [{"word": f"w{i}", "start": start + i * dur,
              "end": start + (i + 1) * dur, "score": 0.9, "speaker": speaker}
             for i in range(n)]
    return Cue(start=words[0]["start"], end=words[-1]["end"],
               words=words, speaker=speaker)

def test_format_ass_time_uses_centiseconds_and_one_hour_digit():
    assert format_ass_time(3661.25) == "1:01:01.25"
    assert format_ass_time(0.0) == "0:00:00.00"

def test_speaker_label_is_emitted_when_names_are_supplied():
    out = render_ass([_cue(speaker="SPEAKER_01")], height=1080,
                     speakers={"SPEAKER_01": "Marco"})
    assert "Marco" in out
    assert "Style: Speaker" in out

def test_unmapped_speaker_produces_no_label():
    out = render_ass([_cue(speaker="SPEAKER_09")], height=1080,
                     speakers={"SPEAKER_01": "Marco"})
    assert "SPEAKER_09" not in out

def test_braces_in_text_are_escaped():
    """An unescaped brace would be parsed as an ASS override block."""
    cue = Cue(start=0.0, end=1.0, speaker="SPEAKER_00",
              words=[{"word": "{hi}", "start": 0.0, "end": 1.0,
                      "score": 0.9, "speaker": "SPEAKER_00"}])
    out = render_ass([cue], height=1080)
    assert "{hi}" not in out
    assert "(hi)" in out

def test_empty_cue_list_still_produces_a_valid_file():
    out = render_ass([], height=1080)
    assert "[Script Info]" in out and "[V4+ Styles]" in out and "[Events]" in out

def test_karaoke_absorbs_pauses_so_highlighting_stays_in_sync():
    r"""libass advances \k cumulatively; a pause dropped from the sum drifts early."""
    words = [{"word": "a", "start": 0.0, "end": 0.3, "score": 0.9, "speaker": "S"},
             {"word": "b", "start": 1.0, "end": 1.4, "score": 0.9, "speaker": "S"}]
    out = render_ass([Cue(start=0.0, end=1.4, words=words, speaker="S")], height=1080)
    assert r"{\k100}a {\k40}b" in out
