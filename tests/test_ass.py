import pytest
from clipper.captions import (
    Cue, font_size_for_height, format_ass_time, render_ass,
)

def _cue(start=0.0, n=3, dur=0.4, speaker="SPEAKER_00"):
    words = [{"word": f"w{i}", "start": start + i * dur,
              "end": start + (i + 1) * dur, "score": 0.9, "speaker": speaker}
             for i in range(n)]
    return Cue(start=words[0]["start"], end=words[-1]["end"],
               words=words, speaker=speaker)

def test_font_size_matches_reference_table():
    assert font_size_for_height(720) == 20
    assert font_size_for_height(1080) == 24
    assert font_size_for_height(2160) == 48

def test_vertical_output_gets_large_captions():
    """1080x1920 shorts want big captions, not 1080p-horizontal sizing."""
    assert font_size_for_height(1920) >= 40

def test_format_ass_time_uses_centiseconds_and_one_hour_digit():
    assert format_ass_time(3661.25) == "1:01:01.25"
    assert format_ass_time(0.0) == "0:00:00.00"

def test_ass_has_required_sections():
    out = render_ass([_cue()], height=1080)
    assert "[Script Info]" in out
    assert "[V4+ Styles]" in out
    assert "[Events]" in out

def test_ass_declares_play_resolution():
    out = render_ass([_cue()], height=1920)
    assert "PlayResY: 1920" in out

def test_each_word_gets_its_own_karaoke_tag():
    out = render_ass([_cue(n=3)], height=1080)
    dialogue = [l for l in out.splitlines() if l.startswith("Dialogue:")][0]
    assert dialogue.count("\\k") == 3

def test_karaoke_durations_are_in_centiseconds():
    out = render_ass([_cue(n=2, dur=0.5)], height=1080)
    assert "\\k50}" in out

def test_speaker_label_is_emitted_when_names_are_supplied():
    out = render_ass([_cue(speaker="SPEAKER_01")], height=1080,
                     speakers={"SPEAKER_01": "Marco"})
    assert "Marco" in out
    assert "Style: Speaker" in out

def test_no_speaker_label_without_names():
    out = render_ass([_cue(speaker="SPEAKER_01")], height=1080)
    assert "Marco" not in out
    assert "Dialogue:" in out

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
    assert "[Events]" in out

def test_karaoke_absorbs_pauses_so_highlighting_stays_in_sync():
    """libass advances \k cumulatively; a pause dropped from the sum drifts early."""
    words = [{"word": "a", "start": 0.0, "end": 0.3, "score": 0.9, "speaker": "S"},
             {"word": "b", "start": 1.0, "end": 1.4, "score": 0.9, "speaker": "S"}]
    out = render_ass([Cue(start=0.0, end=1.4, words=words, speaker="S")], height=1080)
    assert "{\k100}a {\k40}b" in out
