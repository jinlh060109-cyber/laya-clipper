import pytest
from clipper.captions import (
    Cue, build_cues, format_srt_time, rebase, render_srt, words_in_range,
)

def _words(n=12, start=100.0, dur=0.4, speaker="SPEAKER_00"):
    return [{"word": f"word{i}", "start": start + i * dur,
             "end": start + i * dur + dur * 0.9, "score": 0.9, "speaker": speaker}
            for i in range(n)]

def _transcript(words):
    return {"language": "en", "model": "m", "diarized": True,
            "segments": [{"start": words[0]["start"], "end": words[-1]["end"],
                          "speaker": words[0]["speaker"], "text": "x", "words": words}]}

def test_words_in_range_selects_overlapping_words_only():
    t = _transcript(_words(20))
    got = words_in_range(t, 102.0, 104.0)
    assert all(w["end"] > 102.0 and w["start"] < 104.0 for w in got)
    assert got

def test_build_cues_breaks_on_character_limit():
    cues = build_cues(_words(30), max_chars=20)
    assert all(len(c.text) <= 20 for c in cues)

def test_build_cues_breaks_on_duration_limit():
    cues = build_cues(_words(30, dur=0.5), max_chars=500, max_seconds=2.0)
    assert all(c.end - c.start <= 2.0 + 1e-6 for c in cues)

def test_build_cues_breaks_on_speaker_change():
    words = _words(6, speaker="SPEAKER_00") + _words(6, start=103.0, speaker="SPEAKER_01")
    cues = build_cues(words, max_chars=500, max_seconds=60.0)
    assert len(cues) == 2
    assert cues[0].speaker == "SPEAKER_00"
    assert cues[1].speaker == "SPEAKER_01"

def test_rebase_subtracts_clip_start():
    cues = build_cues(_words(6))
    out = rebase(cues, clip_start=100.0)
    assert out[0].start == pytest.approx(0.0, abs=0.01)

def test_rebase_clamps_negative_times_to_zero():
    """A cue that starts before the cut must not produce a negative timestamp."""
    cue = Cue(start=99.0, end=101.0, words=_words(2, start=99.0), speaker="SPEAKER_00")
    assert rebase([cue], clip_start=100.0)[0].start == 0.0

def test_rebase_drops_cues_entirely_before_the_cut():
    cue = Cue(start=90.0, end=95.0, words=_words(2, start=90.0), speaker="SPEAKER_00")
    assert rebase([cue], clip_start=100.0) == []

def test_format_srt_time_uses_comma_separator():
    assert format_srt_time(3661.25) == "01:01:01,250"

def test_format_srt_time_handles_zero():
    assert format_srt_time(0.0) == "00:00:00,000"

def test_render_srt_numbers_cues_from_one():
    out = render_srt(build_cues(_words(12), max_chars=20))
    assert out.startswith("1\n")
    assert "\n2\n" in out

def test_render_srt_ends_with_a_blank_line():
    assert render_srt(build_cues(_words(4))).endswith("\n\n")


def _tokens(*texts, start=0.0, dur=0.3):
    return [{"word": t, "start": start + i * dur, "end": start + (i + 1) * dur,
             "score": 0.9, "speaker": "SPEAKER_00"} for i, t in enumerate(texts)]


def test_cjk_characters_join_without_spaces():
    """WhisperX aligns zh/ja per character; spaces between them read as broken text."""
    from clipper.captions import join_words
    assert join_words(["我", "是", "谁", "？"]) == "我是谁？"
    assert join_words(["これ", "は", "ペン"]) == "これはペン"
    assert join_words(["我", "用", "iPhone", "拍", "的"]) == "我用 iPhone 拍的"
    assert join_words(["hello", "world"]) == "hello world"
    assert join_words(["안녕하세요", "여러분"]) == "안녕하세요 여러분"


def test_cjk_cues_count_characters_not_inserted_spaces():
    words = _tokens(*("字" * 20))
    cues = build_cues(words, max_chars=42, max_seconds=100.0)
    assert len(cues) == 1
    assert cues[0].text == "字" * 20


def test_cjk_characters_count_double_width_so_lines_fit_the_frame():
    cues = build_cues(_tokens(*("字" * 30)), max_chars=42, max_seconds=100.0)
    assert len(cues) == 2
    assert all(len(c.text) <= 21 for c in cues)


def test_cjk_karaoke_has_no_spaces_between_characters():
    from clipper.captions import render_ass
    ass = render_ass(build_cues(_tokens("我", "是", "谁")), 1920)
    line = next(l for l in ass.splitlines() if l.startswith("Dialogue"))
    assert "我{" in line and " {" not in line.split(",,", 1)[1]
