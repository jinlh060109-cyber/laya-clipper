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
    assert format_srt_time(0.0) == "00:00:00,000"

def test_render_srt_numbers_cues_from_one_and_ends_with_a_blank_line():
    out = render_srt(build_cues(_words(12), max_chars=20))
    assert out.startswith("1\n")
    assert "\n2\n" in out
    assert out.endswith("\n\n")


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


def test_cjk_characters_count_double_width_so_lines_fit_the_frame():
    cues = build_cues(_tokens(*("字" * 30)), max_chars=42, max_seconds=100.0)
    assert len(cues) == 2
    assert all(len(c.text) <= 21 for c in cues)
    assert "".join(c.text for c in cues) == "字" * 30  # no inserted spaces


def test_cjk_karaoke_has_no_spaces_between_characters():
    from clipper.captions import render_ass
    ass = render_ass(build_cues(_tokens("我", "是", "谁")), 1920)
    line = next(l for l in ass.splitlines() if l.startswith("Dialogue"))
    assert "我{" in line and " {" not in line.split(",,", 1)[1]


def _spoken(text, start=0.0, step=0.3):
    return [{"word": w, "start": start + i * step, "end": start + i * step + 0.25,
             "score": 0.9, "speaker": "SPEAKER_00"} for i, w in enumerate(text.split())]


def test_a_cue_ends_at_a_sentence_end_rather_than_running_on():
    cues = build_cues(_spoken("You need 97.2 hours. Obviously boosts help a lot."))
    assert cues[0].text == "You need 97.2 hours."


def test_a_number_is_never_left_at_the_end_of_a_cue():
    """The real clip read "between 97.2 hours and 67.9" / "hours to complete it"."""
    # The 3-second limit falls right after the tenth word, "67.9".
    words = _spoken("you need between 97.2 hours and about sixty or 67.9 "
                    "hours to complete it depending on boosts.")
    assert words[9]["word"] == "67.9"
    cues = build_cues(words, max_chars=500)
    assert [c.text for c in cues if c.text.split()[-1][0].isdigit()] == []
    assert " ".join(c.text for c in cues) == " ".join(w["word"] for w in words)


def test_long_sentences_prefer_breaking_after_a_comma():
    cues = build_cues(_spoken("So if you wanted to finish fast, you could grind for six days straight"),
                      max_chars=42)
    assert cues[0].text.endswith("fast,")


def test_uppercase_caption_style():
    from clipper.captions import render_ass
    ass = render_ass(build_cues(_spoken("Hello there friend.")), 1920, uppercase=True)
    assert "HELLO" in ass and "Hello" not in ass


def test_fit_layout_puts_captions_under_the_picture():
    import re
    from clipper.captions import render_ass
    cues = build_cues(_spoken("Hello there friend."))
    margin = lambda ass: int(re.search(r"Style: Caption,(?:[^,]*,){20}(\d+)", ass).group(1))
    assert margin(render_ass(cues, 1920, layout="fit")) > margin(render_ass(cues, 1920))
