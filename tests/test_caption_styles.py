import re

import pytest

from clipper.captions import CAPTION_STYLES, build_cues, cues_for, render_ass


def _words(text, step=0.3):
    return [{"word": w, "start": i * step, "end": i * step + 0.25, "score": 0.9,
             "speaker": "S"} for i, w in enumerate(text.split())]


WORDS = _words("Here is the trick. It works every single time you try it.")


def _style_line(ass):
    return next(line for line in ass.splitlines() if line.startswith("Style: Caption,"))


def _field(line, index):
    return line.split(":", 1)[1].split(",")[index]


def test_there_are_several_distinct_presets_with_names_for_the_page():
    assert {"classic", "bold", "boxed", "minimal", "one_word", "neon"} <= set(CAPTION_STYLES)
    assert all(preset["label"] for preset in CAPTION_STYLES.values())
    lines = {_style_line(render_ass(build_cues(WORDS), 1920, preset=name))
             for name in CAPTION_STYLES}
    assert len(lines) == len(CAPTION_STYLES)


def test_boxed_puts_the_words_on_a_box():
    line = _style_line(render_ass(build_cues(WORDS), 1920, preset="boxed"))
    assert _field(line, 15).strip() == "3"  # BorderStyle 3 = opaque box


def test_presets_without_highlighting_have_no_karaoke_colour_change():
    line = _style_line(render_ass(build_cues(WORDS), 1920, preset="minimal"))
    assert _field(line, 3) == _field(line, 4)  # primary == secondary colour


def test_one_word_style_shows_a_single_word_at_a_time_and_pops_it():
    cues = cues_for(WORDS, "one_word")
    assert all(len(cue.words) == 1 for cue in cues)
    ass = render_ass(cues, 1920, preset="one_word")
    assert "\\t(" in ass and "HERE" in ass  # scale-in animation, uppercase


def test_bold_style_is_uppercase_and_short_lines():
    cues = cues_for(WORDS, "bold")
    assert max(len(c.text) for c in cues) <= 20
    assert "TRICK" in render_ass(cues, 1920, preset="bold")


def test_vertical_captions_are_big_enough_to_read_on_a_phone():
    for name in CAPTION_STYLES:
        size = int(_field(_style_line(render_ass(build_cues(WORDS), 1920, preset=name)), 2))
        assert size >= 56, name


def test_an_unknown_preset_is_refused():
    with pytest.raises(ValueError, match="caption style"):
        render_ass(build_cues(WORDS), 1920, preset="comic-sans")


def test_the_style_setting_accepts_every_preset_and_refuses_others(tmp_path, monkeypatch):
    from clipper import style
    monkeypatch.setenv("CLIPPER_STYLE", str(tmp_path / "s.json"))
    assert style.DEFAULT["caption_style"] == "classic"
    for name in CAPTION_STYLES:
        assert style.save({"caption_style": name})["caption_style"] == name
    with pytest.raises(ValueError, match="caption_style"):
        style.save({"caption_style": "wobbly"})


def test_classic_keeps_its_karaoke_highlight():
    ass = render_ass(build_cues(WORDS), 1920, preset="classic")
    line = _style_line(ass)
    assert _field(line, 3) != _field(line, 4)
    assert re.search(r"\{\\k\d+\}Here", ass)


def test_every_style_line_fits_inside_a_vertical_frame():
    """Classic ran edge to edge: 42 characters at 64 px is wider than 1080 px."""
    from clipper.captions import font_size_for_height
    usable = 1080 - 2 * 60  # the side margins
    for name, look in CAPTION_STYLES.items():
        if look["max_chars"]:
            width = look["max_chars"] * font_size_for_height(1920) * look["scale"] * 0.55
            assert width <= usable, name


def test_libass_may_wrap_a_line_that_still_runs_long():
    assert "WrapStyle: 0" in render_ass(build_cues(WORDS), 1920)


def test_a_hook_is_a_top_title_for_the_first_seconds():
    ass = render_ass(build_cues(WORDS), 1920, preset="bold", hook="This raccoon stole my hat at 3 AM")
    title = next(line for line in ass.splitlines() if ",Title," in line and line.startswith("Dialogue"))
    assert title.startswith("Dialogue: 1,0:00:00.00,0:00:02.80,Title")
    assert "\\N" in title and "\\fad(" in title  # two lines, fades out
    style = next(line for line in ass.splitlines() if line.startswith("Style: Title,"))
    assert style.split(",")[18] == "8"  # top centre


def test_no_hook_no_title():
    assert "Dialogue: 1," not in render_ass(build_cues(WORDS), 1920)
