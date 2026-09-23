import pytest

from clipper import style


@pytest.fixture(autouse=True)
def style_file(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPPER_STYLE", str(tmp_path / "style.json"))


def test_defaults_when_nothing_is_saved():
    assert style.load() == style.DEFAULT
    assert style.DEFAULT["layout"] == "fit"


def test_save_round_trips_and_ignores_unknown_fields():
    saved = style.save({"layout": "crop", "caption_case": "upper", "notes": "Punchy.",
                        "font": "Comic Sans"})
    assert saved["layout"] == "crop" and "font" not in saved
    assert style.load() == saved


@pytest.mark.parametrize("field, value", [
    ("layout", "zoom"), ("captions", "loud"), ("caption_case", "title"), ("encoder", "h266")])
def test_invalid_choices_are_refused(field, value):
    with pytest.raises(ValueError, match=field):
        style.save({field: value})


def test_notes_have_a_length_limit():
    with pytest.raises(ValueError, match="notes"):
        style.save({"notes": "x" * 4001})


def test_a_corrupt_style_file_falls_back_to_defaults(tmp_path):
    (tmp_path / "style.json").write_text("{not json", encoding="utf-8")
    assert style.load() == style.DEFAULT
