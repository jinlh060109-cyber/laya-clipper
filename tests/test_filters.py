from pathlib import Path
import pytest
from clipper.filters import build_filter_chain, crop_expression, subtitles_expression

STAGED = Path("C:/tmp/clipper_x/c.ass")  # never read; must have no spaces

def test_crop_expression_is_nine_by_sixteen_of_the_height():
    assert crop_expression(1920, 1080, "center") == "crop=607:1080:656:0"

def test_explicit_crop_x_is_honoured():
    assert crop_expression(1920, 1080, 100).endswith(":100:0")

def test_crop_x_is_clamped_into_frame():
    assert crop_expression(1920, 1080, 5000).split(":")[2] == str(1920 - 607)
    assert crop_expression(1920, 1080, -50).split(":")[2] == "0"

def test_source_narrower_than_nine_by_sixteen_is_not_cropped():
    """Already-vertical footage must not be cropped to nothing."""
    assert crop_expression(1080, 1920, "center") == ""

def test_filter_order_is_crop_then_scale_then_subtitles():
    chain = build_filter_chain(1920, 1080, vertical=True, subtitle_path=STAGED)
    assert chain.index("crop=") < chain.index("scale=1080:1920") < chain.index("subtitles=")

def test_horizontal_has_no_crop_or_scale():
    chain = build_filter_chain(1920, 1080, vertical=False, subtitle_path=STAGED)
    assert "crop=" not in chain
    assert "scale=" not in chain
    assert "subtitles=" in chain

def test_no_filters_at_all_returns_none():
    assert build_filter_chain(1920, 1080, vertical=False, subtitle_path=None) is None

def test_subtitles_expression_escapes_windows_drive_colon():
    expr = subtitles_expression(STAGED)
    assert "C\\:" in expr or "C\\\\:" in expr

def test_subtitles_expression_rejects_a_path_with_spaces():
    """ffmpeg truncates such paths at the first space. Stage to a temp dir."""
    with pytest.raises(ValueError, match="space"):
        subtitles_expression(Path("C:/Users/Linhao Jin/c.ass"))


def test_fit_layout_burns_captions_after_the_overlay():
    chain = build_filter_chain(1920, 1080, True, Path("sub.ass"), layout="fit")
    assert chain.endswith("subtitles=sub.ass")
    assert chain.index("overlay") < chain.index("subtitles")


def test_fit_layout_on_a_horizontal_render_is_just_captions():
    assert build_filter_chain(1920, 1080, False, Path("s.ass"), layout="fit") == "subtitles=s.ass"


def test_unknown_layout_is_rejected():
    with pytest.raises(ValueError, match="layout"):
        build_filter_chain(1920, 1080, True, None, layout="zoom")


def test_fit_layout_hands_the_encoder_plain_420_frames():
    """Intel Quick Sync painted half the blurred background green when it got
    the overlay's output format directly; converting first fixes it."""
    chain = build_filter_chain(1920, 1080, True, None, layout="fit")
    assert chain.endswith("format=yuv420p")
