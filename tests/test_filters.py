from pathlib import Path
import pytest
from clipper.filters import build_filter_chain, crop_expression, subtitles_expression

STAGED = Path("C:/tmp/clipper_x/c.ass")

def test_crop_expression_is_nine_by_sixteen_of_the_height():
    assert crop_expression(1920, 1080, "center") == "crop=607:1080:656:0"

def test_crop_centers_by_default():
    expr = crop_expression(1920, 1080, "center")
    x = int(expr.split(":")[2])
    assert x == (1920 - 607) // 2

def test_explicit_crop_x_is_honoured():
    assert crop_expression(1920, 1080, 100).endswith(":100:0")

def test_crop_x_is_clamped_into_frame():
    assert crop_expression(1920, 1080, 5000).split(":")[2] == str(1920 - 607)
    assert crop_expression(1920, 1080, -50).split(":")[2] == "0"

def test_source_narrower_than_nine_by_sixteen_is_not_cropped():
    """Already-vertical footage must not be cropped to nothing."""
    assert crop_expression(1080, 1920, "center") == ""

def test_filter_order_is_crop_then_scale_then_subtitles(tmp_path):
    sub = STAGED  # never read; tmp_path may contain a space (e.g. a Windows username)
    chain = build_filter_chain(1920, 1080, vertical=True, subtitle_path=sub)
    assert chain.index("crop=") < chain.index("scale=") < chain.index("subtitles=")

def test_vertical_scales_to_1080x1920(tmp_path):
    chain = build_filter_chain(1920, 1080, vertical=True, subtitle_path=None)
    assert "scale=1080:1920" in chain

def test_horizontal_has_no_crop_or_scale(tmp_path):
    sub = STAGED  # never read; tmp_path may contain a space (e.g. a Windows username)
    chain = build_filter_chain(1920, 1080, vertical=False, subtitle_path=sub)
    assert "crop=" not in chain
    assert "scale=" not in chain
    assert "subtitles=" in chain

def test_no_filters_at_all_returns_none():
    assert build_filter_chain(1920, 1080, vertical=False, subtitle_path=None) is None

def test_subtitles_expression_escapes_windows_drive_colon(tmp_path):
    expr = subtitles_expression(Path("C:/tmp/clipper_x/c.ass"))
    assert "C\\:" in expr or "C\\\\:" in expr

def test_subtitles_expression_rejects_a_path_with_spaces():
    """ffmpeg truncates such paths at the first space. Stage to a temp dir."""
    with pytest.raises(ValueError, match="space"):
        subtitles_expression(Path("C:/Users/Linhao Jin/c.ass"))


def test_fit_layout_keeps_the_whole_frame_over_a_blurred_copy():
    from clipper.filters import build_filter_chain
    chain = build_filter_chain(1920, 1080, True, None, layout="fit")
    assert "split=2" in chain and "boxblur" in chain
    assert "overlay=(W-w)/2:(H-h)/2" in chain
    assert "crop=607" not in chain  # the picture itself is never cropped


def test_fit_layout_burns_captions_after_the_overlay():
    from pathlib import Path
    from clipper.filters import build_filter_chain
    chain = build_filter_chain(1920, 1080, True, Path("sub.ass"), layout="fit")
    assert chain.endswith("subtitles=sub.ass")
    assert chain.index("overlay") < chain.index("subtitles")


def test_fit_layout_on_a_horizontal_render_is_just_captions():
    from pathlib import Path
    from clipper.filters import build_filter_chain
    assert build_filter_chain(1920, 1080, False, Path("s.ass"), layout="fit") == "subtitles=s.ass"


def test_unknown_layout_is_rejected():
    import pytest
    from clipper.filters import build_filter_chain
    with pytest.raises(ValueError, match="layout"):
        build_filter_chain(1920, 1080, True, None, layout="zoom")


def test_fit_layout_hands_the_encoder_plain_420_frames():
    """Intel Quick Sync painted half the blurred background green when it got
    the overlay's output format directly; converting first fixes it."""
    from clipper.filters import build_filter_chain
    chain = build_filter_chain(1920, 1080, True, None, layout="fit")
    assert chain.endswith("format=yuv420p")
