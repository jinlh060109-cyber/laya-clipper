from pathlib import Path
import pytest
from clipper.filters import build_filter_chain, crop_expression, subtitles_expression

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
    sub = tmp_path / "c.ass"
    sub.write_text("")
    chain = build_filter_chain(1920, 1080, vertical=True, subtitle_path=sub)
    assert chain.index("crop=") < chain.index("scale=") < chain.index("subtitles=")

def test_vertical_scales_to_1080x1920(tmp_path):
    chain = build_filter_chain(1920, 1080, vertical=True, subtitle_path=None)
    assert "scale=1080:1920" in chain

def test_horizontal_has_no_crop_or_scale(tmp_path):
    sub = tmp_path / "c.ass"
    sub.write_text("")
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
