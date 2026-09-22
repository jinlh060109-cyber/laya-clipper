import pytest
from pathlib import Path
from clipper.preflight import (
    PreflightError, find_binary, has_subtitles_filter,
)

def test_find_binary_prefers_env_var(tmp_path, monkeypatch):
    fake = tmp_path / "ffmpeg.exe"
    fake.write_text("")
    monkeypatch.setenv("FFMPEG_PATH", str(fake))
    assert find_binary("ffmpeg", "FFMPEG_PATH") == fake

def test_find_binary_raises_actionable_error(monkeypatch):
    monkeypatch.setenv("FFMPEG_PATH", "/nonexistent/ffmpeg")
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(PreflightError) as err:
        find_binary("ffmpeg", "FFMPEG_PATH")
    assert "install" in str(err.value).lower()

def test_has_subtitles_filter_detects_presence(monkeypatch):
    monkeypatch.setattr(
        "clipper.preflight._run_filters",
        lambda _: " T.. subtitles         V->V       Render text subtitles onto input video using the libass library.",
    )
    assert has_subtitles_filter(Path("ffmpeg")) is True

def test_has_subtitles_filter_detects_absence(monkeypatch):
    monkeypatch.setattr("clipper.preflight._run_filters", lambda _: "... scale ...")
    assert has_subtitles_filter(Path("ffmpeg")) is False
