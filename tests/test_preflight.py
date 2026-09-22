import subprocess

import pytest
from pathlib import Path
from clipper.preflight import (
    PreflightError, Preflight, find_binary, has_subtitles_filter, preflight,
)

def test_find_binary_prefers_env_var(tmp_path, monkeypatch):
    fake = tmp_path / "ffmpeg.exe"
    fake.write_text("")
    monkeypatch.setenv("FFMPEG_PATH", str(fake))
    assert find_binary("ffmpeg", "FFMPEG_PATH") == fake

def test_find_binary_raises_when_env_var_path_missing(monkeypatch):
    monkeypatch.setenv("FFMPEG_PATH", "/nonexistent/ffmpeg")

    def fail_which(_):
        raise AssertionError("shutil.which should not be called when the env var is set but invalid")

    monkeypatch.setattr("shutil.which", fail_which)
    with pytest.raises(PreflightError) as err:
        find_binary("ffmpeg", "FFMPEG_PATH")
    message = str(err.value)
    assert "FFMPEG_PATH" in message
    assert "/nonexistent/ffmpeg" in message

def test_find_binary_falls_back_to_path_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("FFMPEG_PATH", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ffmpeg")
    assert find_binary("ffmpeg", "FFMPEG_PATH") == Path("/usr/bin/ffmpeg")

def test_find_binary_raises_actionable_error_when_not_found_anywhere(monkeypatch):
    monkeypatch.delenv("FFMPEG_PATH", raising=False)
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

def test_has_subtitles_filter_raises_when_ffmpeg_exits_nonzero(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0] if args else [],
            returncode=1,
            stdout="",
            stderr="error while loading shared libraries: libx264.so.164: cannot open shared object file",
        )

    monkeypatch.setattr("clipper.preflight.subprocess.run", fake_run)
    with pytest.raises(PreflightError) as err:
        has_subtitles_filter(Path("ffmpeg"))
    message = str(err.value).lower()
    assert "libass" not in message
    assert "exited" in message or "failed" in message

def test_preflight_raises_libass_message_when_exit_zero_no_subtitles(monkeypatch, tmp_path):
    ffmpeg = tmp_path / "ffmpeg"
    ffprobe = tmp_path / "ffprobe"
    ffmpeg.write_text("")
    ffprobe.write_text("")
    monkeypatch.setenv("FFMPEG_PATH", str(ffmpeg))
    monkeypatch.setenv("FFPROBE_PATH", str(ffprobe))
    monkeypatch.setattr("clipper.preflight._run_filters", lambda _: "... scale ...")
    with pytest.raises(PreflightError) as err:
        preflight()
    assert "libass" in str(err.value).lower()

def test_preflight_success_without_subtitles_requirement(monkeypatch):
    fake_ffmpeg = Path("/fake/ffmpeg")
    fake_ffprobe = Path("/fake/ffprobe")

    def fake_find_binary(name: str, env_var: str) -> Path:
        return fake_ffmpeg if name == "ffmpeg" else fake_ffprobe

    monkeypatch.setattr("clipper.preflight.find_binary", fake_find_binary)
    result = preflight(require_subtitles=False)
    assert result == Preflight(ffmpeg=fake_ffmpeg, ffprobe=fake_ffprobe)
