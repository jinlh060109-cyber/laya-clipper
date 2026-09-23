import shutil
import subprocess

import pytest

from clipper.frames import contact_sheet, sheet_times

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def test_times_are_centred_in_six_equal_slices():
    assert sheet_times(10.0, 22.0) == [11.0, 13.0, 15.0, 17.0, 19.0, 21.0]


def make_video(path):
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi",
                    "-i", "testsrc2=s=640x360:d=6:r=10", str(path)], check=True)


@needs_ffmpeg
def test_a_sheet_is_three_by_two_frames_of_320px(tmp_path):
    video = tmp_path / "v.mp4"
    make_video(video)
    out = tmp_path / "frames" / "action-00.jpg"
    times = contact_sheet(shutil.which("ffmpeg"), video, 0.0, 6.0, out)
    assert times == sheet_times(0.0, 6.0)
    probe = subprocess.run([shutil.which("ffprobe"), "-v", "error", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", str(out)],
                           capture_output=True, text=True, check=True)
    assert probe.stdout.strip() == "960,360"
    assert [p.name for p in out.parent.iterdir()] == ["action-00.jpg"]


@needs_ffmpeg
def test_paths_with_apostrophes_and_non_ascii_work(tmp_path):
    base = tmp_path / "Jev's 游戏 run"
    base.mkdir()
    video = base / "v.mp4"
    make_video(video)
    out = base / "frames" / "action-00.jpg"
    contact_sheet(shutil.which("ffmpeg"), video, 1.0, 5.0, out)
    assert out.stat().st_size > 0


@needs_ffmpeg
def test_a_missing_video_is_a_runtime_error(tmp_path):
    with pytest.raises(RuntimeError, match="frame"):
        contact_sheet(shutil.which("ffmpeg"), tmp_path / "nope.mp4", 0, 6, tmp_path / "s.jpg")
