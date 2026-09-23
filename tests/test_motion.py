import shutil
import subprocess

import pytest

from clipper.motion import measure_motion, parse_scdet

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def frame(t, mafd, score):
    return [f"[Parsed_metadata_3 @ 0] frame:0 pts:0 pts_time:{t}",
            f"[Parsed_metadata_3 @ 0] lavfi.scd.mafd={mafd}",
            f"[Parsed_metadata_3 @ 0] lavfi.scd.score={score}"]


def test_frames_are_averaged_per_whole_second_and_cuts_counted():
    lines = frame(0, 1.0, 0) + frame(0.5, 3.0, 0) + frame(1.0, 20.0, 19.0) + frame(1.5, 4.0, 2.0)
    raw, cuts = parse_scdet(lines, 3)
    assert raw == [2.0, 12.0, 12.0]  # second 2 has no frames: carried forward
    assert cuts == [0, 1, 0]


def test_scdets_own_summary_line_is_not_a_second_cut():
    lines = frame(1.0, 20.0, 19.0) + [
        "[Parsed_scdet_2 @ 0] lavfi.scd.score: 19.051, lavfi.scd.time: 1"]
    assert parse_scdet(lines, 2)[1] == [0, 1]


def test_frames_past_the_last_second_are_ignored_and_progress_is_reported():
    seen = []
    raw, cuts = parse_scdet(frame(0, 1, 0) + frame(1, 1, 0) + frame(9, 5, 50),
                            2, progress=lambda d, t: seen.append((d, t)))
    assert seen == [(1, 2), (2, 2)]
    assert cuts == [0, 0]


@needs_ffmpeg
def test_real_video_moves_more_after_a_static_start_and_the_join_is_a_cut(tmp_path):
    video = tmp_path / "v.mp4"
    subprocess.run([shutil.which("ffmpeg"), "-v", "error",
                    "-f", "lavfi", "-i", "color=c=gray:s=320x180:d=4:r=10",
                    "-f", "lavfi", "-i", "testsrc2=s=320x180:d=4:r=10",
                    "-filter_complex", "[0][1]concat=n=2:v=1[v]", "-map", "[v]",
                    str(video)], check=True)
    out = measure_motion(shutil.which("ffmpeg"), video, 8)
    assert len(out["motion_raw"]) == len(out["motion"]) == len(out["cuts"]) == 8
    assert max(out["motion_raw"][:3]) < 0.5 < min(out["motion_raw"][5:])
    assert sum(out["cuts"][3:6]) >= 1


@needs_ffmpeg
def test_ffmpeg_failure_is_a_runtime_error_quoting_ffmpeg(tmp_path):
    with pytest.raises(RuntimeError, match="Motion"):
        measure_motion(shutil.which("ffmpeg"), tmp_path / "missing.mp4", 5)
