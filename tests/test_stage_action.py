import shutil
import subprocess

import pytest

import clipper.stages.action as stage
from clipper.run import Run

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def silent_run(tmp_path, seconds=30):
    run = Run.create(tmp_path, "game")
    video = run.path("video.mp4")
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi",
                    "-i", f"testsrc2=s=320x180:d={seconds}:r=10", str(video)], check=True)
    run.write_json("source.json", {"path": str(video), "duration": float(seconds),
                                   "energy": [0.5] * seconds})
    run.write_json("transcript.json", {"language": "en", "segments": []})
    return run


@needs_ffmpeg
def test_a_silent_video_gets_candidates_with_sheets(tmp_path):
    run = silent_run(tmp_path)
    result = stage.run_action(run)
    action = run.read_json("action.json")
    assert result == {"silent_seconds": 30.0, "spans": 1,
                      "candidates": len(action["candidates"])}
    assert action["spans"] == [[0.0, 30.0]] and action["threshold"] == 0.6
    assert 1 <= len(action["candidates"]) <= 5
    first = action["candidates"][0]
    assert first["sheet"] == "frames/action-00.jpg" and len(first["sheet_times"]) == 6
    assert run.path(first["sheet"]).exists()
    assert run.exists("motion.json")


@needs_ffmpeg
def test_stale_sheets_are_removed_and_motion_is_reused(tmp_path, monkeypatch):
    run = silent_run(tmp_path)
    stage.run_action(run)
    run.path("frames/action-14.jpg").write_bytes(b"old")
    monkeypatch.setattr(stage, "measure_motion",
                        lambda *a, **k: pytest.fail("motion.json should be reused"))
    stage.run_action(run)
    assert not run.path("frames/action-14.jpg").exists()


def test_a_video_that_talks_throughout_never_decodes_motion(tmp_path, monkeypatch):
    run = Run.create(tmp_path, "talk")
    words = [{"word": "w", "start": t / 2, "end": t / 2 + 0.4} for t in range(60)]
    run.write_json("source.json", {"path": "x.mp4", "duration": 30.0, "energy": [0.5] * 30})
    run.write_json("transcript.json", {"segments": [{"words": words}]})
    monkeypatch.setattr(stage, "preflight", lambda **k: pytest.fail("no tools needed"))
    assert stage.run_action(run) == {"silent_seconds": 0.0, "spans": 0, "candidates": 0}
    assert run.read_json("action.json")["candidates"] == []
    assert not run.exists("motion.json")
