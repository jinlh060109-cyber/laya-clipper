import json
import shutil
import subprocess
from pathlib import Path

import pytest

from clipper import edit
from clipper.run import Run
from clipper.style import DEFAULT

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def test_slugify_makes_a_filesystem_safe_name():
    assert edit.slugify("The thing nobody says about funding!") == "the-thing-nobody-says-about-funding"
    assert len(edit.slugify("word " * 60)) <= 60
    assert edit.slugify("!!!") == "clip"


def test_staged_subtitles_sit_alone_and_are_cleaned_up():
    with edit.staged_subtitles("content", ".ass") as path:
        assert " " not in path.name
        assert [p.name for p in path.parent.iterdir()] == [path.name]
        parent = path.parent
    assert not parent.exists()


def test_display_size_swaps_dimensions_for_quarter_turns():
    assert edit.display_size({"width": 1920, "height": 1080, "rotation": -90}) == (1080, 1920)
    assert edit.display_size({"width": 1920, "height": 1080}) == (1920, 1080)


def test_cut_command_seeks_accurately_and_uses_the_chosen_encoder():
    cmd = edit.cut_command("ffmpeg", Path("src.mp4"), 10.0, 40.0, Path("cut.mp4"), "h264_qsv")
    assert cmd[cmd.index("-ss") + 1] == "10.000" and cmd[cmd.index("-to") + 1] == "40.000"
    assert cmd[cmd.index("-c:v") + 1] == "h264_qsv" and "-global_quality" in cmd
    assert "-vf" not in cmd


def test_final_command_applies_the_filter_chain():
    cmd = edit.final_command("ffmpeg", Path("s.mp4"), 0.0, 20.0, Path("f.mp4"), "scale=1:1",
                             "libx264")
    assert cmd[cmd.index("-vf") + 1] == "scale=1:1" and "-crf" in cmd


def _run(tmp_path, source, clips, words=None):
    run = Run.create(tmp_path, "ep")
    run.write_json("source.json", {"path": str(source), "duration": 30.0,
                                   "video": {"width": 640, "height": 360, "rotation": 0}})
    run.write_json("transcript.json", {"language": "en", "segments": [
        {"start": 1.0, "end": 12.0, "speaker": "S", "text": "t", "words": words or []}]})
    run.write_json("segments.json", {"content_type": "tutorial"})
    run.write_json("selection.json", {"rule": {}, "clips": clips})
    return run


def _clip(cid, start, end, include=True, category="tip"):
    return {"id": cid, "start": start, "end": end, "duration": end - start,
            "category": category, "score": 0.6, "confidence": 0.4, "uncertain": False,
            "title_hint": f"Clip {cid}", "reason": "", "source": "ai",
            "include": include, "fill_in": False}


def test_a_clip_under_ten_seconds_is_refused_before_ffmpeg_runs(tmp_path, monkeypatch):
    run = _run(tmp_path, tmp_path / "src.mp4", [_clip("c0", 1.0, 6.0)])
    monkeypatch.setattr(edit.subprocess, "run", lambda *a, **k: pytest.fail("ffmpeg ran"))
    with pytest.raises(ValueError, match="under the 10 s minimum"):
        edit.run_edit(run, DEFAULT, ffmpeg="ffmpeg", encoders=[])


def test_nothing_ticked_is_an_error(tmp_path):
    run = _run(tmp_path, tmp_path / "src.mp4", [_clip("c0", 1.0, 20.0, include=False)])
    with pytest.raises(ValueError, match="No clips are ticked"):
        edit.run_edit(run, DEFAULT, ffmpeg="ffmpeg", encoders=[])


def test_a_failed_render_leaves_no_partial_file(tmp_path, monkeypatch):
    run = _run(tmp_path, tmp_path / "src.mp4", [_clip("c0", 1.0, 20.0)])

    def fake_run(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"partial")
        return subprocess.CompletedProcess(cmd, 1, "", "line\n" * 50 + "Invalid data found")

    monkeypatch.setattr(edit.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="Invalid data found"):
        edit.run_edit(run, {**DEFAULT, "captions": "none"}, ffmpeg="ffmpeg", encoders=[])
    assert not list(run.clips_dir().rglob("*.mp4"))


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_each_ticked_clip_becomes_a_folder_with_cut_final_and_prompt(tmp_path):
    source = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=30:size=640x360:rate=25",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=30",
                    "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source)],
                   capture_output=True, check=True)
    words = [{"word": w, "start": 1.0 + i, "end": 1.8 + i, "score": 0.9, "speaker": "S"}
             for i, w in enumerate("Here is the trick. It works every time.".split())]
    run = _run(tmp_path, source, [_clip("c0", 1.0, 13.0), _clip("a0", 14.0, 26.0, category="action"),
                                  _clip("c1", 2.0, 20.0, include=False)], words)
    fills = {"c0": {"title": "The Trick", "hook": "h", "description": "d",
                    "caption_quote": "Here is the trick.", "punch_ins": []}}
    done = edit.run_edit(run, {**DEFAULT, "notes": "Calm."}, fills=fills,
                         ffmpeg=shutil.which("ffmpeg"), encoders=[])

    assert [d["folder"] for d in done] == ["01-the-trick", "02-clip-a0"]
    folder = run.clips_dir() / "01-the-trick"
    assert {p.name for p in folder.iterdir()} >= {"cut.mp4", "final.mp4", "edit.json",
                                                  "prompt.md", "captions.srt"}
    spec = json.loads((folder / "edit.json").read_text(encoding="utf-8"))
    assert spec["title"] == "The Trick" and spec["layout"] == "fit"
    assert "Calm." in (folder / "prompt.md").read_text(encoding="utf-8")
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", str(folder / "final.mp4")],
                           capture_output=True, text=True, check=True)
    assert probe.stdout.split()[0] == "1080,1920"
    action = run.clips_dir() / "02-clip-a0"
    assert not (action / "captions.srt").exists()  # nobody speaks: no captions
