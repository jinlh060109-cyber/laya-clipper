import shutil
import subprocess
from pathlib import Path
import pytest
from clipper.render import (
    build_command, display_size, slugify, staged_subtitles, validate_clip,
)

def test_slugify_makes_a_filesystem_safe_name():
    assert slugify("The thing nobody says about funding!") == "the-thing-nobody-says-about-funding"

def test_slugify_truncates_long_titles():
    assert len(slugify("word " * 60)) <= 60

def test_slugify_never_returns_empty():
    assert slugify("!!!") == "clip"

def test_staged_subtitles_sit_alone_under_a_space_free_name():
    """ffmpeg runs with the staging dir as cwd and sees only the bare name."""
    with staged_subtitles("content", ".ass") as path:
        assert " " not in path.name
        assert [p.name for p in path.parent.iterdir()] == [path.name]
        assert path.read_text() == "content"

def test_staged_subtitles_cleans_up():
    with staged_subtitles("x", ".ass") as path:
        parent = path.parent
    assert not parent.exists()

def test_validate_clip_clamps_out_to_duration():
    out = validate_clip({"in": 90.0, "out": 200.0}, duration=100.0)
    assert out["out"] == 100.0

def test_validate_clip_rejects_clips_under_ten_seconds():
    with pytest.raises(ValueError, match="10"):
        validate_clip({"in": 10.0, "out": 18.0}, duration=100.0)

def test_validate_clip_rejects_inverted_boundaries():
    with pytest.raises(ValueError, match="after"):
        validate_clip({"in": 50.0, "out": 40.0}, duration=100.0)

def test_validate_clip_rejects_in_point_past_the_end():
    with pytest.raises(ValueError, match="beyond"):
        validate_clip({"in": 500.0, "out": 520.0}, duration=100.0)

def test_build_command_re_encodes_rather_than_stream_copying():
    """Claude trimmed to the word; stream copy would snap back to keyframes."""
    cmd = build_command(Path("ffmpeg"), Path("src.mp4"),
                        {"in": 10.0, "out": 40.0}, Path("out.mp4"), None)
    assert "-c" not in cmd and "copy" not in cmd
    assert "libx264" in cmd

def test_build_command_seeks_accurately():
    cmd = build_command(Path("ffmpeg"), Path("src.mp4"),
                        {"in": 10.0, "out": 40.0}, Path("out.mp4"), None)
    assert "-ss" in cmd and "-to" in cmd
    assert "-accurate_seek" in cmd

def test_build_command_includes_the_filter_chain():
    cmd = build_command(Path("ffmpeg"), Path("src.mp4"), {"in": 0.0, "out": 20.0},
                        Path("out.mp4"), "scale=1080:1920")
    assert "-vf" in cmd
    assert cmd[cmd.index("-vf") + 1] == "scale=1080:1920"

@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_render_produces_a_playable_file(tmp_path):
    """Generate a source at test time; no media is committed to the repo."""
    source = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=20:size=640x480:rate=25",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
         "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source)],
        capture_output=True, check=True,
    )
    from clipper.render import render_clip
    out = render_clip(Path(shutil.which("ffmpeg")), source,
                      {"in": 2.0, "out": 14.0}, tmp_path / "clip.mp4", None)
    assert out.exists() and out.stat().st_size > 0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    )
    assert float(probe.stdout.strip()) == pytest.approx(12.0, abs=0.5)


def test_display_size_swaps_dimensions_for_quarter_turns():
    """ffmpeg autorotates before filters, so crop must use the displayed frame."""
    assert display_size({"width": 1920, "height": 1080, "rotation": -90}) == (1080, 1920)
    assert display_size({"width": 1920, "height": 1080, "rotation": 90}) == (1080, 1920)
    assert display_size({"width": 1920, "height": 1080, "rotation": 180}) == (1920, 1080)
    assert display_size({"width": 1920, "height": 1080}) == (1920, 1080)


def test_render_metadata_carries_laya_not_jev(tmp_path, monkeypatch):
    from clipper import render
    from clipper.run import Run

    run = Run.create(tmp_path, "ep")
    run.write_json("source.json", {"path": "C:/src.mp4", "duration": 100.0,
                                   "video": {"width": 1920, "height": 1080, "rotation": 0}})
    run.write_json("transcript.json", {"language": "en", "segments": []})
    run.write_json("plan.json", {"clips": [{"in": 10.0, "out": 40.0, "title": "T",
                                            "laya": {"clipworthy": 0.72}}]})
    monkeypatch.setattr(render, "preflight",
                        lambda **k: type("T", (), {"ffmpeg": Path("ffmpeg")})())
    monkeypatch.setattr(render, "render_clip", lambda *a, **k: a[3])
    written = render.run_render(run, captions="none")
    assert written[0]["laya"] == {"clipworthy": 0.72}
    assert "jev" not in written[0]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_burned_subtitles_survive_a_temp_dir_with_spaces(tmp_path, monkeypatch):
    """The failure this guards: ffmpeg truncating the subtitle path at a space."""
    import tempfile
    from clipper.captions import Cue, render_ass
    from clipper.filters import build_filter_chain
    from clipper.render import render_clip

    spaced = tmp_path / "has space"
    spaced.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(spaced))
    source = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=12:size=640x360:rate=25",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
         "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source)],
        capture_output=True, check=True,
    )
    words = [{"word": "hello", "start": 0.0, "end": 1.0, "score": 0.9, "speaker": "S"}]
    ass = render_ass([Cue(start=0.0, end=1.0, words=words, speaker="S")], 360)
    with staged_subtitles(ass, ".ass") as staged:
        assert " " in str(staged.parent)
        chain = build_filter_chain(640, 360, False, Path(staged.name))
        out = render_clip(Path(shutil.which("ffmpeg")), source, {"in": 0.0, "out": 11.0},
                          tmp_path / "clip.mp4", chain, cwd=staged.parent)
    assert out.exists() and out.stat().st_size > 0


def test_render_refuses_a_malformed_plan_before_touching_ffmpeg(tmp_path, monkeypatch):
    from clipper import render
    from clipper.run import Run

    run = Run.create(tmp_path, "ep")
    run.write_json("source.json", {"path": "C:/src.mp4", "duration": 100.0,
                                   "video": {"width": 1920, "height": 1080, "rotation": 0}})
    run.write_json("transcript.json", {"language": "en", "segments": []})
    run.write_json("plan.json", {"clips": [{"in": 10.0, "out": 40.0, "crop_x": "left"}]})
    monkeypatch.setattr(render, "preflight",
                        lambda **k: type("T", (), {"ffmpeg": Path("ffmpeg")})())
    called = []
    monkeypatch.setattr(render, "render_clip", lambda *a, **k: called.append(a))
    with pytest.raises(ValueError, match="crop_x"):
        render.run_render(run, captions="none")
    assert called == []


def test_a_failed_render_leaves_no_partial_file_and_quotes_ffmpeg_briefly(tmp_path, monkeypatch):
    import subprocess

    from clipper import render

    target = tmp_path / "01.mp4"

    def fake_run(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"partial")
        banner = "ffmpeg version 9 Copyright\n" + "  --enable-x\n" * 300
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=banner + "Invalid argument\n")

    monkeypatch.setattr(render.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as info:
        render.render_clip(Path("ffmpeg"), Path("src.mp4"), {"in": 0.0, "out": 12.0},
                           target, None)
    assert not target.exists()
    assert "Invalid argument" in str(info.value)
    assert "Copyright" not in str(info.value)


def test_a_clip_with_no_speech_is_rendered_without_burned_subtitles(tmp_path, monkeypatch):
    from clipper import render
    from clipper.run import Run

    run = Run.create(tmp_path, "game")
    run.write_json("source.json", {"path": "C:/src.mp4", "duration": 100.0,
                                   "video": {"width": 1920, "height": 1080, "rotation": 0}})
    run.write_json("transcript.json", {"language": "en", "segments": []})
    run.write_json("plan.json", {"clips": [{"in": 10.0, "out": 30.0, "title": "T"}]})
    monkeypatch.setattr(render, "preflight",
                        lambda **k: type("T", (), {"ffmpeg": Path("ffmpeg")})())
    chains = []
    monkeypatch.setattr(render, "render_clip",
                        lambda ffmpeg, source, clip, target, chain, cwd=None: chains.append(chain))
    render.run_render(run, captions="burn")
    assert "subtitles" not in (chains[0] or "")
