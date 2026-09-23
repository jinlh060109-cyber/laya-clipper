import json
import shutil
import subprocess
from pathlib import Path

import pytest

from clipper.ingest import extract_audio, ingest, parse_probe, per_second_rms, probe_source
from clipper.run import Run

FIXTURE = Path(__file__).parent / "fixtures" / "ffprobe_1080p.json"


def test_parse_probe_extracts_duration_and_streams():
    probe = json.loads(FIXTURE.read_text())
    out = parse_probe(probe, Path("/videos/ep47.mp4"))
    assert out["duration"] == pytest.approx(5400.12)
    assert out["video"]["width"] == 1920
    assert out["video"]["height"] == 1080
    assert out["audio"]["sample_rate"] == 48000
    assert out["path"] == "/videos/ep47.mp4"


def test_parse_probe_converts_fractional_frame_rate():
    probe = json.loads(FIXTURE.read_text())
    assert parse_probe(probe, Path("x.mp4"))["video"]["fps"] == pytest.approx(29.97, abs=0.01)


def test_parse_probe_reads_rotation_side_data():
    """Portrait phone footage must not be cropped sideways at render time."""
    probe = json.loads(FIXTURE.read_text())
    assert parse_probe(probe, Path("x.mp4"))["video"]["rotation"] == -90


def test_parse_probe_defaults_rotation_to_zero():
    probe = {"format": {"duration": "10.0", "format_name": "mp4"},
             "streams": [{"codec_type": "video", "codec_name": "h264",
                          "width": 640, "height": 480, "avg_frame_rate": "25/1"},
                         {"codec_type": "audio", "codec_name": "aac",
                          "sample_rate": "44100", "channels": 1}]}
    assert parse_probe(probe, Path("x.mp4"))["video"]["rotation"] == 0


def test_parse_probe_reads_rotation_from_tags_rotate_when_no_side_data():
    """Many phone-shot MP4/MOV files carry only tags.rotate, no display matrix.

    ffmpeg's display-matrix convention (which source.json stores) is the
    NEGATION of tags.rotate: side_data rotation -90 <-> tags.rotate "90".
    """
    probe = {"format": {"duration": "10.0", "format_name": "mp4"},
             "streams": [{"codec_type": "video", "codec_name": "h264",
                          "width": 640, "height": 480, "avg_frame_rate": "25/1",
                          "tags": {"rotate": "90"}},
                         {"codec_type": "audio", "codec_name": "aac",
                          "sample_rate": "44100", "channels": 1}]}
    assert parse_probe(probe, Path("x.mp4"))["video"]["rotation"] == -90


def test_parse_probe_side_data_rotation_wins_over_tags_rotate():
    probe = {"format": {"duration": "10.0", "format_name": "mp4"},
             "streams": [{"codec_type": "video", "codec_name": "h264",
                          "width": 640, "height": 480, "avg_frame_rate": "25/1",
                          "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}],
                          "tags": {"rotate": "180"}},
                         {"codec_type": "audio", "codec_name": "aac",
                          "sample_rate": "44100", "channels": 1}]}
    assert parse_probe(probe, Path("x.mp4"))["video"]["rotation"] == -90


def test_parse_probe_handles_non_numeric_tags_rotate_without_crashing():
    probe = {"format": {"duration": "10.0", "format_name": "mp4"},
             "streams": [{"codec_type": "video", "codec_name": "h264",
                          "width": 640, "height": 480, "avg_frame_rate": "25/1",
                          "tags": {"rotate": "not-a-number"}},
                         {"codec_type": "audio", "codec_name": "aac",
                          "sample_rate": "44100", "channels": 1}]}
    assert parse_probe(probe, Path("x.mp4"))["video"]["rotation"] == 0


def test_parse_probe_rejects_file_with_no_audio():
    probe = {"format": {"duration": "10.0", "format_name": "mp4"},
             "streams": [{"codec_type": "video", "codec_name": "h264",
                          "width": 640, "height": 480, "avg_frame_rate": "25/1"}]}
    with pytest.raises(ValueError, match="no audio"):
        parse_probe(probe, Path("x.mp4"))


# --- extract_audio: atomic write via temp file + os.replace (Decision 2) ---
#
# ffmpeg is not installed on this machine, so these fake the subprocess layer.
# The fake stands in for ffmpeg: it inspects the argv it was given, writes
# bytes to whatever path ffmpeg would have written to (the last argument),
# and returns a canned CompletedProcess.

def _fake_run_that_writes(returncode: int, payload: bytes):
    def fake_run(cmd, **kwargs):
        out_path = Path(cmd[-1])
        out_path.write_bytes(payload)
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr="boom" if returncode else "")
    return fake_run


def test_extract_audio_success_leaves_audio_in_place_with_no_leftover_temp(tmp_path, monkeypatch):
    dest = tmp_path / "audio.wav"
    monkeypatch.setattr(
        "clipper.ingest.subprocess.run", _fake_run_that_writes(0, b"RIFF-fake-wav-bytes")
    )

    extract_audio(Path("ffmpeg"), Path("video.mp4"), dest)

    assert dest.exists()
    assert dest.read_bytes() == b"RIFF-fake-wav-bytes"
    assert not (tmp_path / "audio.wav.tmp").exists()


def test_extract_audio_failure_raises_and_leaves_no_audio_or_temp(tmp_path, monkeypatch):
    dest = tmp_path / "audio.wav"
    # A crashing ffmpeg may still have written a truncated file before it died.
    monkeypatch.setattr(
        "clipper.ingest.subprocess.run", _fake_run_that_writes(1, b"partial-garbage")
    )

    with pytest.raises(RuntimeError, match="Audio extraction failed"):
        extract_audio(Path("ffmpeg"), Path("video.mp4"), dest)

    assert not dest.exists()
    assert not (tmp_path / "audio.wav.tmp").exists()


def test_extract_audio_failure_does_not_destroy_preexisting_audio(tmp_path, monkeypatch):
    dest = tmp_path / "audio.wav"
    dest.write_bytes(b"original-good-audio")
    monkeypatch.setattr(
        "clipper.ingest.subprocess.run", _fake_run_that_writes(1, b"partial-garbage")
    )

    with pytest.raises(RuntimeError, match="Audio extraction failed"):
        extract_audio(Path("ffmpeg"), Path("video.mp4"), dest)

    assert dest.read_bytes() == b"original-good-audio"
    assert not (tmp_path / "audio.wav.tmp").exists()


# --- per_second_rms: dB->linear conversion, floors, returncode, padding/truncation ---

def _fake_run_returning(returncode: int, stdout: str = "", stderr: str = ""):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)
    return fake_run


def _rms_stderr(*db_values: str) -> str:
    lines = ["frame:0    pts:0   pts_time:0"]
    lines += [f"lavfi.astats.Overall.RMS_level={v}" for v in db_values]
    return "\n".join(lines)


def test_per_second_rms_converts_db_to_linear_amplitude(monkeypatch):
    monkeypatch.setattr(
        "clipper.ingest.subprocess.run",
        _fake_run_returning(0, stderr=_rms_stderr("-20.000000", "-10.000000", "-30.000000")),
    )
    levels = per_second_rms(Path("ffmpeg"), Path("audio.wav"), duration=3.0)
    assert levels == pytest.approx([10 ** (-20 / 20), 10 ** (-10 / 20), 10 ** (-30 / 20)])


def test_per_second_rms_treats_negative_inf_as_floor(monkeypatch):
    monkeypatch.setattr(
        "clipper.ingest.subprocess.run",
        _fake_run_returning(0, stderr=_rms_stderr("-inf")),
    )
    levels = per_second_rms(Path("ffmpeg"), Path("audio.wav"), duration=1.0)
    assert levels == pytest.approx([10 ** (-90 / 20)])


def test_per_second_rms_treats_nan_as_floor(monkeypatch):
    monkeypatch.setattr(
        "clipper.ingest.subprocess.run",
        _fake_run_returning(0, stderr=_rms_stderr("nan")),
    )
    levels = per_second_rms(Path("ffmpeg"), Path("audio.wav"), duration=1.0)
    assert levels == pytest.approx([10 ** (-90 / 20)])


def test_per_second_rms_clamps_values_below_the_floor(monkeypatch):
    # A real (non-nan/-inf) numeric reading quieter than -90dB still clamps.
    monkeypatch.setattr(
        "clipper.ingest.subprocess.run",
        _fake_run_returning(0, stderr=_rms_stderr("-120.000000")),
    )
    levels = per_second_rms(Path("ffmpeg"), Path("audio.wav"), duration=1.0)
    assert levels == pytest.approx([10 ** (-90 / 20)])


def test_per_second_rms_pads_when_ffmpeg_emits_fewer_lines_than_duration(monkeypatch):
    monkeypatch.setattr(
        "clipper.ingest.subprocess.run",
        _fake_run_returning(0, stderr=_rms_stderr("-20.000000", "-10.000000")),
    )
    levels = per_second_rms(Path("ffmpeg"), Path("audio.wav"), duration=5.0)
    expected_tail = 10 ** (-10 / 20)
    assert len(levels) == 5
    assert levels[:2] == pytest.approx([10 ** (-20 / 20), 10 ** (-10 / 20)])
    assert levels[2:] == pytest.approx([expected_tail] * 3)


def test_per_second_rms_truncates_when_ffmpeg_emits_more_lines_than_duration(monkeypatch):
    monkeypatch.setattr(
        "clipper.ingest.subprocess.run",
        _fake_run_returning(
            0, stderr=_rms_stderr("-20.000000", "-10.000000", "-30.000000", "-5.000000")
        ),
    )
    levels = per_second_rms(Path("ffmpeg"), Path("audio.wav"), duration=2.0)
    assert levels == pytest.approx([10 ** (-20 / 20), 10 ** (-10 / 20)])


def test_per_second_rms_raises_on_nonzero_returncode(monkeypatch):
    monkeypatch.setattr(
        "clipper.ingest.subprocess.run",
        _fake_run_returning(1, stderr="astats: filter init failed"),
    )
    with pytest.raises(RuntimeError, match="RMS extraction failed"):
        per_second_rms(Path("ffmpeg"), Path("audio.wav"), duration=5.0)


def test_per_second_rms_raises_when_no_rms_lines_found(monkeypatch):
    # returncode 0 but nothing recognizable in the output: must not silently
    # fall through to an all-zero energy array.
    monkeypatch.setattr(
        "clipper.ingest.subprocess.run",
        _fake_run_returning(0, stderr="frame:0    pts:0   pts_time:0\n"),
    )
    with pytest.raises(RuntimeError, match="No RMS levels parsed"):
        per_second_rms(Path("ffmpeg"), Path("audio.wav"), duration=5.0)


# --- probe_source: JSON parsing over a faked ffprobe subprocess ---

def test_probe_source_parses_faked_ffprobe_output(monkeypatch):
    monkeypatch.setattr(
        "clipper.ingest.subprocess.run",
        _fake_run_returning(0, stdout=FIXTURE.read_text()),
    )
    out = probe_source(Path("ffprobe"), Path("/videos/ep47.mp4"))
    assert out["duration"] == pytest.approx(5400.12)
    assert out["video"]["width"] == 1920
    assert out["audio"]["sample_rate"] == 48000
    assert out["path"] == Path("/videos/ep47.mp4").as_posix()


def test_probe_source_raises_on_nonzero_returncode(monkeypatch):
    monkeypatch.setattr(
        "clipper.ingest.subprocess.run",
        _fake_run_returning(1, stderr="ep47.mp4: No such file or directory"),
    )
    with pytest.raises(RuntimeError, match="ffprobe failed"):
        probe_source(Path("ffprobe"), Path("/videos/ep47.mp4"))


# --- ingest: full orchestration, writing source.json THROUGH Run.write_json ---

_INGEST_PROBE = {
    "format": {"duration": "3.0", "format_name": "mp4"},
    "streams": [
        {"codec_type": "video", "codec_name": "h264", "width": 640, "height": 480,
         "avg_frame_rate": "25/1",
         "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}]},
        {"codec_type": "audio", "codec_name": "aac", "sample_rate": "44100", "channels": 1},
    ],
}


def test_ingest_writes_source_json_through_run_write_json(tmp_path, monkeypatch):
    run = Run.create(tmp_path, "ep47")

    def fake_run(cmd, **kwargs):
        if "-show_streams" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(_INGEST_PROBE), stderr="")
        if "-af" in cmd:
            stderr = _rms_stderr("-20.000000", "-10.000000", "-30.000000")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=stderr)
        # extract_audio call: ffmpeg writes its output to the last argument.
        Path(cmd[-1]).write_bytes(b"fake-wav-bytes")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("clipper.ingest.subprocess.run", fake_run)

    result = ingest(Path("ffmpeg"), Path("ffprobe"), Path("/videos/ep47.mp4"), run)

    assert result["path"] == "/videos/ep47.mp4"
    assert result["duration"] == pytest.approx(3.0)
    assert result["container"] == "mp4"
    assert result["video"] == {
        "codec": "h264", "width": 640, "height": 480, "fps": pytest.approx(25.0), "rotation": -90,
    }
    assert result["audio"] == {"codec": "aac", "sample_rate": 44100, "channels": 1}
    assert len(result["energy"]) == 3
    assert all(isinstance(v, float) for v in result["energy"])

    # Persisted through Run.write_json (atomic write), and readable back as
    # the identical artifact - not just an in-memory dict.
    on_disk = run.read_json("source.json")
    assert on_disk == result
    assert not run.path("source.json.tmp").exists()
    assert run.path("audio.wav").read_bytes() == b"fake-wav-bytes"


def test_extract_audio_with_real_ffmpeg_writes_a_wav(tmp_path):
    """The temp name ends in .tmp, so ffmpeg must be told the format."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH")
    video = tmp_path / "tone.mp4"
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=size=64x64:duration=1",
                    "-f", "lavfi", "-i", "sine=duration=1", "-shortest",
                    "-c:v", "libx264", "-c:a", "aac", str(video)], check=True)
    dest = tmp_path / "audio.wav"
    extract_audio(Path(ffmpeg), video, dest)
    assert dest.read_bytes()[:4] == b"RIFF"
    assert not dest.with_suffix(".wav.tmp").exists()


def test_extract_audio_failure_reports_ffmpeg_tail_not_its_banner(tmp_path, monkeypatch):
    banner = "ffmpeg version 9 Copyright\n" + "  --enable-thing\n" * 200
    monkeypatch.setattr("clipper.ingest.subprocess.run",
                        _fake_run_returning(1, stderr=banner + "video.mp4: Invalid data found\n"))
    with pytest.raises(RuntimeError) as info:
        extract_audio(Path("ffmpeg"), Path("video.mp4"), tmp_path / "audio.wav")
    message = str(info.value)
    assert "Invalid data found" in message
    assert "Copyright" not in message
    assert len(message) < 1000


def test_ffmpeg_output_with_non_ascii_metadata_is_decoded_as_utf8(tmp_path):
    """ffmpeg writes UTF-8; a non-UTF-8 locale (cp936 here) must not break parsing."""
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("ffmpeg not on PATH")
    video = tmp_path / "titled.mp4"
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=size=64x64:duration=2",
                    "-f", "lavfi", "-i", "sine=duration=2", "-shortest", "-c:v", "libx264",
                    "-c:a", "aac", "-metadata", "title=naïve — “quoted” ü 🎮", str(video)],
                   check=True)
    assert probe_source(Path(ffprobe), video)["duration"] > 1
    assert len(per_second_rms(Path(ffmpeg), video, 2.0)) == 2
