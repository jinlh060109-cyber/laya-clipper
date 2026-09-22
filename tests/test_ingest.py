import json
import subprocess
from pathlib import Path

import pytest

from clipper.ingest import extract_audio, parse_probe

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
