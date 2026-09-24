import json
import shutil
import subprocess
from pathlib import Path

import pytest

from clipper.ingest import extract_audio, ingest, parse_probe, per_second_rms, probe_source
from clipper.run import Run

FIXTURE = Path(__file__).parent / "fixtures" / "ffprobe_1080p.json"


def test_parse_probe_reads_rotation_side_data():
    """Portrait phone footage must not be cropped sideways at render time."""
    probe = json.loads(FIXTURE.read_text())
    assert parse_probe(probe, Path("x.mp4"))["video"]["rotation"] == -90


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


@pytest.mark.parametrize("db", ["-inf", "nan", "-120.000000"])
def test_per_second_rms_floors_silence_and_unreadable_levels(monkeypatch, db):
    monkeypatch.setattr(
        "clipper.ingest.subprocess.run",
        _fake_run_returning(0, stderr=_rms_stderr(db)),
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


# --- probe_source ---

def test_probe_source_raises_on_nonzero_returncode(monkeypatch):
    monkeypatch.setattr(
        "clipper.ingest.subprocess.run",
        _fake_run_returning(1, stderr="ep47.mp4: No such file or directory"),
    )
    with pytest.raises(RuntimeError, match="ffprobe failed"):
        probe_source(Path("ffprobe"), Path("/videos/ep47.mp4"))


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


def test_per_second_rms_with_real_ffmpeg_measures_whole_seconds(tmp_path):
    """astats reset=1 resets per audio frame (~64 ms); each value must cover one second."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH")
    wav = tmp_path / "burst.wav"
    # 3 s near-silence, 1 s loud tone, 2 s near-silence, at 16 kHz mono.
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:sample_rate=16000:duration=6",
                    "-af", "volume='if(between(t,3,4),1,0.001)':eval=frame",
                    "-ac", "1", str(wav)], check=True)
    levels = per_second_rms(Path(ffmpeg), wav, 6.0)
    assert len(levels) == 6
    assert levels.index(max(levels)) == 3
    assert levels[3] > 3 * max(levels[:3] + levels[4:])  # tone bleeds ~1 frame into s4


# --- ingest ---

_INGEST_PROBE = {
    "format": {"duration": "3.0", "format_name": "mp4"},
    "streams": [
        {"codec_type": "video", "codec_name": "h264", "width": 640, "height": 480,
         "avg_frame_rate": "25/1",
         "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}]},
        {"codec_type": "audio", "codec_name": "aac", "sample_rate": "44100", "channels": 1},
    ],
}


def _fake_ingest_run(cmd, **kwargs):
    if "-show_streams" in cmd:
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(_INGEST_PROBE), stderr="")
    if "-af" in cmd:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=_rms_stderr("-20.0"))
    Path(cmd[-1]).write_bytes(b"fake-wav-bytes")
    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")


def test_ingest_stores_an_absolute_path_and_the_file_size(tmp_path, monkeypatch):
    """render resolves source.json's path later, possibly from another directory."""
    video = tmp_path / "ep47.mp4"
    video.write_bytes(b"x" * 1234)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("clipper.ingest.subprocess.run", _fake_ingest_run)
    out = ingest(Path("ffmpeg"), Path("ffprobe"), Path("ep47.mp4"), Run.create(tmp_path, "r"))
    assert Path(out["path"]).is_absolute()
    assert Path(out["path"]) == video.resolve()
    assert out["size"] == 1234


def test_a_run_holding_another_video_is_refused(tmp_path):
    from clipper.ingest import check_same_source

    run = Run.create(tmp_path, "r")
    first, second = tmp_path / "a.mp4", tmp_path / "b.mp4"
    first.write_bytes(b"a")
    second.write_bytes(b"bb")
    run.write_json("source.json", {"path": first.resolve().as_posix(), "size": 1})
    check_same_source(run, first)                      # same video: fine
    with pytest.raises(ValueError, match="--run"):
        check_same_source(run, second)
    first.write_bytes(b"changed")                      # same path, different file
    with pytest.raises(ValueError, match="--run"):
        check_same_source(run, first)
