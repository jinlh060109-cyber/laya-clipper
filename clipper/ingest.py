from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from clipper.energy import rolling_baseline
from clipper.run import Run


def ffmpeg_tail(stderr: str, lines: int = 8) -> str:
    """The last lines of ffmpeg's stderr, where the actual error is.

    ffmpeg opens with a long version and build banner; quoting all of it
    buries the one line that says what went wrong.
    """
    kept = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    return "\n".join(kept[-lines:])[-800:]


def _fps(rate: str) -> float:
    if "/" in rate:
        num, den = rate.split("/", 1)
        return float(num) / float(den) if float(den) else 0.0
    return float(rate)


def _rotation(stream: dict) -> int:
    for side in stream.get("side_data_list", []):
        if "rotation" in side:
            return int(side["rotation"])
    # Fallback for phone-shot files that carry no display matrix, only a
    # `tags.rotate` string (e.g. many MP4/MOV files from mobile cameras).
    #
    # Sign convention: ffmpeg's display-matrix `rotation` (the side-data
    # convention above, and the convention source.json stores) is the
    # NEGATION of `tags.rotate`. Portrait footage needing 90 degrees
    # clockwise for display reports side_data rotation: -90 alongside
    # tags: {"rotate": "90"}. Negate here so the fallback matches the
    # side-data convention - do not "fix" this back to +raw.
    raw = stream.get("tags", {}).get("rotate")
    if raw is not None:
        try:
            return -int(raw)
        except (TypeError, ValueError):
            return 0
    return 0


def parse_probe(probe: dict, source: Path) -> dict:
    streams = probe.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise ValueError(f"{source} has no video stream.")
    if audio is None:
        raise ValueError(f"{source} has no audio stream; there is nothing to transcribe.")
    return {
        "path": source.as_posix(),
        "duration": float(probe["format"]["duration"]),
        "container": probe["format"]["format_name"],
        "video": {
            "codec": video["codec_name"],
            "width": int(video["width"]),
            "height": int(video["height"]),
            "fps": _fps(video.get("avg_frame_rate", "0/1")),
            "rotation": _rotation(video),
        },
        "audio": {
            "codec": audio["codec_name"],
            "sample_rate": int(audio["sample_rate"]),
            "channels": int(audio["channels"]),
        },
    }


def probe_source(ffprobe: Path, video: Path) -> dict:
    result = subprocess.run(
        [str(ffprobe), "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(video)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video}:\n{ffmpeg_tail(result.stderr)}")
    return parse_probe(json.loads(result.stdout), video)


def extract_audio(ffmpeg: Path, video: Path, dest: Path) -> None:
    """Extract mono 16kHz PCM audio to `dest`, atomically.

    ffmpeg writes to a temporary path alongside `dest` and only `os.replace`s
    it into place after a successful exit. This mirrors the staging approach
    `Run.write_json` uses for JSON artifacts, so a crash or interruption mid
    write can never leave a truncated `audio.wav` for a later stage to pick
    up as if it were valid.
    """
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        result = subprocess.run(
            [str(ffmpeg), "-hide_banner", "-y", "-i", str(video), "-vn",
             "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
             # The temp name ends in .tmp, which ffmpeg cannot map to a
             # muxer, so the container has to be named explicitly.
             "-f", "wav", str(tmp)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Audio extraction failed:\n{ffmpeg_tail(result.stderr)}")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def per_second_rms(ffmpeg: Path, wav: Path, duration: float) -> list[float]:
    """One RMS value per second, via ffmpeg's astats filter."""
    result = subprocess.run(
        [str(ffmpeg), "-v", "info", "-i", str(wav),
         "-af", "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level",
         "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"RMS extraction failed on {wav}:\n{ffmpeg_tail(result.stderr)}")
    levels: list[float] = []
    for line in (result.stderr + result.stdout).splitlines():
        if "RMS_level=" in line:
            raw = line.rsplit("=", 1)[1].strip()
            db = -90.0 if raw in {"-inf", "-nan", "nan"} else float(raw)
            levels.append(10 ** (max(db, -90.0) / 20.0))
    if not levels:
        raise RuntimeError(
            f"No RMS levels parsed from ffmpeg astats output for {wav}; "
            f"the astats filter may not have run."
        )
    expected = max(1, int(duration))
    if len(levels) < expected:
        levels.extend([levels[-1]] * (expected - len(levels)))
    return levels[:expected]


def ingest(ffmpeg: Path, ffprobe: Path, video: Path, run: Run) -> dict:
    source = probe_source(ffprobe, video)
    wav = run.path("audio.wav")
    extract_audio(ffmpeg, video, wav)
    raw = per_second_rms(ffmpeg, wav, source["duration"])
    source["energy"] = rolling_baseline(raw)
    run.write_json("source.json", source)
    return source
