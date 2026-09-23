from __future__ import annotations

import re
import subprocess
from collections import deque
from pathlib import Path

from clipper.energy import rolling_baseline
from clipper.ingest import ffmpeg_tail

CUT_SCORE = 10.0
# Two tiny grey frames a second are plenty to tell a fight from a menu, and
# ffmpeg's scdet does the pixel work, so no image library is needed.
FILTER = "fps=2,scale=64:-2,scdet=threshold=10,metadata=print"
_PTS = re.compile(r"pts_time:(\S+)")


def _value(line: str, key: str) -> float | None:
    if key not in line:
        return None
    try:
        return float(line.rsplit("=", 1)[1])
    except ValueError:
        return None


def parse_scdet(lines, seconds: int, progress=None) -> tuple[list[float], list[int]]:
    """Mean frame difference and scene-cut count per whole second."""
    sums, counts, cuts = [0.0] * seconds, [0] * seconds, [0] * seconds
    second: int | None = None
    for line in lines:
        match = _PTS.search(line)
        if match:
            try:
                t = float(match.group(1))
            except ValueError:
                second = None
                continue
            now = int(t) if 0 <= t < seconds else None
            if progress and now is not None and now != second:
                progress(now + 1, seconds)
            second = now
            continue
        if second is None:
            continue
        mafd = _value(line, "lavfi.scd.mafd=")
        if mafd is not None:
            sums[second] += mafd
            counts[second] += 1
            continue
        score = _value(line, "lavfi.scd.score=")
        if score is not None and score >= CUT_SCORE:
            cuts[second] += 1
    raw: list[float] = []
    for total, n in zip(sums, counts):
        # A second with no sampled frame (variable frame rate, or the tail)
        # repeats the last known value rather than reading as a freeze.
        raw.append(round(total / n, 4) if n else (raw[-1] if raw else 0.0))
    return raw, cuts


def measure_motion(ffmpeg: Path, video: Path, seconds: int, progress=None) -> dict:
    """One decode of the video: per-second picture change, normalized, and cuts."""
    proc = subprocess.Popen(
        [str(ffmpeg), "-v", "info", "-i", str(video), "-vf", FILTER, "-an", "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace")
    tail: deque[str] = deque(maxlen=40)

    def lines():
        for line in proc.stderr:
            tail.append(line)
            yield line

    raw, cuts = parse_scdet(lines(), seconds, progress)
    if proc.wait() != 0:
        raise RuntimeError(
            f"Motion measurement failed on {video}:\n{ffmpeg_tail(''.join(tail))}")
    if progress:
        progress(seconds, seconds)
    return {"motion_raw": raw, "motion": rolling_baseline(raw), "cuts": cuts}
