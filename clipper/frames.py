from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from clipper.ingest import ffmpeg_tail

COLS, ROWS = 3, 2
FRAME_WIDTH = 320


def sheet_times(start: float, end: float, n: int = COLS * ROWS) -> list[float]:
    """`n` times across [start, end], each centred in its equal slice."""
    step = (end - start) / n
    return [round(start + (k + 0.5) * step, 2) for k in range(n)]


def _ffmpeg(args: list[str], cwd: Path, what: str) -> None:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Could not {what}:\n{ffmpeg_tail(result.stderr)}")


def contact_sheet(ffmpeg: Path, video: Path, start: float, end: float, out: Path) -> list[float]:
    """Six frames across [start, end], tiled 3x2 into one JPG. Returns their times."""
    times = sheet_times(start, end)
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    video = Path(video).resolve()
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        work = Path(tmp)
        names = []
        for k, t in enumerate(times):
            name = f"f{k}.jpg"
            what = f"pull the frame at {t:.1f}s from {video.name}"
            _ffmpeg([str(ffmpeg), "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
                     "-frames:v", "1", "-vf", f"scale={FRAME_WIDTH}:-2", "-q:v", "3",
                     "-y", name], work, what)
            if not (work / name).exists():
                raise RuntimeError(f"Could not {what}.")
            names.append(name)
        # Bare names in the list, run from the temp dir: a run path containing
        # an apostrophe would otherwise break the concat demuxer's quoting.
        (work / "list.txt").write_text("".join(f"file '{n}'\n" for n in names),
                                       encoding="utf-8")
        _ffmpeg([str(ffmpeg), "-v", "error", "-f", "concat", "-safe", "0", "-i", "list.txt",
                 "-vf", f"tile={COLS}x{ROWS}", "-frames:v", "1", "-q:v", "3", "-y", str(out)],
                work, f"tile the frames for {out.name}")
    return times
