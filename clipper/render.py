from __future__ import annotations

import contextlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from clipper.captions import (
    build_cues, rebase, render_ass, render_srt, words_in_range,
)
from clipper.filters import build_filter_chain
from clipper.preflight import preflight
from clipper.run import Run

MIN_CLIP_SECONDS = 10.0


def slugify(text: str, limit: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:limit].rstrip("-") or "clip")


@contextlib.contextmanager
def staged_subtitles(content: str, suffix: str):
    """Write subtitles alone into a fresh temp directory under a bare name.

    ffmpeg's subtitles filter truncates paths at the first space. Quoting,
    single quotes and backslash escaping have all been tried and none work.
    So ffmpeg runs with this directory as its cwd and is given only the bare
    file name; the temp root may contain spaces (it does whenever Windows 8.3
    short names are disabled and the username has one).
    """
    directory = Path(tempfile.mkdtemp(prefix="clipper_"))
    try:
        path = directory / f"sub{suffix}"
        path.write_text(content, encoding="utf-8")
        yield path
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def display_size(video: dict) -> tuple[int, int]:
    """(width, height) as displayed. ffmpeg autorotates before any filter runs,
    so crop arithmetic on stored dimensions would be wrong for phone footage."""
    width, height = video["width"], video["height"]
    if int(video.get("rotation") or 0) % 180 == 90:
        return height, width
    return width, height


def validate_clip(clip: dict, duration: float) -> dict:
    start, end = float(clip["in"]), float(clip["out"])
    if start >= duration:
        raise ValueError(f"Clip in-point {start:.2f}s is beyond the source end ({duration:.2f}s).")
    if end <= start:
        raise ValueError(f"Clip out-point {end:.2f}s is not after its in-point {start:.2f}s.")
    end = min(end, duration)
    if end - start < MIN_CLIP_SECONDS:
        raise ValueError(
            f"Clip is {end - start:.2f}s, under the {MIN_CLIP_SECONDS:.0f}s minimum. "
            "Shorter clips read as incomplete."
        )
    return {**clip, "in": start, "out": end}


def build_command(ffmpeg: Path, source: Path, clip: dict,
                  output: Path, filter_chain: str | None) -> list[str]:
    cmd = [str(ffmpeg), "-y", "-accurate_seek",
           "-ss", f"{clip['in']:.3f}", "-to", f"{clip['out']:.3f}",
           "-i", str(source)]
    if filter_chain:
        cmd += ["-vf", filter_chain]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart", str(output)]
    return cmd


def render_clip(ffmpeg: Path, source: Path, clip: dict,
                output: Path, filter_chain: str | None,
                cwd: Path | None = None) -> Path:
    """`cwd` is the staged-subtitles directory when the chain names a bare file."""
    result = subprocess.run(build_command(ffmpeg, source, clip, output, filter_chain),
                            capture_output=True, text=True, check=False, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Render failed for {output.name}:\n{result.stderr[-2000:]}")
    return output


def run_render(run: Run, vertical: bool = False, captions: str = "burn") -> list[dict]:
    plan = run.read_json("plan.json")
    source_meta = run.read_json("source.json")
    transcript = run.read_json("transcript.json")
    tools = preflight(require_subtitles=captions == "burn")

    source = Path(source_meta["path"]).resolve()
    width, height = display_size(source_meta["video"])
    speakers = plan.get("speakers") or {}
    out_dir = run.clips_dir().resolve()  # absolute: ffmpeg may run in another cwd
    written: list[dict] = []

    for index, raw in enumerate(plan["clips"], start=1):
        clip = validate_clip(raw, source_meta["duration"])
        mode = clip.get("captions", captions)
        want_vertical = clip.get("vertical", vertical)
        stem = f"{index:02d}-{slugify(clip.get('title', ''))}"

        cues = rebase(build_cues(words_in_range(transcript, clip["in"], clip["out"])),
                      clip["in"])
        (out_dir / f"{stem}.srt").write_text(render_srt(cues), encoding="utf-8")

        out_height = 1920 if want_vertical else height
        labelled = speakers if clip.get("speaker_label") else None
        ass = render_ass(cues, out_height, labelled) if mode == "burn" else None

        target = out_dir / f"{stem}.mp4"
        if ass is not None:
            with staged_subtitles(ass, ".ass") as staged:
                chain = build_filter_chain(width, height, want_vertical,
                                           Path(staged.name),
                                           clip.get("crop_x", "center"))
                render_clip(tools.ffmpeg, source, clip, target, chain,
                            cwd=staged.parent)
        else:
            chain = build_filter_chain(width, height, want_vertical, None,
                                       clip.get("crop_x", "center"))
            render_clip(tools.ffmpeg, source, clip, target, chain)

        meta = {k: v for k, v in clip.items() if k != "laya"}
        meta["duration"] = round(clip["out"] - clip["in"], 2)
        meta["laya"] = clip.get("laya", {})
        (out_dir / f"{stem}.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append({"file": target.name, **meta})
    return written
