"""Step 10: every ticked clip is cut from the original and edited.

Each clip gets its own folder under clips/:
  cut.mp4       the untouched cut, for any editor to start from
  final.mp4     the default edit: layout (fit or crop), captions, encoder
  edit.json     the structured edit instruction (step 9)
  prompt.md     the same for an AI editor, with the style notes and words
  captions.srt  the clip's captions, when anyone speaks
"""
from __future__ import annotations

import contextlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from clipper.captions import cues_for, rebase, render_ass, render_srt, words_in_range
from clipper.fillin import clip_slice
from clipper.filters import build_filter_chain
from clipper.hardware import encoder_args, pick_encoder, probe_encoders
from clipper.ingest import ffmpeg_tail
from clipper.preflight import preflight
from clipper.prompts import edit_spec, render_prompt
from clipper.run import Run
from clipper.silence import speech_only

MIN_CLIP_SECONDS = 10.0


def slugify(text: str, limit: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:limit].rstrip("-") or "clip")


@contextlib.contextmanager
def staged_subtitles(content: str, suffix: str):
    """Write subtitles alone into a fresh temp directory under a bare name.

    ffmpeg's subtitles filter truncates paths at the first space whatever the
    quoting, so ffmpeg runs in this directory and is given only the bare name.
    """
    directory = Path(tempfile.mkdtemp(prefix="clipper_"))
    try:
        path = directory / f"sub{suffix}"
        path.write_text(content, encoding="utf-8")
        yield path
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def display_size(video: dict) -> tuple[int, int]:
    """(width, height) as displayed: ffmpeg autorotates before any filter runs."""
    width, height = video["width"], video["height"]
    if int(video.get("rotation") or 0) % 180 == 90:
        return height, width
    return width, height


def _encode(encoder: str) -> list[str]:
    # Plain 4:2:0 for players everywhere; hardware encoders convert it themselves.
    return [*encoder_args(encoder), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart"]


# Errors only: at the default level ffmpeg's closing statistics fill the last
# lines of stderr, which is all an error message quotes (see ffmpeg_tail).


def cut_command(ffmpeg, source: Path, start: float, end: float, output: Path,
                encoder: str) -> list[str]:
    return [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-accurate_seek", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
            "-i", str(source), *_encode(encoder), str(output)]


def final_command(ffmpeg, source: Path, start: float, end: float, output: Path,
                  chain: str | None, encoder: str) -> list[str]:
    cmd = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-accurate_seek", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
           "-i", str(source)]
    if chain:
        cmd += ["-vf", chain]
    return cmd + [*_encode(encoder), str(output)]


def _ffmpeg(cmd: list[str], output: Path, cwd: Path | None = None) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", check=False, cwd=cwd)
    if result.returncode != 0:
        output.unlink(missing_ok=True)  # never leave a half-written clip behind
        raise RuntimeError(f"Could not render {output.parent.name}/{output.name}:\n"
                           f"{ffmpeg_tail(result.stderr)}")


def preview_frame(run: Run, clip_id: str, style: dict, ffmpeg=None) -> bytes:
    """One half-size JPEG frame of a clip as it will look: layout and caption
    style applied, taken while a caption is on screen."""
    clip = next((c for c in run.read_json("selection.json")["clips"] if c["id"] == clip_id), None)
    if clip is None:
        raise ValueError(f"Unknown clip id {clip_id!r}.")
    source_meta = run.read_json("source.json")
    ffmpeg = ffmpeg or preflight(require_subtitles=True).ffmpeg
    width, height = display_size(source_meta["video"])
    vertical, layout = style["vertical"], style["layout"]
    preset = style.get("caption_style", "classic")
    transcript = speech_only(run.read_json("transcript.json"))
    cues = cues_for(words_in_range(transcript, clip["start"], clip["end"]), preset)
    # A moment a little into a caption, so the word highlighting shows.
    cue = next((c for c in cues if len(c.words) > 1), cues[0] if cues else None)
    at = cue.start + 0.6 * (cue.end - cue.start) if cue else (clip["start"] + clip["end"]) / 2
    shown = rebase(cues, at) if style["captions"] == "burn" else []
    out_height = 1920 if vertical else height

    def grab(chain: str | None, cwd: Path | None) -> bytes:
        vf = ",".join(part for part in (chain, "scale=iw/2:ih/2") if part)
        cmd = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-ss", f"{at:.3f}",
               "-i", str(Path(source_meta["path"]).resolve()), "-frames:v", "1",
               "-vf", vf, "-f", "image2", "-c:v", "mjpeg", "-q:v", "4", "pipe:1"]
        result = subprocess.run(cmd, capture_output=True, check=False, cwd=cwd)
        if result.returncode != 0 or not result.stdout:
            raise RuntimeError("Could not render the preview:\n"
                               + ffmpeg_tail(result.stderr.decode("utf-8", "replace")))
        return result.stdout

    if shown:
        ass = render_ass(shown, out_height, uppercase=style["caption_case"] == "upper",
                         layout=layout, preset=preset)
        with staged_subtitles(ass, ".ass") as staged:
            return grab(build_filter_chain(width, height, vertical, Path(staged.name),
                                           layout=layout), staged.parent)
    return grab(build_filter_chain(width, height, vertical, None, layout=layout), None)


def run_edit(run: Run, style: dict, fills: dict | None = None,
             progress: Callable[[int, int], None] | None = None,
             ffmpeg=None, encoders: list[str] | None = None) -> list[dict]:
    fills = fills or {}
    source_meta = run.read_json("source.json")
    duration = float(source_meta["duration"])
    clips = [c for c in run.read_json("selection.json")["clips"] if c.get("include")]
    if not clips:
        raise ValueError("No clips are ticked. Tick at least one clip in the preview.")
    for clip in clips:
        length = min(clip["end"], duration) - clip["start"]
        if length < MIN_CLIP_SECONDS:
            raise ValueError(f"Clip {clip['id']} is {length:.1f} s, under the 10 s minimum.")

    if ffmpeg is None:
        ffmpeg = preflight(require_subtitles=style["captions"] == "burn").ffmpeg
    encoder = pick_encoder(style["encoder"],
                           probe_encoders(str(ffmpeg)) if encoders is None else encoders)
    segments = run.read_json("segments.json") if run.exists("segments.json") else {}
    content_type = segments.get("content_type", "unknown")
    transcript = speech_only(run.read_json("transcript.json"))
    source = Path(source_meta["path"]).resolve()
    width, height = display_size(source_meta["video"])
    root = run.clips_dir().resolve()  # absolute: ffmpeg may run in another cwd

    done = []
    for index, clip in enumerate(clips, start=1):
        start, end = clip["start"], min(clip["end"], duration)
        fill = fills.get(clip["id"])
        title = (fill or {}).get("title") or clip.get("title_hint") or clip["id"]
        stem = f"{index:02d}-{slugify(title)}"
        folder = root / stem
        folder.mkdir(parents=True, exist_ok=True)

        spec = edit_spec(clip, style, fill, content_type, stem)
        preset = style.get("caption_style", "classic")
        cues = rebase(cues_for(words_in_range(transcript, start, end), preset), start)
        (folder / "edit.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
        (folder / "prompt.md").write_text(
            render_prompt(spec, style.get("notes", ""), clip_slice(transcript, start, end)),
            encoding="utf-8")
        srt = folder / "captions.srt"
        if cues:
            srt.write_text(render_srt(cues), encoding="utf-8")
        else:
            srt.unlink(missing_ok=True)

        _ffmpeg(cut_command(ffmpeg, source, start, end, folder / "cut.mp4", encoder),
                folder / "cut.mp4")
        vertical = style["vertical"]
        out_height = 1920 if vertical else height
        final = folder / "final.mp4"
        # The AI's punch-in moments become real zooms when the style asks.
        zooms = ([p["at"] for p in spec["punch_ins"]]
                 if style.get("punch_ins", True) else None)
        fps = float(source_meta["video"].get("fps") or 30.0)
        if style["captions"] == "burn" and cues:
            ass = render_ass(cues, out_height, uppercase=style["caption_case"] == "upper",
                             layout=style["layout"], preset=preset)
            with staged_subtitles(ass, ".ass") as staged:
                chain = build_filter_chain(width, height, vertical, Path(staged.name),
                                           layout=style["layout"], punch_ins=zooms, fps=fps)
                _ffmpeg(final_command(ffmpeg, source, start, end, final, chain, encoder),
                        final, cwd=staged.parent)
        else:
            chain = build_filter_chain(width, height, vertical, None, layout=style["layout"],
                                       punch_ins=zooms, fps=fps)
            _ffmpeg(final_command(ffmpeg, source, start, end, final, chain, encoder), final)

        done.append({"id": clip["id"], "folder": stem, "title": spec["title"],
                     "duration": round(end - start, 2), "layout": style["layout"],
                     "encoder": encoder, "punch_ins": len(zooms or []),
                     "final": f"clips/{stem}/final.mp4",
                     "prompt": f"clips/{stem}/prompt.md"})
        if progress is not None:
            progress(index, len(clips))
    run.write_json("clips.json", {"clips": done})
    return done
