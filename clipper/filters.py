from __future__ import annotations

from pathlib import Path

VERTICAL_W, VERTICAL_H = 1080, 1920


def crop_expression(width: int, height: int, crop_x: str | int = "center") -> str:
    target = int(height * 9 / 16)
    if target >= width:
        return ""  # already at or narrower than 9:16
    if crop_x == "center":
        x = (width - target) // 2
    else:
        x = min(max(0, int(crop_x)), width - target)
    return f"crop={target}:{height}:{x}:0"


def subtitles_expression(path: Path) -> str:
    """Build a `subtitles=` filter argument.

    The path must already be space-free. ffmpeg truncates subtitle paths at the
    first space regardless of quoting or backslash escaping, so callers stage
    the file into a short temp directory first (see render.staged_subtitles).
    """
    text = path.as_posix()
    if " " in text:
        raise ValueError(
            f"Subtitle path contains a space: {text}. ffmpeg's subtitles filter "
            "truncates at the first space. Stage the file into a space-free "
            "temp directory before building the filter."
        )
    return "subtitles=" + text.replace(":", r"\:")


LAYOUTS = ("fit", "crop")
PUNCH_ZOOM = 0.15    # how far in a punch-in goes: 1.15x
PUNCH_RAMP = 0.2     # seconds to zoom in, and again to zoom out
PUNCH_HOLD = 1.2     # seconds held at full zoom


def punch_in_expression(times: list[float], width: int, height: int, fps: float) -> str:
    """A quick centred zoom at each time (seconds from the clip's start).

    Each punch-in ramps in over PUNCH_RAMP s, holds, and ramps out, so the cut
    feels like a camera push, not a jump. Overlapping ones merge (max)."""
    ramps = []
    for at in sorted(times)[:6]:
        end = at + 2 * PUNCH_RAMP + PUNCH_HOLD
        ramps.append(f"max(0,min(1,min((it-{at:.3f})/{PUNCH_RAMP},({end:.3f}-it)/{PUNCH_RAMP})))")
    level = ramps[0]
    for ramp in ramps[1:]:
        level = f"max({level},{ramp})"
    # Single quotes keep the commas inside the expression away from the
    # filter graph parser; zoompan evaluates `z` once per frame.
    zoom = f"1+{PUNCH_ZOOM}*{level}"
    return (f"zoompan=z='{zoom}':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":s={width}x{height}:fps={fps:g}")


def fit_expression() -> str:
    """The whole frame, scaled to the output width, centred over a blurred and
    zoomed copy of itself: nothing on screen is lost to a crop."""
    return (f"split=2[bg][fg];"
            f"[bg]scale={VERTICAL_W}:{VERTICAL_H}:force_original_aspect_ratio=increase,"
            f"crop={VERTICAL_W}:{VERTICAL_H},boxblur=20:2[bgb];"
            f"[fg]scale={VERTICAL_W}:-2[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,format=yuv420p")


def build_filter_chain(width: int, height: int, vertical: bool,
                       subtitle_path: Path | None,
                       crop_x: str | int = "center", layout: str = "crop",
                       punch_ins: list[float] | None = None, fps: float = 30.0) -> str | None:
    """Layout, then punch-in zooms, then captions on top (they never zoom)."""
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown layout {layout!r}. Valid: {', '.join(LAYOUTS)}.")
    parts: list[str] = []
    if vertical and layout == "fit" and int(height * 9 / 16) < width:
        parts.append(fit_expression())
    elif vertical:
        crop = crop_expression(width, height, crop_x)
        if crop:
            parts.append(crop)
        parts.append(f"scale={VERTICAL_W}:{VERTICAL_H}")
    if punch_ins:
        out_w, out_h = (VERTICAL_W, VERTICAL_H) if vertical else (width - width % 2,
                                                                  height - height % 2)
        parts.append(punch_in_expression(punch_ins, out_w, out_h, fps))
    if subtitle_path is not None:
        parts.append(subtitles_expression(subtitle_path))
    return ",".join(parts) if parts else None
