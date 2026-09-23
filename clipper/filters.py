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


def fit_expression() -> str:
    """The whole frame, scaled to the output width, centred over a blurred and
    zoomed copy of itself: nothing on screen is lost to a crop."""
    return (f"split=2[bg][fg];"
            f"[bg]scale={VERTICAL_W}:{VERTICAL_H}:force_original_aspect_ratio=increase,"
            f"crop={VERTICAL_W}:{VERTICAL_H},boxblur=20:2[bgb];"
            f"[fg]scale={VERTICAL_W}:-2[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2")


def build_filter_chain(width: int, height: int, vertical: bool,
                       subtitle_path: Path | None,
                       crop_x: str | int = "center", layout: str = "crop") -> str | None:
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
    if subtitle_path is not None:
        parts.append(subtitles_expression(subtitle_path))
    return ",".join(parts) if parts else None
