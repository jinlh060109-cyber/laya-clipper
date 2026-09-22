from __future__ import annotations

import math


def _flatten(transcript: dict) -> list[dict]:
    words: list[dict] = []
    for seg in transcript["segments"]:
        words.extend(seg["words"])
    return words


def _render(words: list[dict]) -> str:
    """Words to speaker-labelled text, emitting a label only on speaker change."""
    lines: list[str] = []
    current: str | None = None
    buffer: list[str] = []
    for word in words:
        if word["speaker"] != current:
            if buffer:
                lines.append(f"{current}: {' '.join(buffer)}")
            current = word["speaker"]
            buffer = []
        buffer.append(word["word"])
    if buffer:
        lines.append(f"{current}: {' '.join(buffer)}")
    return "\n".join(lines)


def _energy_stats(energy: list[float], start: float, end: float) -> tuple[float, float, float]:
    """Mean, peak, and peak-offset over the energy bins fully contained in [start, end).

    `energy[i]` is the value for the whole second `[i, i + 1)`. Only bins entirely
    inside the window are used: a bin straddling `start` would leak pre-window
    energy into the stats, and a bin straddling `end` would count a mostly
    out-of-window second as if it were fully in-window. Both ends are therefore
    trimmed inward with ceil/floor -- never truncated with int() -- so nothing
    outside the window can leak in and nothing inside it is silently mis-weighted.

    The peak's time is taken at its bin's midpoint (bin `i` -> `i + 0.5`), since a
    per-second value represents that whole second, not its left edge.

    If the window is shorter than the distance to the next bin boundary, no bin
    is fully contained; fall back to the single bin covering the window's
    midpoint and report offset 0.5 (undefined within a sub-second window) rather
    than deriving a value that could fall outside [0, 1] and get silently
    clamped, which would mask the peak having come from outside the window.
    """
    span = max(1e-6, end - start)
    lo = max(0, math.ceil(start))
    hi = min(len(energy), math.floor(end))

    if lo < hi:
        slice_ = energy[lo:hi]
        peak = max(slice_)
        peak_bin = lo + slice_.index(peak)
        offset = ((peak_bin + 0.5) - start) / span
    else:
        mid = max(0, min(len(energy) - 1, int((start + end) / 2))) if energy else 0
        slice_ = [energy[mid]] if energy else [0.0]
        peak = slice_[0]
        offset = 0.5

    mean = sum(slice_) / len(slice_)
    # By construction `offset` already lies in [0, 1] in both branches above.
    # This clamp is float-safety only -- it must never again be relied on to
    # mask an out-of-range value (that was the original defect).
    offset = min(1.0, max(0.0, offset))
    return mean, peak, offset


def build_windows(transcript: dict, energy: list[float], duration: float,
                  window_seconds: float = 30.0, step_seconds: float = 10.0,
                  context_seconds: float = 20.0) -> list[dict]:
    words = _flatten(transcript)
    if not words:
        return []

    windows: list[dict] = []
    cursor = 0
    index = 0
    while cursor < len(words):
        anchor = words[cursor]["start"]
        inside = [w for w in words if anchor <= w["start"] < anchor + window_seconds]
        if not inside:
            break
        start, end = inside[0]["start"], inside[-1]["end"]
        context = [w for w in words if anchor - context_seconds <= w["start"] < anchor]

        energy_mean, energy_peak, energy_peak_offset = _energy_stats(energy, start, end)

        windows.append({
            "id": index,
            "start": start,
            "end": end,
            "text": _render(inside),
            "preceding": _render(context),
            "position": start / duration if duration else 0.0,
            "energy_mean": energy_mean,
            "energy_peak": energy_peak,
            "energy_peak_offset": energy_peak_offset,
        })
        index += 1

        if end >= words[-1]["end"]:
            break
        nxt = next((i for i, w in enumerate(words) if w["start"] >= anchor + step_seconds), None)
        if nxt is None or nxt <= cursor:
            break
        cursor = nxt
    return windows


def write_windows(run) -> list[dict]:
    """Build windows from the run's transcript and energy, and write windows.json.

    The one code path for this stage; the CLI and the web runner both call it.
    """
    source = run.read_json("source.json")
    windows = build_windows(run.read_json("transcript.json"), source["energy"],
                            duration=source["duration"])
    run.write_json("windows.json", {"windows": windows})
    return windows
