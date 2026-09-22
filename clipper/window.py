from __future__ import annotations


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

        lo, hi = int(start), max(int(start) + 1, int(end))
        slice_ = energy[lo:hi] or [0.0]
        peak = max(slice_)
        span = max(1e-6, end - start)
        offset = (lo + slice_.index(peak) - start) / span

        windows.append({
            "id": index,
            "start": start,
            "end": end,
            "text": _render(inside),
            "preceding": _render(context),
            "position": start / duration if duration else 0.0,
            "energy_mean": sum(slice_) / len(slice_),
            "energy_peak": peak,
            "energy_peak_offset": min(1.0, max(0.0, offset)),
        })
        index += 1

        if end >= words[-1]["end"]:
            break
        nxt = next((i for i, w in enumerate(words) if w["start"] >= anchor + step_seconds), None)
        if nxt is None or nxt <= cursor:
            break
        cursor = nxt
    return windows
