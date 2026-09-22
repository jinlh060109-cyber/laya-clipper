from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    words: list[dict]
    speaker: str

    @property
    def text(self) -> str:
        return " ".join(w["word"].strip() for w in self.words)


def words_in_range(transcript: dict, start: float, end: float) -> list[dict]:
    out: list[dict] = []
    for segment in transcript["segments"]:
        for word in segment["words"]:
            if word["end"] > start and word["start"] < end:
                out.append(word)
    return out


def build_cues(words: list[dict], max_chars: int = 42,
               max_seconds: float = 3.0) -> list[Cue]:
    cues: list[Cue] = []
    buffer: list[dict] = []

    def flush() -> None:
        if buffer:
            cues.append(Cue(start=buffer[0]["start"], end=buffer[-1]["end"],
                            words=list(buffer), speaker=buffer[0]["speaker"]))
            buffer.clear()

    for word in words:
        if buffer:
            too_long = len(" ".join(w["word"].strip() for w in buffer + [word])) > max_chars
            too_slow = word["end"] - buffer[0]["start"] > max_seconds
            changed = word["speaker"] != buffer[0]["speaker"]
            if too_long or too_slow or changed:
                flush()
        buffer.append(word)
    flush()
    return cues


def rebase(cues: list[Cue], clip_start: float) -> list[Cue]:
    """Shift cues into clip-local time. Reference project: adjusted = original
    - clip_start, clamped at 0."""
    out: list[Cue] = []
    for cue in cues:
        if cue.end <= clip_start:
            continue
        out.append(Cue(
            start=max(0.0, cue.start - clip_start),
            end=max(0.0, cue.end - clip_start),
            words=[{**w,
                    "start": max(0.0, w["start"] - clip_start),
                    "end": max(0.0, w["end"] - clip_start)} for w in cue.words],
            speaker=cue.speaker,
        ))
    return out


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def render_srt(cues: list[Cue]) -> str:
    blocks = [
        f"{i}\n{format_srt_time(c.start)} --> {format_srt_time(c.end)}\n{c.text}\n"
        for i, c in enumerate(cues, start=1)
    ]
    return "\n".join(blocks) + "\n"
