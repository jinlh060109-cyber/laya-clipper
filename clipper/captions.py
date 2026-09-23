from __future__ import annotations

from dataclasses import dataclass


def _unspaced(ch: str) -> bool:
    """Scripts written without spaces between words: CJK ideographs, kana,
    and fullwidth punctuation. Hangul is excluded; Korean spaces its words."""
    code = ord(ch)
    return (0x3000 <= code <= 0x30FF      # CJK punctuation, hiragana, katakana
            or 0x3400 <= code <= 0x4DBF   # CJK extension A
            or 0x4E00 <= code <= 0x9FFF   # CJK unified ideographs
            or 0xF900 <= code <= 0xFAFF   # CJK compatibility ideographs
            or 0xFF00 <= code <= 0xFFEF)  # fullwidth forms


def join_words(words: list[str]) -> str:
    """Join word tokens, with no space where both neighbours are CJK.

    WhisperX aligns Chinese and Japanese per character, so a plain
    " ".join would print "我 是 谁" instead of "我是谁".
    """
    out = ""
    for word in (w.strip() for w in words):
        if not word:
            continue
        if out and not (_unspaced(out[-1]) and _unspaced(word[0])):
            out += " "
        out += word
    return out


def display_width(text: str) -> int:
    """Width in Latin-character units; a CJK glyph is about two wide."""
    return sum(2 if _unspaced(ch) else 1 for ch in text)


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    words: list[dict]
    speaker: str

    @property
    def text(self) -> str:
        return join_words([w["word"] for w in self.words])


def words_in_range(transcript: dict, start: float, end: float) -> list[dict]:
    out: list[dict] = []
    for segment in transcript["segments"]:
        for word in segment["words"]:
            if word["end"] > start and word["start"] < end:
                out.append(word)
    return out


def build_cues(words: list[dict], max_chars: int = 42,
               max_seconds: float = 3.0) -> list[Cue]:
    """Group words into caption lines that read as phrases.

    A line ends at a sentence end, or after a comma once it is half full, and
    otherwise where the length or time limit forces it. A forced break never
    strands a number at the end of a line ("... and 67.9" / "hours ..."): the
    number moves to the next line with the word it belongs to.
    """
    cues: list[Cue] = []
    buffer: list[dict] = []

    def flush(upto: int | None = None) -> None:
        taken = buffer[:upto] if upto is not None else list(buffer)
        if taken:
            cues.append(Cue(start=taken[0]["start"], end=taken[-1]["end"],
                            words=taken, speaker=taken[0]["speaker"]))
        del buffer[:len(taken)]

    for word in words:
        if buffer:
            text = join_words([w["word"] for w in buffer + [word]])
            too_long = display_width(text) > max_chars
            too_slow = word["end"] - buffer[0]["start"] > max_seconds
            changed = word["speaker"] != buffer[0]["speaker"]
            if changed:
                flush()
            elif too_long or too_slow:
                last = buffer[-1]["word"].strip()
                strand = len(buffer) > 1 and last[:1].isdigit()
                flush(-1 if strand else None)
        buffer.append(word)
        token = word["word"].strip()
        if token.endswith((".", "?", "!", "。", "？", "！")):
            flush()
        elif token.endswith((",", ";", ":", "，")) and \
                display_width(join_words([w["word"] for w in buffer])) >= max_chars / 2:
            flush()
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


ASS_HEADER = """[Script Info]
ScriptType: v4.00+
WrapStyle: 2
ScaledBorderAndShadow: yes
PlayResX: {play_x}
PlayResY: {play_y}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Arial,{size},&H00FFFFFF,&H0000D7FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},1,2,60,60,{margin},1
Style: Speaker,Arial,{speaker_size},&H00D0D0D0,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,1,40,40,{speaker_margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def font_size_for_height(height: int) -> int:
    """Reference guide: 720p -> 20, 1080p -> 24, 4K -> 48."""
    if height <= 720:
        return 20
    if height <= 1080:
        return 24
    if height <= 1440:
        return 32
    return 48


def format_ass_time(seconds: float) -> str:
    centis = int(round(seconds * 100))
    hours, centis = divmod(centis, 360_000)
    minutes, centis = divmod(centis, 6_000)
    secs, centis = divmod(centis, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _escape(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")")


def _karaoke(cue: Cue, uppercase: bool = False) -> str:
    # \k durations are cumulative from the line start, so each word holds until
    # the next begins; using word length alone drifts early after every pause.
    line, previous = "", ""
    for i, word in enumerate(cue.words):
        until = cue.words[i + 1]["start"] if i + 1 < len(cue.words) else word["end"]
        centis = max(1, int(round((until - word["start"]) * 100)))
        text = word["word"].strip()
        if uppercase:
            text = text.upper()
        if not text:
            continue
        if previous and not (_unspaced(previous[-1]) and _unspaced(text[0])):
            line += " "
        line += f"{{\\k{centis}}}{_escape(text)}"
        previous = text
    return line


def render_ass(cues: list[Cue], height: int,
               speakers: dict[str, str] | None = None,
               uppercase: bool = False, layout: str = "crop") -> str:
    """`layout="fit"` on a vertical render puts the captions just under the
    picture, which then sits in the middle of the frame."""
    size = font_size_for_height(height)
    under_picture = layout == "fit" and height >= 1920
    header = ASS_HEADER.format(
        play_x=int(height * 9 / 16) if height >= 1920 else int(height * 16 / 9),
        play_y=height,
        size=size,
        outline=max(1, size // 10),
        margin=int(height * (0.27 if under_picture else 0.08)),
        speaker_size=max(14, int(size * 0.7)),
        speaker_margin=int(height * 0.04),
    )
    lines: list[str] = []
    for cue in cues:
        start, end = format_ass_time(cue.start), format_ass_time(cue.end)
        lines.append(f"Dialogue: 0,{start},{end},Caption,,0,0,0,,{_karaoke(cue, uppercase)}")
        name = (speakers or {}).get(cue.speaker)
        if name:
            lines.append(
                f"Dialogue: 0,{start},{end},Speaker,,0,0,0,,{_escape(name)}"
            )
    return header + "\n".join(lines) + "\n"
