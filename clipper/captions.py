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
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: {play_x}
PlayResY: {play_y}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{caption_style}
Style: Speaker,Arial,{speaker_size},&H00D0D0D0,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,1,40,40,{speaker_margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


# Caption looks the user can pick. Colours are ASS &HAABBGGRR. With karaoke,
# each word starts in `secondary` and turns `primary` as it is spoken.
# `max_chars` keeps a line inside a 1080 px frame at that font size (a
# character is about 0.55 x the size wide); libass wraps anything longer.
CAPTION_STYLES: dict[str, dict] = {
    "classic": {"label": "Classic: white, words turn gold as spoken",
                "font": "Arial", "bold": True, "scale": 1.35, "primary": "&H0000D7FF",
                "secondary": "&H00FFFFFF", "outline_colour": "&H00000000",
                "back": "&H80000000", "border": 1, "outline": 4, "shadow": 1,
                "karaoke": True, "upper": False, "max_chars": 26, "tags": ""},
    "bold": {"label": "Bold: big UPPERCASE, words turn green",
             "font": "Arial Black", "bold": True, "scale": 1.6, "primary": "&H0000FF7F",
             "secondary": "&H00FFFFFF", "outline_colour": "&H00000000",
             "back": "&H80000000", "border": 1, "outline": 6, "shadow": 2,
             "karaoke": True, "upper": True, "max_chars": 18, "tags": ""},
    "boxed": {"label": "Boxed: white text on a dark box",
              "font": "Arial", "bold": True, "scale": 1.25, "primary": "&H00FFFFFF",
              "secondary": "&H00FFFFFF", "outline_colour": "&H90000000",
              "back": "&H90000000", "border": 3, "outline": 14, "shadow": 0,
              "karaoke": False, "upper": False, "max_chars": 28, "tags": ""},
    "minimal": {"label": "Minimal: clean white with a soft shadow",
                "font": "Arial", "bold": False, "scale": 1.2, "primary": "&H00FFFFFF",
                "secondary": "&H00FFFFFF", "outline_colour": "&H00000000",
                "back": "&H00000000", "border": 1, "outline": 1, "shadow": 3,
                "karaoke": False, "upper": False, "max_chars": 30, "tags": ""},
    "one_word": {"label": "One word: a single big word at a time, popping in",
                 "font": "Arial Black", "bold": True, "scale": 2.4, "primary": "&H0000E5FF",
                 "secondary": "&H0000E5FF", "outline_colour": "&H00000000",
                 "back": "&H80000000", "border": 1, "outline": 8, "shadow": 2,
                 "karaoke": False, "upper": True, "max_chars": 0,
                 "tags": r"{\fscx125\fscy125\t(0,120,\fscx100\fscy100)}"},
    "neon": {"label": "Neon: glowing cyan outline, words turn pink",
             "font": "Arial", "bold": True, "scale": 1.45, "primary": "&H00FF40FF",
             "secondary": "&H00FFFFFF", "outline_colour": "&H00FFFF00",
             "back": "&H00000000", "border": 1, "outline": 5, "shadow": 0,
             "karaoke": True, "upper": False, "max_chars": 25, "tags": r"{\blur4}"},
}


def _preset(name: str) -> dict:
    if name not in CAPTION_STYLES:
        raise ValueError(f"Unknown caption style {name!r}. "
                         f"Valid: {', '.join(CAPTION_STYLES)}.")
    return CAPTION_STYLES[name]


def cues_for(words: list[dict], preset: str = "classic") -> list[Cue]:
    """Caption lines shaped for a style: one word at a time, or short lines."""
    look = _preset(preset)
    if look["max_chars"] == 0:
        cues = []
        for i, word in enumerate(words):
            nxt = words[i + 1]["start"] if i + 1 < len(words) else word["end"]
            cues.append(Cue(start=word["start"], end=max(word["end"], min(nxt, word["start"] + 1.0)),
                            words=[word], speaker=word["speaker"]))
        return cues
    return build_cues(words, max_chars=look["max_chars"])

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


def _plain(cue: Cue, uppercase: bool = False) -> str:
    text = join_words([w["word"] for w in cue.words])
    return _escape(text.upper() if uppercase else text)


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
               uppercase: bool = False, layout: str = "crop",
               preset: str = "classic") -> str:
    """`layout="fit"` on a vertical render puts the captions just under the
    picture, which then sits in the middle of the frame. `preset` is one of
    CAPTION_STYLES."""
    look = _preset(preset)
    uppercase = uppercase or look["upper"]
    size = int(font_size_for_height(height) * look["scale"])
    under_picture = layout == "fit" and height >= 1920
    margin = int(height * (0.27 if under_picture else 0.08))
    caption_style = (
        f"Style: Caption,{look['font']},{size},{look['primary']},{look['secondary']},"
        f"{look['outline_colour']},{look['back']},{-1 if look['bold'] else 0},0,0,0,"
        f"100,100,0,0,{look['border']},{look['outline']},{look['shadow']},2,60,60,{margin},1")
    header = ASS_HEADER.format(
        play_x=int(height * 9 / 16) if height >= 1920 else int(height * 16 / 9),
        play_y=height,
        caption_style=caption_style,
        speaker_size=max(14, int(font_size_for_height(height) * 0.7)),
        speaker_margin=int(height * 0.04),
    )
    lines: list[str] = []
    for cue in cues:
        start, end = format_ass_time(cue.start), format_ass_time(cue.end)
        text = _karaoke(cue, uppercase) if look["karaoke"] else _plain(cue, uppercase)
        lines.append(f"Dialogue: 0,{start},{end},Caption,,0,0,0,,{look['tags']}{text}")
        name = (speakers or {}).get(cue.speaker)
        if name:
            lines.append(
                f"Dialogue: 0,{start},{end},Speaker,,0,0,0,,{_escape(name)}"
            )
    return header + "\n".join(lines) + "\n"
