"""Step 8 (optional): the AI fills in one clip's details.

It reads only that clip's words, in clip-local time, plus the style notes,
never the whole video, so each call stays small.
"""
from __future__ import annotations

from clipper.ai import AIConfig, complete_json
from clipper.captions import join_words

_STR = {"type": "string"}
SCHEMA = {
    "type": "object",
    "properties": {
        "title": _STR, "hook": _STR, "description": _STR, "caption_quote": _STR,
        "punch_ins": {"type": "array", "items": {
            "type": "object",
            "properties": {"at": {"type": "number"}, "reason": _STR},
            "required": ["at", "reason"], "additionalProperties": False}},
    },
    "required": ["title", "hook", "description", "caption_quote", "punch_ins"],
    "additionalProperties": False,
}

SYSTEM = """You prepare one short video clip for posting. You get only this
clip's transcript, with times in seconds from the clip's start, and the
creator's style notes. Return:
- title: under 60 characters, specific, not clickbait.
- hook: the one line to put on screen in the first second.
- description: two sentences for the post.
- caption_quote: the single most quotable sentence, word for word from the transcript.
- punch_ins: 0-3 moments (seconds from the clip's start) where a quick zoom
  would land the point, each with a short reason.
Follow the style notes for tone and wording."""


def clip_slice(transcript: dict, start: float, end: float) -> str:
    """The clip's words as `[s-e] text` lines, in seconds from the clip's start."""
    lines = []
    for seg in transcript.get("segments") or []:
        words = [w for w in seg.get("words") or []
                 if w["start"] >= start - 1e-6 and w["end"] <= end + 1e-6]
        if words:
            lines.append(f"[{words[0]['start'] - start:.2f}-{words[-1]['end'] - start:.2f}] "
                         f"{join_words([w['word'] for w in words])}")
    return "\n".join(lines)


def fill_in(config: AIConfig, clip: dict, transcript: dict, notes: str,
            complete=complete_json) -> dict:
    user = (f"Clip category: {clip.get('category', 'clip')}\n"
            f"Clip length: {clip['duration']:.1f} seconds\n"
            f"Style notes: {notes.strip() or '(none)'}\n\n"
            f"Transcript:\n{clip_slice(transcript, clip['start'], clip['end'])}")
    answer = complete(config, SYSTEM, user, SCHEMA, 16000)
    length = float(clip["duration"])
    punch_ins = []
    for item in answer.get("punch_ins") or []:
        try:
            at = min(max(0.0, float(item["at"])), length)
        except (KeyError, TypeError, ValueError):
            continue
        punch_ins.append({"at": round(at, 2), "reason": str(item.get("reason") or "")})
    return {"title": str(answer.get("title") or clip.get("title_hint") or "")[:80],
            "hook": str(answer.get("hook") or ""),
            "description": str(answer.get("description") or ""),
            "caption_quote": str(answer.get("caption_quote") or ""),
            "punch_ins": punch_ins}
