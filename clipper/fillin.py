"""Step 8 (optional): the AI fills in one clip's details.

It reads only that clip's words, in clip-local time, plus the style notes,
never the whole video, so each call stays small.
"""
from __future__ import annotations

from clipper.ai import AIConfig, AIError, complete_json
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

SYSTEM = """You prepare one short vertical video clip for posting on
TikTok, Reels or Shorts. You get only this clip: its kind, length, the
creator's style notes and its transcript, with times in seconds from the
clip's start. Write in the language the clip is spoken in. The creator's
style notes win over everything below.

Return one JSON object:
- title: the post title. Under 60 characters, specific to what is said
  ("Why salting pasta water doesn't make it boil faster"), not a teaser
  ("You won't believe this"). No hashtags, no emoji unless the notes ask.
- hook: the words shown big on screen for the first 2.8 seconds, over the
  opening. 3-8 words that make a stranger want the rest: the question the
  clip answers, the surprising claim, or the stakes. It must not repeat the
  first spoken line word for word, and must not give away the payoff.
- description: two sentences for the post caption: what the viewer gets,
  then why it matters. Plain words.
- caption_quote: the single most quotable sentence, copied exactly from the
  transcript.
- punch_ins: 0-3 moments where a quick zoom lands the point: the punchline,
  the key number, the reveal. Give the time (seconds from the clip's start,
  taken from the transcript, just as the key word starts) and a short
  reason. Keep them at least 3 seconds apart and not in the first 3
  seconds, where the hook title is on screen. None is better than a zoom
  on an ordinary line."""


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
    # Models occasionally return the schema with every field blank. That is a
    # failed answer, not a clip without details: ask once more, then say so.
    for _ in range(2):
        answer = complete(config, SYSTEM, user, SCHEMA, 16000)
        if str(answer.get("title") or "").strip() and str(answer.get("hook") or "").strip():
            break
    else:
        raise AIError(f"{config.model} returned an empty fill-in for clip {clip.get('id')} "
                      f"twice. Try again, pick another model, or switch fill-in off.")
    length = float(clip["duration"])
    punch_ins = []
    for item in answer.get("punch_ins") or []:
        try:
            at = min(max(0.0, float(item["at"])), length)
        except (KeyError, TypeError, ValueError):
            continue
        punch_ins.append({"at": round(at, 2), "reason": str(item.get("reason") or "")})
    return {"title": str(answer["title"]).strip()[:80],
            "hook": str(answer.get("hook") or ""),
            "description": str(answer.get("description") or ""),
            "caption_quote": str(answer.get("caption_quote") or ""),
            "punch_ins": punch_ins}
