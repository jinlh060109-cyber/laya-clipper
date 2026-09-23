"""Steps 3-4: the AI reads the whole transcript once and proposes clips.

One call per video returns the kind of video, the candidate clips (start and
end on the transcript's timestamps) and the typed questions Laya should
answer about each clip. Everything the AI returns is checked: boundaries are
moved onto real words, clips under 10 s are dropped, overlaps removed, and
questions Laya cannot take are left out. Without an AI, or if the call fails,
the transcript is cut into sentence-aligned chunks instead, and Laya asks
only its built-in questions.
"""
from __future__ import annotations

import math

from clipper import questions as q
from clipper.ai import AIConfig, AIError, complete_json
from clipper.captions import join_words
from clipper.run import Run
from clipper.silence import speech_only

CONTENT_TYPES = ("podcast", "interview", "talking_head", "tutorial", "lecture",
                 "comedy", "stream", "gaming", "vlog", "news", "other")
MIN_CLIP = 10.0
TOKENS_PER_WORD = 1.35
LAYA_TEXT_BUDGET = 300  # of 512: the question and its options take up to 192
CHUNK_MIN, CHUNK_MAX = 20.0, 45.0

_STR = {"type": "string"}
SCHEMA = {
    "type": "object",
    "properties": {
        "content_type": {"type": "string", "enum": list(CONTENT_TYPES)},
        "summary": _STR,
        "questions": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": _STR,
                "type": {"type": "string", "enum": ["score", "choice", "noul"]},
                "instructions": _STR,
                "levels": {"type": "array", "items": _STR},
                "choices": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"key": _STR, "description": _STR},
                    "required": ["key", "description"],
                    "additionalProperties": False}},
                "true_means": _STR,
                "false_means": _STR,
            },
            "required": ["id", "type", "instructions", "levels", "choices",
                         "true_means", "false_means"],
            "additionalProperties": False}},
        "weights": {"type": "array", "items": {
            "type": "object",
            "properties": {"question_id": _STR, "weight": {"type": "number"}},
            "required": ["question_id", "weight"],
            "additionalProperties": False}},
        "candidates": {"type": "array", "items": {
            "type": "object",
            "properties": {"start": {"type": "number"}, "end": {"type": "number"},
                           "category": _STR, "reason": _STR, "hook_line": _STR},
            "required": ["start", "end", "category", "reason", "hook_line"],
            "additionalProperties": False}},
    },
    "required": ["content_type", "summary", "questions", "weights", "candidates"],
    "additionalProperties": False,
}

SYSTEM = f"""You are the editor who picks short clips from a long video.

You get the video's full transcript, one line per stretch of speech, each
starting with [start-end] in seconds. Do three things in one answer:

1. content_type: what kind of video this is.
2. candidates: the moments worth posting as short clips. For each, give start
   and end in seconds taken from the transcript's timestamps, a short
   category (for example tip, story, joke, insight, reveal, rant), why it
   works (reason), and the line a viewer would hear first (hook_line).
   - Each clip is 15-60 seconds and at most about 150 words. A separate model
     with a 512-token window reads each clip; longer clips get cut off before
     it has read them.
   - Start on the strongest line, not on a wind-up. End when the thought
     lands. Start and end on sentence boundaries.
   - Clips never overlap. Propose 5-25, more for longer videos, best first.
   - Every clip must make sense to someone who has not seen the video.
3. questions: up to 6 typed questions that a small decision model will
   answer about every clip, written for this video (the model already asks
   how clip-worthy the clip is, how strong its first line is, whether it
   stands alone and whether it ends cleanly, so ask about what matters for
   this kind of video: for a tutorial, whether the tip is concrete; for
   comedy, whether the joke lands). Types:
   - score: fill `levels` with 3-5 short descriptions from worst to best.
   - choice: fill `choices` with 2-6 options (key + description).
   - noul: a yes/no statement; fill `true_means` and `false_means`.
   Leave the fields a type does not use empty. Keep every text under
   {q.MAX_TEXT} characters. weights: how much each score or noul question
   should count (any positive numbers; they are rescaled).

If the viewer's goal is given, let it decide which moments you pick."""


def _words(transcript: dict) -> list[dict]:
    return [w for seg in transcript.get("segments") or [] for w in seg.get("words") or []]


def transcript_lines(transcript: dict) -> str:
    lines = []
    for seg in speech_only(transcript)["segments"]:
        words = seg["words"]
        lines.append(f"[{words[0]['start']:.2f}-{words[-1]['end']:.2f}] "
                     f"{join_words([w['word'] for w in words])}")
    return "\n".join(lines)


def build_prompt(transcript: dict, duration: float, user_prompt: str) -> tuple[str, str]:
    goal = user_prompt.strip() or "(none given: pick what most viewers would enjoy)"
    user = (f"Video length: {duration:.0f} seconds.\n"
            f"Viewer's goal: {goal}\n\n"
            f"Transcript:\n{transcript_lines(transcript)}")
    return SYSTEM, user


def _clip(start: float, end: float, words: list[dict], **fields) -> dict:
    inside = [w for w in words if w["start"] >= start - 1e-6 and w["end"] <= end + 1e-6]
    text = join_words([w["word"] for w in inside])
    tokens = math.ceil(len(inside) * TOKENS_PER_WORD)
    return {"start": round(start, 3), "end": round(end, 3),
            "duration": round(end - start, 3), **fields,
            "text": text, "est_tokens": tokens, "truncated": tokens > LAYA_TEXT_BUDGET}


def snap(candidates: list[dict], words: list[dict], duration: float) -> list[dict]:
    """Put each clip on real word boundaries and keep only usable, separate clips."""
    words = sorted(words, key=lambda w: w["start"])
    placed = []
    for cand in candidates:
        try:
            start, end = float(cand["start"]), min(float(cand["end"]), duration)
        except (KeyError, TypeError, ValueError):
            continue
        first = next((w for w in words if w["end"] > start), None)
        last = next((w for w in reversed(words) if w["start"] < end), None)
        if first is None or last is None or last["end"] <= first["start"]:
            continue
        s, e = first["start"], min(last["end"], duration)
        if e - s < MIN_CLIP:
            continue
        placed.append(_clip(s, e, words, category=str(cand.get("category") or "clip"),
                            reason=str(cand.get("reason") or ""),
                            hook_line=str(cand.get("hook_line") or "")))
    kept: list[dict] = []
    for clip in sorted(placed, key=lambda c: c["start"]):
        if kept and clip["start"] < kept[-1]["end"]:
            continue
        kept.append(clip)
    return [{"id": i, **clip} for i, clip in enumerate(kept)]


def fallback_candidates(transcript: dict, duration: float) -> list[dict]:
    """Sentence-aligned 20-45 s chunks, for runs without an AI."""
    words = sorted(_words(speech_only(transcript)), key=lambda w: w["start"])
    sentences, current = [], []
    for word in words:
        current.append(word)
        if word["word"].strip().endswith((".", "?", "!", "。", "？", "！")):
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)

    chunks, buffer = [], []
    for sentence in sentences:
        if buffer and sentence[-1]["end"] - buffer[0]["start"] > CHUNK_MAX:
            chunks.append(buffer)
            buffer = []
        buffer = buffer + sentence
        if buffer[-1]["end"] - buffer[0]["start"] >= CHUNK_MIN:
            chunks.append(buffer)
            buffer = []
    if buffer:
        chunks.append(buffer)

    out = []
    for chunk in chunks:
        start, end = chunk[0]["start"], min(chunk[-1]["end"], duration)
        if end - start >= MIN_CLIP:
            out.append(_clip(start, end, words, category="chunk", reason="",
                             hook_line=""))
    return [{"id": i, **clip} for i, clip in enumerate(out)]


def _without_ai(transcript: dict, duration: float, error: str | None) -> dict:
    return {"content_type": "unknown", "summary": "", "ai": None, "ai_error": error,
            "questions": dict(q.BUILTIN), "weights": q.combined_weights({}, {}),
            "candidates": fallback_candidates(transcript, duration), "problems": []}


def run_segment(run: Run, config: AIConfig | None, user_prompt: str = "",
                complete=complete_json) -> dict:
    transcript = run.read_json("transcript.json")
    duration = float(run.read_json("source.json")["duration"])
    words = _words(speech_only(transcript))

    if not words:
        # Nothing was said: nothing to read, and nothing for Laya to score.
        result = {**_without_ai(transcript, duration, None), "candidates": []}
    elif config is None:
        result = _without_ai(transcript, duration, None)
    else:
        system, user = build_prompt(transcript, duration, user_prompt)
        try:
            answer = complete(config, system, user, SCHEMA, 64000)
            candidates = snap(answer.get("candidates") or [], words, duration)
            if not candidates:
                raise AIError("The AI proposed no usable clips (all too short or off the transcript).")
        except AIError as exc:
            result = _without_ai(transcript, duration, str(exc))
        else:
            ai_questions, problems = q.from_ai(answer.get("questions") or [])
            weights = {str(w.get("question_id")): w.get("weight")
                       for w in answer.get("weights") or [] if isinstance(w, dict)}
            weights = {q._slug(k): v for k, v in weights.items()}
            content_type = answer.get("content_type")
            result = {
                "content_type": content_type if content_type in CONTENT_TYPES else "other",
                "summary": str(answer.get("summary") or ""),
                "ai": config.describe(), "ai_error": None,
                "questions": {**q.BUILTIN, **ai_questions},
                "weights": q.combined_weights(weights, ai_questions),
                "candidates": candidates, "problems": problems,
            }
    run.write_json("segments.json", result)
    return result
