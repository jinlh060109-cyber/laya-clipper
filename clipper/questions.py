"""The typed questions Laya answers for every candidate clip.

Four built-in questions are always asked; the AI adds up to six of its own,
written for this particular video. Laya's three answer types are `score`
(pick one of k levels), `choice` (pick one option) and `noul` (yes/no).
A question and its options share 192 of Laya's 512 tokens, so levels,
options and their wording are kept short.
"""
from __future__ import annotations

import re

VALID_TYPES = ("score", "choice", "noul")
MAX_AI_QUESTIONS = 6
MAX_TEXT = 160  # characters per instruction or option

BUILTIN: dict[str, dict] = {
    "clipworthy": {
        "type": "score",
        "instructions": ("How likely a stranger scrolling past would stop and watch this "
                         "segment, judged on the segment itself rather than the surrounding "
                         "episode."),
        "criteria": [
            "Filler. Logistics, small talk, or a thought going nowhere.",
            "Mildly interesting to an existing fan, dull to anyone else.",
            "Solid. One clear idea or moment worth hearing.",
            "Strong. A stranger would finish it and might share it.",
            "Exceptional. A stranger would stop scrolling within a second.",
        ],
    },
    "hook_strength": {
        "type": "score",
        "instructions": ("How well the FIRST sentence of this segment works as the opening "
                         "line of a short video. Judge only the opening, not the whole segment."),
        "criteria": [
            "Dead open. Filler words, logistics, or mid-admin chatter.",
            "Slow. Understandable but gives no reason to keep watching.",
            "Adequate. States something concrete.",
            "Strong. A claim, question, or image that demands the next sentence.",
            "Arresting. Impossible to scroll past.",
        ],
    },
    "self_contained": {
        "type": "noul",
        "instructions": "This segment is understandable to someone who has heard nothing before it.",
        "criteria": {"true": "A new listener follows it completely.",
                     "false": "A new listener would be lost or confused."},
    },
    "ends_cleanly": {
        "type": "noul",
        "instructions": "The thought reaches a natural conclusion before the segment ends.",
        "criteria": {"true": "The point lands and completes.",
                     "false": "It cuts off mid-thought or trails away."},
    },
}
BUILTIN_WEIGHTS = {"clipworthy": 0.40, "hook_strength": 0.25,
                   "self_contained": 0.10, "ends_cleanly": 0.05}
AI_SHARE = 0.20


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "question"


def _short(text) -> str:
    return " ".join(str(text or "").split())[:MAX_TEXT]


def problems(questions: dict) -> list[str]:
    """What Laya would reject in a question dict; empty when all are usable."""
    out = []
    for qid, question in questions.items():
        qtype = question.get("type")
        crit = question.get("criteria")
        if qtype not in VALID_TYPES:
            out.append(f"{qid}: type {qtype!r} is not one of {', '.join(VALID_TYPES)}.")
        elif not question.get("instructions"):
            out.append(f"{qid}: has no instructions.")
        elif qtype == "score" and not (isinstance(crit, list) and 2 <= len(crit) <= 5):
            out.append(f"{qid}: a score needs 2-5 levels.")
        elif qtype == "choice" and not (isinstance(crit, dict) and 2 <= len(crit) <= 6):
            out.append(f"{qid}: a choice needs 2-6 options.")
        elif qtype == "noul" and not (isinstance(crit, dict) and set(crit) == {"true", "false"}):
            out.append(f"{qid}: a yes/no question needs exactly 'true' and 'false'.")
    return out


def _convert(item: dict) -> dict:
    qtype = item.get("type")
    question = {"type": qtype, "instructions": _short(item.get("instructions"))}
    if qtype == "score":
        question["criteria"] = [_short(level) for level in item.get("levels") or []]
    elif qtype == "choice":
        question["criteria"] = {_slug(c.get("key")): _short(c.get("description"))
                                for c in item.get("choices") or [] if isinstance(c, dict)}
    elif qtype == "noul":
        question["criteria"] = {"true": _short(item.get("true_means")) or "Yes, it does.",
                                "false": _short(item.get("false_means")) or "No, it does not."}
    return question


def from_ai(items: list[dict]) -> tuple[dict, list[str]]:
    """The AI's question list as Laya questions, plus what had to be dropped."""
    kept: dict[str, dict] = {}
    dropped: list[str] = []
    for item in items or []:
        if not isinstance(item, dict):
            dropped.append("A question was not an object.")
            continue
        qid = _slug(item.get("id"))
        if qid in BUILTIN:
            qid = f"ai_{qid}"
        base, n = qid, 2
        while qid in kept:
            qid, n = f"{base}_{n}", n + 1
        question = _convert(item)
        issues = problems({qid: question})
        if issues:
            dropped.extend(issues)
            continue
        if len(kept) == MAX_AI_QUESTIONS:
            dropped.append(f"{qid}: only six AI questions are asked; this one was left out.")
            continue
        kept[qid] = question
    return kept, dropped


def combined_weights(ai_weights: dict, ai_questions: dict) -> dict[str, float]:
    """Weights for the score: built-ins keep their shares and the AI's own
    score/yes-no questions split the rest. Choice answers describe a clip
    rather than grade it, so they carry no weight."""
    graded = [qid for qid, q in ai_questions.items() if q["type"] in ("score", "noul")]
    if not graded:
        total = sum(BUILTIN_WEIGHTS.values())
        return {qid: w / total for qid, w in BUILTIN_WEIGHTS.items()}
    raw = {qid: max(0.0, float(ai_weights.get(qid, 0) or 0)) for qid in graded}
    if sum(raw.values()) <= 0:
        raw = {qid: 1.0 for qid in graded}
    total = sum(raw.values())
    return {**BUILTIN_WEIGHTS, **{qid: AI_SHARE * w / total for qid, w in raw.items()}}
