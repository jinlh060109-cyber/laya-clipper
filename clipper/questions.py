"""The typed questions Laya answers for every candidate clip.

Two layers of questions. The fixed layer (FIXED) is asked about every clip;
the variable layer is up to six questions the AI writes for this video. Laya's three answer types are `score`
(pick one of k levels), `choice` (pick one option) and `noul` (yes/no).
A question and its options share 192 of Laya's 512 tokens, so levels,
options and their wording are kept short.

Yes/no questions are written as `noul` but sent to Laya as a two-option
`choice` with neutral keys, A = yes and B = no (`as_laya`). Laya's noul reads
its two options as `false:`/`true:`, and on the English checkpoint that label
pair outweighs the clip (laya issue #156): over 13 clips of one real run,
"ends cleanly" as a noul stayed between 0.39 and 0.54 (spread 0.04), while the
same question as an A/B choice ranged 0.35-0.80 (spread 0.13) and followed
the clips that really end mid-thought. The answer is still graded as the
probability of yes.
"""
from __future__ import annotations

import re

VALID_TYPES = ("score", "choice", "noul")
MAX_AI_QUESTIONS = 6
MAX_TEXT = 160  # characters per instruction or option

# The fixed layer: asked about every clip of every video, worded the same
# way each time. Laya reads each question against one of two states (`reads`,
# never sent to Laya): "clip" is the clip's whole text; "edges" is only its
# opening and closing sentence. Judged on the whole text, "does it start and
# end cleanly" were no better than chance on 18 hand-labelled clips of a real
# video (AUC 0.48 and 0.29: the rest of the text drowns the two sentences
# that decide it); on the edges alone the same checks reached 0.76 and 0.71.
FIXED: dict[str, dict] = {
    "clipworthy": {
        "reads": "clip",
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
        "reads": "edges",
        "type": "score",
        "instructions": "How well opening_sentence works as the first line of a short video.",
        "criteria": [
            "Dead open. Filler, logistics or chatter.",
            "Slow. Understandable, but no reason to keep watching.",
            "Adequate. States something concrete.",
            "Strong. A claim, question or image that demands the next sentence.",
            "Arresting. Impossible to scroll past.",
        ],
    },
    "has_start": {
        "reads": "edges",
        "type": "noul",
        "instructions": ("Does opening_sentence introduce its own topic, instead of continuing "
                         "something said before?"),
        "criteria": {"true": "It opens a topic: a question, a claim or a scene a newcomer can follow.",
                     "false": ("It continues: an answer, 'this', 'that means', 'with this', "
                               "'well' referring to earlier talk.")},
    },
    "has_end": {
        "reads": "edges",
        "type": "noul",
        "instructions": "Is closing_sentence a conclusion?",
        "criteria": {"true": "Yes: a final result, answer, number or joke.",
                     "false": "No: a step in the middle, a setup, or a new question."},
    },
}
FIXED_WEIGHTS = {"clipworthy": 0.30, "hook_strength": 0.20, "has_start": 0.15, "has_end": 0.15}
# The variable layer: the AI's own questions for this video share the rest.
AI_SHARE = 0.20
# Below this, the fixed layer flags a clip as starting or ending mid-thought.
EDGE_FLAG_BELOW = 0.40
YES, NO = "A", "B"


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
        if qid in FIXED:
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


def as_laya(question: dict) -> dict:
    """The question as Laya is asked it: a yes/no becomes an A/B choice."""
    if question.get("type") != "noul":
        return question
    crit = question.get("criteria") or {}
    return {"type": "choice", "instructions": question["instructions"],
            "criteria": {YES: f"Yes. {crit.get('true', 'It does.')}",
                         NO: f"No. {crit.get('false', 'It does not.')}"}}


def reads(question: dict) -> str:
    """Which state Laya reads for this question: "clip" or "edges"."""
    return question.get("reads", "clip")


def laya_questions(questions: dict) -> dict:
    """The questions as Laya is asked them (without the app's own fields)."""
    return {qid: as_laya({k: v for k, v in question.items() if k != "reads"})
            for qid, question in questions.items()}


def combined_weights(ai_weights: dict, ai_questions: dict) -> dict[str, float]:
    """Weights for the score: the fixed layer keeps its shares and the AI's
    own score/yes-no questions split the rest. Choice answers describe a clip
    rather than grade it, so they carry no weight."""
    graded = [qid for qid, q in ai_questions.items() if q["type"] in ("score", "noul")]
    if not graded:
        total = sum(FIXED_WEIGHTS.values())
        return {qid: w / total for qid, w in FIXED_WEIGHTS.items()}
    # A question renamed to avoid a fixed id (ai_...) keeps its weight.
    raw = {qid: max(0.0, float(ai_weights.get(qid, ai_weights.get(qid.removeprefix("ai_"), 0)) or 0))
           for qid in graded}
    if sum(raw.values()) <= 0:
        raw = {qid: 1.0 for qid in graded}
    total = sum(raw.values())
    return {**FIXED_WEIGHTS, **{qid: AI_SHARE * w / total for qid, w in raw.items()}}
