"""Step 5: Laya answers the questions for every candidate clip.

Laya never writes text: per question it picks a level, an option or yes/no,
with a confidence. Two passes per clip, one per state (questions.reads):
the whole clip text, and only its opening and closing sentence (the fixed
hook/start/end checks). The clip's score is the weighted mean of the graded
answers on a 0..1 scale.
"""
from __future__ import annotations

import re
from typing import Callable

from clipper.agent import load_agent
from clipper.questions import EDGE_FLAG_BELOW, NO, YES, laya_questions, reads
from clipper.run import Run

# Laya computes confidence per answer type: score/choice as 1 - H(p)/log k
# (runs low by construction), noul as max(p, 1-p) (never below 0.5).
UNCERTAIN_BELOW = {"score": 0.10, "choice": 0.15, "noul": 0.55}


_SENTENCE = re.compile(r"(?<=[.?!。？！])\s+")


def edge_sentences(text: str) -> tuple[str, str]:
    """The clip's first and last sentence."""
    parts = [p for p in _SENTENCE.split(text.strip()) if p.strip()] or [text.strip()]
    return parts[0], parts[-1]


def clip_state(candidate: dict, content_type: str) -> dict:
    """What Laya reads for "clip" questions. Laya cuts overlong input from the
    right, so the clip text comes before anything that could be spared."""
    return {"content_type": content_type,
            "clip": {"text": candidate["text"]},
            "duration": round(float(candidate["duration"]), 1)}


def edges_state(candidate: dict, content_type: str) -> dict:
    """What Laya reads for "edges" questions: only where the clip starts and ends."""
    opening, closing = edge_sentences(candidate["text"])
    return {"content_type": content_type, "opening_sentence": opening,
            "closing_sentence": closing}


def normalize(answer: dict, question: dict) -> float | None:
    """0..1: a score's expected level over k-1 levels, a yes/no's probability."""
    qtype = question.get("type")
    if qtype == "score":
        levels = len(question.get("criteria") or [])
        if levels < 2:
            return None
        return min(1.0, max(0.0, float(answer.get("score", 0.0)) / (levels - 1)))
    if qtype == "noul":
        return min(1.0, max(0.0, float(answer.get("noul", 0.0))))
    return None


def _value(answer: dict):
    kind = answer.get("type")
    return answer.get("choice") if kind == "choice" else answer.get(
        "score" if kind == "score" else "noul")


def _probabilities(answer: dict) -> dict:
    """Laya's full distribution: per level (score), per option (choice), or
    false/true (noul, which Laya reports as one probability of yes)."""
    if answer.get("type") == "noul":
        yes = float(answer.get("noul", 0.0))
        return {"false": round(1.0 - yes, 4), "true": round(yes, 4)}
    return dict(answer.get("probabilities") or {})


def _unsure(answer: dict, question: dict) -> bool:
    conf = answer.get("confidence")
    floor = UNCERTAIN_BELOW.get(question.get("type"))
    return isinstance(conf, (int, float)) and floor is not None and conf < floor


def as_noul(answer: dict) -> dict:
    """A yes/no asked as an A/B choice (see questions.as_laya), read back as
    Laya reports a noul: the probability of yes, and max(p, 1-p) as the
    confidence."""
    probs = answer.get("probabilities") or {}
    yes = float(probs.get(YES, 0.0))
    total = yes + float(probs.get(NO, 0.0))
    yes = yes / total if total > 0 else 0.0
    return {"type": "noul", "noul": round(yes, 4),
            "confidence": round(max(yes, 1.0 - yes), 4)}


def rate_one(candidate: dict, content_type: str, questions: dict, weights: dict,
             agent) -> dict:
    try:
        raw = {}
        for state, which in ((clip_state, "clip"), (edges_state, "edges")):
            asked = {qid: q for qid, q in questions.items() if reads(q) == which}
            if not asked:
                continue
            response = agent.system_one(state(candidate, content_type), laya_questions(asked))
            raw.update({qid: a for qid, a in (response.get("answers") or {}).items()
                        if qid in asked})
        raw = {qid: as_noul(a) if questions.get(qid, {}).get("type") == "noul" else a
               for qid, a in raw.items()}
    except Exception as exc:  # noqa: BLE001 - one bad clip must not end the run
        return {"id": candidate["id"], "answers": {}, "flags": [], "score": 0.0, "confidence": 0.0,
                "uncertain": True, "failed": True, "error": f"{type(exc).__name__}: {exc}"}

    answers, total, weight_sum, confidences, uncertain = {}, 0.0, 0.0, [], False
    for qid, answer in raw.items():
        question = questions.get(qid, {})
        norm = normalize(answer, question)
        weight = weights.get(qid, 0.0)
        answers[qid] = {"type": answer.get("type"), "value": _value(answer),
                        "normalized": norm, "confidence": answer.get("confidence"),
                        "probabilities": _probabilities(answer),
                        "weight": round(weight, 4),
                        "uncertain": _unsure(answer, question)}
        if weight and norm is not None:
            total += weight * norm
            weight_sum += weight
            conf = answer.get("confidence")
            if isinstance(conf, (int, float)):
                confidences.append(float(conf))
                floor = UNCERTAIN_BELOW.get(question.get("type"))
                if floor is not None and conf < floor:
                    uncertain = True
    # The fixed layer's verdict on the edges, shown next to the clip.
    edges = {name: answers[name]["value"] < EDGE_FLAG_BELOW
             for name in ("has_start", "has_end") if name in answers}
    return {"id": candidate["id"], "answers": answers,
            "flags": [f"no {name.removeprefix('has_')}" for name, low in edges.items() if low],
            "score": total / weight_sum if weight_sum else 0.0,
            "confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "uncertain": uncertain, "failed": False, "error": None}


def run_rate(run: Run, device: str | None = None, agent: tuple | None = None,
             progress: Callable[[int, int], None] | None = None) -> dict:
    """Rate every candidate in segments.json; write scored.json.

    `agent` accepts a prepared `(agent, laya_model)` pair, which is how the
    web runner and tests supply one.
    """
    segments = run.read_json("segments.json")
    candidates = segments["candidates"]
    if not candidates:
        run.write_json("scored.json", {"laya_model": None, "candidates": []})
        return {"scored": 0, "failed": 0, "device": "none"}

    if agent is None:
        language = run.read_json("transcript.json").get("language")
        agent = load_agent(language, segments["questions"], device=device)
    model, meta = agent

    rated = []
    for candidate in candidates:
        rating = rate_one(candidate, segments.get("content_type", "unknown"),
                          segments["questions"], segments["weights"], model)
        rated.append({**candidate, **rating})
        if progress is not None:
            progress(len(rated), len(candidates))
    del model, agent  # see hardware.free_device_memory
    run.write_json("scored.json", {"laya_model": meta, "candidates": rated})
    failed = sum(1 for r in rated if r["failed"])
    return {"scored": len(rated) - failed, "failed": failed, "device": meta.get("device")}
