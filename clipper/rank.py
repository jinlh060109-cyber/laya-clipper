from __future__ import annotations

from clipper.profiles.loader import Profile


def _normalized(answers: dict, key: str) -> float:
    """The 0..1 value for one question, or 0.0 if absent or non-numeric.

    Reads `normalized`, never `value`: Laya's raw score is 0..k-1, and feeding
    that into weights calibrated for 0..1 would silently inflate every composite.
    """
    answer = answers.get(key)
    if not answer:
        return 0.0
    value = answer.get("normalized")
    if not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def composite(answers: dict, profile: Profile) -> float:
    total = sum(weight * _normalized(answers, key)
                for key, weight in profile.weights.items())
    total -= sum(weight * _normalized(answers, key)
                 for key, weight in profile.penalties.items())
    return min(1.0, max(0.0, total))


def passes_gate(answers: dict, profile: Profile) -> bool:
    """Fails closed: a missing gate answer disqualifies rather than passing."""
    return all(_normalized(answers, key) >= floor
               for key, floor in profile.gates.items())


def is_uncertain(answers: dict, profile: Profile) -> bool:
    """True when any question driving the composite is below its type's threshold.

    Per-primitive because Laya computes confidence two different ways:
    `score`/`choice` use 1 - H(p)/log k and run low by construction, while
    `noul` uses max(p, 1-p) and cannot go below 0.5. A single threshold marks
    either every window or no window uncertain.
    """
    for key in profile.weights:
        answer = answers.get(key)
        if not answer:
            continue
        floor = profile.uncertain_confidence.get(answer.get("type"))
        if floor is None:
            continue
        confidence = answer.get("confidence")
        if not isinstance(confidence, (int, float)):
            continue
        if float(confidence) < floor:
            return True
    return False


def rank(scores: list[dict], profile: Profile) -> list[dict]:
    out: list[dict] = []
    for record in scores:
        if record.get("failed"):
            out.append(record)
            continue
        answers = record["answers"]
        out.append({**record,
                    "composite": composite(answers, profile),
                    "gated": passes_gate(answers, profile),
                    "uncertain": is_uncertain(answers, profile)})
    return out
