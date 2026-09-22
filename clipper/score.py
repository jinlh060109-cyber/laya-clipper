from __future__ import annotations

import copy

from clipper.profiles.loader import Profile


def build_questions(profile: Profile) -> dict[str, dict]:
    """Profile YAML is already Laya's schema, so this is a pass-through copy.

    Kept as a named seam so a future profile feature has one place to land.
    """
    return {qid: copy.deepcopy(q) for qid, q in profile.questions.items()}


def window_state(window: dict, profile_name: str) -> dict:
    """Build the text-only state Laya scores.

    Field order is load-bearing: Laya truncates state from the RIGHT at roughly
    395 tokens, so the window being judged leads and `preceding` trails, ensuring
    any overflow sheds context rather than the material itself.
    """
    return {
        "profile": profile_name,
        "window": {
            "start": window["start"],
            "end": window["end"],
            "text": window["text"],
        },
        "position": window.get("position"),
        "energy_mean": window.get("energy_mean"),
        "energy_peak": window.get("energy_peak"),
        "energy_peak_offset": window.get("energy_peak_offset"),
        "preceding": window.get("preceding", ""),
    }


def normalize_answer(answer: dict, qdef: dict) -> float | None:
    """Rebase one Laya answer onto 0..1.

    `score` returns an expected value over level INDICES (0..k-1), not Jev's
    2..10, so it divides by k-1. `noul` is already a probability. `choice` is
    categorical and never weighted, so it has no normalized form.
    """
    qtype = qdef.get("type")
    if qtype == "score":
        levels = len(qdef.get("criteria") or [])
        if levels < 2:
            return None
        raw = float(answer.get("score", 0.0)) / (levels - 1)
        return min(1.0, max(0.0, raw))
    if qtype == "noul":
        return min(1.0, max(0.0, float(answer.get("noul", 0.0))))
    return None


def answers_to_dict(response: dict, profile: Profile) -> dict[str, dict]:
    """Flatten a `system_one` response, keeping raw values alongside normalized ones."""
    out: dict[str, dict] = {}
    for qid, answer in (response.get("answers") or {}).items():
        qdef = profile.questions.get(qid, {})
        qtype = answer.get("type")
        if qtype == "choice":
            value = answer.get("choice")
        elif qtype == "score":
            value = answer.get("score")
        else:
            value = answer.get("noul")
        out[qid] = {
            "type": qtype,
            "value": value,
            "normalized": normalize_answer(answer, qdef),
            "confidence": answer.get("confidence"),
            "probabilities": answer.get("probabilities", {}),
            "legend": answer.get("legend"),
        }
    return out


def score_windows(windows: list[dict], profile: Profile, agent) -> list[dict]:
    """Score every window with one `system_one` call each.

    `agent` is anything exposing `system_one(state, questions)`. Scoring is
    synchronous: Laya is local and compute-bound, so there is nothing to overlap.
    A window whose call raises is marked `failed` and the run continues — losing
    3 of 540 windows is not a reason to discard an eight-minute transcription.
    """
    questions = build_questions(profile)
    records: list[dict] = []
    for window in windows:
        state = window_state(window, profile.name)
        try:
            response = agent.system_one(state, questions)
            answers = answers_to_dict(response, profile)
        except Exception as exc:  # noqa: BLE001 - any inference failure is per-window
            records.append({"id": window["id"], "answers": {}, "failed": True,
                            "error": f"{type(exc).__name__}: {exc}"})
            continue
        records.append({"id": window["id"], "answers": answers, "failed": False})
    return records
