"""Step 6: choose which clips to make, and let the user change the choice.

Rated clips are ranked by Laya's score (ties by confidence); the top N at or
above the minimum score start ticked. Moments without speech (the action
stage) are listed after them. selection.json is what the preview shows and
what "Make clips" works from.
"""
from __future__ import annotations

import re

from clipper.run import Run

MIN_CLIP = 10.0


def _first_sentence(text: str, limit: int = 80) -> str:
    match = re.match(r"(.+?[.?!])(\s|$)", text.strip())
    sentence = match.group(1) if match else text.strip()
    return sentence if len(sentence) <= limit else sentence[:limit - 1].rstrip() + "…"


def _clock(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def select(scored: list[dict], action: list[dict], top_n: int = 5,
           min_score: float = 0.30, action_n: int = 2) -> list[dict]:
    usable = [c for c in scored if not c.get("failed")]
    usable.sort(key=lambda c: (-c["score"], -c["confidence"]))
    rows = []
    for rank, cand in enumerate(usable):
        chunk = cand.get("category") == "chunk"
        rows.append({
            "id": f"c{cand['id']}", "start": cand["start"], "end": cand["end"],
            "duration": cand["duration"], "category": cand.get("category", "clip"),
            "score": cand["score"], "confidence": cand["confidence"],
            "uncertain": cand.get("uncertain", False),
            "title_hint": cand.get("hook_line") or _first_sentence(cand.get("text", "")),
            "reason": cand.get("reason", ""), "source": "chunks" if chunk else "ai",
            "include": rank < top_n and cand["score"] >= min_score, "fill_in": False,
        })

    moments = [a for a in action if a["end"] - a["start"] >= MIN_CLIP]
    moments.sort(key=lambda a: -a.get("peak_score", 0.0))
    for rank, moment in enumerate(moments):
        rows.append({
            "id": f"a{moment['id']}", "start": moment["start"], "end": moment["end"],
            "duration": round(moment["end"] - moment["start"], 3), "category": "action",
            "score": moment.get("peak_score", 0.0), "confidence": 0.0, "uncertain": False,
            "title_hint": f"Action at {_clock(moment['start'])}", "reason": "",
            "source": "action", "include": rank < action_n, "fill_in": False,
        })
    return rows


def run_select(run: Run, top_n: int = 5, min_score: float = 0.30,
               action_n: int = 2) -> dict:
    scored = run.read_json("scored.json")["candidates"]
    action = run.read_json("action.json")["candidates"] if run.exists("action.json") else []
    out = {"rule": {"top_n": top_n, "min_score": min_score, "action_n": action_n},
           "clips": select(scored, action, top_n, min_score, action_n)}
    run.write_json("selection.json", out)
    return out


def apply_choices(run: Run, choices: list[dict]) -> dict:
    """Tick or untick clips, and switch AI fill-in per clip, from the preview."""
    selection = run.read_json("selection.json")
    by_id = {clip["id"]: clip for clip in selection["clips"]}
    unknown = [str(c.get("id")) for c in choices if c.get("id") not in by_id]
    if unknown:
        raise ValueError(f"Unknown clip id(s): {', '.join(unknown)}.")
    for choice in choices:
        clip = by_id[choice["id"]]
        if "include" in choice:
            clip["include"] = bool(choice["include"])
        if "fill_in" in choice:
            clip["fill_in"] = bool(choice["fill_in"])
    run.write_json("selection.json", selection)
    return selection
