from __future__ import annotations

from clipper.agent import load_agent
from clipper.merge import merge_candidates
from clipper.profiles.loader import load_profile
from clipper.rank import rank
from clipper.run import Run
from clipper.score import score_windows


def run_score(run: Run, profile_name: str, device: str | None = None,
              agent=None) -> dict:
    """Score every window and write scores.json plus candidates.json.

    Synchronous by design: Laya runs locally and is compute-bound, so there is
    no latency to overlap. The agent is loaded once and reused across windows.

    `agent` accepts a prepared `(agent, laya_model)` pair, which is how tests
    run the whole stage without loading a model.
    """
    profile = load_profile(profile_name)
    windows = run.read_json("windows.json")["windows"]
    language = (run.read_json("transcript.json") or {}).get("language")

    if agent is None:
        agent_obj, meta = load_agent(language, profile, device=device)
    else:
        agent_obj, meta = agent

    records = score_windows(windows, profile, agent_obj)
    ranked = rank(records, profile)
    merged = merge_candidates(windows, ranked, profile)

    run.write_json("scores.json", {"profile": profile.name,
                                   "laya_model": meta,
                                   "windows": ranked})
    run.write_json("candidates.json", {"profile": profile.name,
                                       "laya_model": meta,
                                       "candidates": merged["candidates"],
                                       "uncertain": merged["uncertain"]})

    failed = sum(1 for r in ranked if r.get("failed"))
    return {"scored": len(ranked) - failed,
            "failed": failed,
            "candidates": len(merged["candidates"]),
            "uncertain": len(merged["uncertain"]),
            "device": meta["device"]}
