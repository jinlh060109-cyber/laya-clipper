from __future__ import annotations

from collections import Counter

from clipper.profiles.loader import Profile

SIGNAL_KEYS = ("buried_lede", "opens_with_windup", "ends_cleanly",
               "needs_speaker_id", "requires_visual", "poster_line",
               "audible_reaction", "self_contained")


def _group(ids: list[int], by_id: dict[int, dict]) -> list[list[int]]:
    """Group hot windows into runs that are contiguous in id AND in time.

    Id adjacency alone is not enough: windowing anchors on word starts and skips
    across silences, so consecutive windows can sit far apart in the source.
    """
    runs: list[list[int]] = []
    for wid in ids:
        prev = runs[-1][-1] if runs else None
        if (prev is not None and wid == prev + 1
                and by_id[wid]["start"] <= by_id[prev]["end"]):
            runs[-1].append(wid)
        else:
            runs.append([wid])
    return runs


def _build(run: list[int], by_id: dict[int, dict], ranked_by_id: dict[int, dict],
           profile: Profile, index: int) -> dict:
    max_seconds = profile.merge.get("max_seconds", 60.0)
    extend_by = profile.merge.get("backward_extend_seconds", 8.0)
    extend_above = profile.merge.get("backward_extend_offset", 0.6)

    curve = [ranked_by_id[i]["composite"] for i in run]
    peak_id = run[curve.index(max(curve))]
    start = by_id[run[0]]["start"]
    end = by_id[run[-1]]["end"]

    # Reactions lag their cause: if the peak window's energy spikes late, the
    # material that caused it sits before the window, not inside it.
    peak_window = by_id[peak_id]
    extended = peak_window["energy_peak_offset"] >= extend_above
    if extended:
        start = max(0.0, start - extend_by)

    if end - start > max_seconds:
        centre = (by_id[peak_id]["start"] + by_id[peak_id]["end"]) / 2
        start = max(start, centre - max_seconds / 2)
        end = min(end, start + max_seconds)

    formats = Counter(
        ranked_by_id[i]["answers"].get("clip_format", {}).get("value", "filler")
        for i in run
    )
    peak_answers = ranked_by_id[peak_id]["answers"]
    return {
        "id": index,
        "start": start,
        "end": end,
        "window_ids": run,
        "curve": curve,
        "peak_composite": max(curve),
        "mean_composite": sum(curve) / len(curve),
        "clip_format": formats.most_common(1)[0][0],
        # Signals are on the 0..1 scale like the composite; a raw Laya score is
        # 0..k-1. clip_format above keeps "value" because a choice has no
        # normalized form.
        "signals": {k: peak_answers[k].get("normalized")
                    for k in SIGNAL_KEYS if k in peak_answers},
        "backward_extended": extended,
    }


def merge_candidates(windows: list[dict], ranked: list[dict], profile: Profile) -> dict:
    threshold = profile.thresholds.get("candidate", 0.55)
    by_id = {w["id"]: w for w in windows}
    ranked_by_id = {r["id"]: r for r in ranked if not r.get("failed")}

    hot, unsure = [], []
    for wid, record in sorted(ranked_by_id.items()):
        if not record.get("gated") or record["composite"] < threshold:
            continue
        (unsure if record.get("uncertain") else hot).append(wid)

    candidates = [_build(run, by_id, ranked_by_id, profile, i)
                  for i, run in enumerate(_group(hot, by_id))]
    candidates.sort(key=lambda c: c["peak_composite"], reverse=True)
    for i, candidate in enumerate(candidates):
        candidate["id"] = i

    uncertain = [_build(run, by_id, ranked_by_id, profile, i)
                 for i, run in enumerate(_group(unsure, by_id))]
    return {"candidates": candidates, "uncertain": uncertain}
