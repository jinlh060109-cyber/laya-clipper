from __future__ import annotations

import math

WINDOW_SECONDS = 20.0
STEP_SECONDS = 10.0
MIN_SPAN = 8.0
WEIGHTS = {"loudness": 0.45, "motion": 0.35, "cut_rate": 0.20}
THRESHOLD = 0.6
MAX_SECONDS = 60.0
TOP_N = 15
MIN_CANDIDATES = 5
# Mean raw frame difference below this is a menu, pause or loading screen.
STATIC_BELOW = 0.5
CUTS_FOR_FULL = 4


def _starts(start: float, end: float) -> list[float]:
    if end - start <= WINDOW_SECONDS:
        return [start]
    n = math.ceil((end - start - WINDOW_SECONDS) / STEP_SECONDS)
    return [start + k * STEP_SECONDS for k in range(n)] + [end - WINDOW_SECONDS]


def _seconds(start: float, end: float, n: int) -> list[int]:
    """Whole seconds inside [start, end), or the one under its midpoint."""
    lo, hi = max(0, math.ceil(start)), min(n, math.floor(end))
    if lo < hi:
        return list(range(lo, hi))
    return [min(n - 1, max(0, int((start + end) / 2)))] if n else []


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def action_windows(spans, energy, motion, motion_raw, cuts) -> list[dict]:
    """Score every window inside the silent spans from sound and picture alone."""
    windows: list[dict] = []
    for span_start, span_end in spans:
        if span_end - span_start < MIN_SPAN:
            continue
        for start in _starts(span_start, span_end):
            end = min(span_end, start + WINDOW_SECONDS)
            secs = _seconds(start, end, len(energy))
            if not secs:
                continue
            # Loudness is the peaks, not the average: an explosion lasts a second.
            loudest = sorted((energy[i] for i in secs), reverse=True)[:3]
            signals = {
                "loudness": round(_mean(loudest), 3),
                "motion": round(_mean(motion[i] for i in secs), 3),
                "cut_rate": round(min(1.0, sum(cuts[i] for i in secs) / CUTS_FOR_FULL), 3),
            }
            windows.append({
                "start": round(start, 3), "end": round(end, 3),
                "score": round(sum(WEIGHTS[k] * v for k, v in signals.items()), 4),
                "peak_time": max(secs, key=lambda i: energy[i]) + 0.5,
                "static": _mean(motion_raw[i] for i in secs) < STATIC_BELOW,
                "signals": signals,
            })
    return windows


def _candidate(group: list[dict]) -> dict:
    best = max(group, key=lambda w: w["score"])
    return {"start": group[0]["start"], "end": max(w["end"] for w in group),
            "peak_score": best["score"], "peak_time": best["peak_time"],
            "signals": best["signals"]}


def _overlaps(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def choose_action(windows: list[dict]) -> list[dict]:
    """Group hot windows into candidates; top up to a handful when few are hot."""
    live = [w for w in windows if not w["static"]]
    groups: list[list[dict]] = []
    for win in sorted((w for w in live if w["score"] >= THRESHOLD), key=lambda w: w["start"]):
        group = groups[-1] if groups else None
        if (group and win["start"] <= group[-1]["end"]
                and win["end"] - group[0]["start"] <= MAX_SECONDS):
            group.append(win)
        else:
            groups.append([win])
    chosen = sorted((_candidate(g) for g in groups),
                    key=lambda c: c["peak_score"], reverse=True)[:TOP_N]
    # A silent video with nothing "hot" still gives Claude something to look at.
    for win in sorted(live, key=lambda w: w["score"], reverse=True):
        if len(chosen) >= MIN_CANDIDATES:
            break
        if not any(_overlaps(win, c) for c in chosen):
            chosen.append(_candidate([win]))
    chosen.sort(key=lambda c: c["peak_score"], reverse=True)
    return [{"id": i, **c} for i, c in enumerate(chosen)]
