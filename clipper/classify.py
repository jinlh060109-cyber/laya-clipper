from __future__ import annotations

from collections import Counter

from clipper.score import window_state

# Fixed order: the criteria order Laya sees, and the tie-break.
PROFILE_ORDER: tuple[str, ...] = ("podcast", "talking_head", "lecture", "stream")
DESCRIPTIONS = {
    "podcast": "Two or more people in conversation or an interview.",
    "talking_head": "One person speaking directly to the camera.",
    "lecture": "A talk, class or presentation teaching a subject.",
    "stream": "A live stream: gaming, just chatting, or reacting to content.",
}
QUESTION_ID = "content_type"
FALLBACK = "podcast"


def sample_windows(windows: list[dict], n: int = 12) -> list[dict]:
    """Up to `n` windows evenly spaced across the source, centred in each stride."""
    if n <= 0 or not windows:
        return []
    if len(windows) <= n:
        return list(windows)
    stride = len(windows) / n
    return [windows[int(i * stride + stride / 2)] for i in range(n)]


def _question(profiles: list[str]) -> dict:
    return {QUESTION_ID: {
        "type": "choice",
        "instructions": "What kind of video is this segment from?",
        "criteria": {name: DESCRIPTIONS[name] for name in profiles},
    }}


def choose_profile(windows: list[dict], agent, profiles=PROFILE_ORDER,
                   n: int = 12) -> dict:
    """Majority vote over sampled windows. A failed or unknown answer casts no vote."""
    ordered = [name for name in PROFILE_ORDER if name in profiles]
    questions = _question(ordered)
    sampled = sample_windows(windows, n)
    votes: Counter[str] = Counter()
    for window in sampled:
        try:
            response = agent.system_one(window_state(window, "auto"), questions)
            choice = response["answers"][QUESTION_ID]["choice"]
        except Exception:  # noqa: BLE001 - one bad window costs one vote, not the run
            continue
        if choice in ordered:
            votes[choice] += 1

    result = {"votes": {name: votes.get(name, 0) for name in ordered},
              "sampled": len(sampled)}
    if not votes:
        return {**result, "profile": FALLBACK, "fallback": True}
    best = max(votes.values())
    winner = next(name for name in ordered if votes.get(name, 0) == best)
    return {**result, "profile": winner, "fallback": False}
