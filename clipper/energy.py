from __future__ import annotations

import statistics


def rolling_baseline(rms: list[float], baseline_seconds: int = 300) -> list[float]:
    """Normalize per-second RMS against a local window, as a clamped z-score.

    Returns values in 0..1 where 0.5 is "typical for this part of the file".
    """
    if not rms:
        return []
    half = max(1, baseline_seconds // 2)
    out: list[float] = []
    for i in range(len(rms)):
        lo = max(0, i - half)
        hi = min(len(rms), i + half + 1)
        local = rms[lo:hi]
        mean = statistics.fmean(local)
        stdev = statistics.pstdev(local) if len(local) > 1 else 0.0
        if stdev < 1e-9:
            out.append(0.5)
            continue
        z = (rms[i] - mean) / stdev
        out.append(min(1.0, max(0.0, 0.5 + z / 6.0)))
    return out
