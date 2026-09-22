from __future__ import annotations

from clipper.run import Run

# Spec section 8, from retention data by format.
LENGTH_TARGETS: dict[str, tuple[float, float]] = {
    "hot_take": (15.0, 25.0),
    "cold_open": (15.0, 25.0),
    "confession": (20.0, 35.0),
    "debate": (20.0, 35.0),
    "how_to": (25.0, 40.0),
    "list": (25.0, 40.0),
    "story_arc": (30.0, 45.0),
}
HARD_MIN = 10.0
SOFT_MAX = 60.0


def validate_plan(plan: dict, duration: float) -> list[str]:
    warnings: list[str] = []
    if not plan.get("laya_model"):
        warnings.append(
            "plan.json has no laya_model; thresholds cannot be traced to a checkpoint."
        )
    elif not plan["laya_model"].get("confidence_calibrated", True):
        buckets = ", ".join(plan["laya_model"].get("uncalibrated_buckets") or [])
        warnings.append(
            f"laya_model reports uncalibrated confidence buckets ({buckets}); "
            f"the uncertain bucket may be unreliable."
        )

    spans: list[tuple[float, float, int]] = []
    for index, clip in enumerate(plan.get("clips", []), start=1):
        start, end = float(clip["in"]), float(clip["out"])
        length = end - start
        label = f"clip {index}"

        if end > duration:
            warnings.append(f"{label}: out-point {end:.1f}s is past the source end ({duration:.1f}s).")
        if length < HARD_MIN:
            warnings.append(f"{label}: {length:.1f}s is under the {HARD_MIN:.0f}s minimum and will be rejected.")
        elif length > SOFT_MAX:
            warnings.append(f"{label}: {length:.1f}s exceeds {SOFT_MAX:.0f}s; worth a second look.")
        else:
            fmt = clip.get("clip_format")
            target = LENGTH_TARGETS.get(fmt)
            if target and not target[0] <= length <= target[1]:
                warnings.append(
                    f"{label}: {length:.1f}s is outside the {fmt} target "
                    f"of {target[0]:.0f}-{target[1]:.0f}s."
                )
        if not clip.get("title"):
            warnings.append(f"{label}: missing a title.")
        if clip.get("speaker_label") and not plan.get("speakers"):
            warnings.append(f"{label}: speaker_label is set but plan.json has no speakers map.")
        spans.append((start, end, index))

    for a_start, a_end, a in sorted(spans):
        for b_start, b_end, b in sorted(spans):
            if a < b and a_start < b_end and b_start < a_end:
                warnings.append(f"clips {a} and {b} overlap.")
    return warnings


def check_plan(run: Run) -> list[str]:
    return validate_plan(run.read_json("plan.json"), run.read_json("source.json")["duration"])
