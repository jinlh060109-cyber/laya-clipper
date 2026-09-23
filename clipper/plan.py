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


CAPTION_MODES = ("burn", "sidecar", "none")


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _clip_errors(clip, label: str, duration: float) -> list[str]:
    if not isinstance(clip, dict):
        return [f"{label}: is not an object."]
    errors: list[str] = []
    start, end = _number(clip.get("in")), _number(clip.get("out"))
    if start is None:
        errors.append(f"{label}: 'in' is missing or not a number.")
    if end is None:
        errors.append(f"{label}: 'out' is missing or not a number.")
    if start is not None and end is not None:
        if start < 0:
            errors.append(f"{label}: in-point {start:.1f}s is negative.")
        if start >= duration:
            errors.append(f"{label}: in-point {start:.1f}s is beyond the source end ({duration:.1f}s).")
        elif end <= start:
            errors.append(f"{label}: out-point {end:.1f}s is not after its in-point {start:.1f}s.")
        elif min(end, duration) - start < HARD_MIN:
            errors.append(f"{label}: {min(end, duration) - start:.1f}s is under the "
                          f"{HARD_MIN:.0f}s minimum and will be rejected.")
    crop_x = clip.get("crop_x", "center")
    if crop_x != "center" and (isinstance(crop_x, bool) or _number(crop_x) is None):
        errors.append(f"{label}: crop_x must be \"center\" or a pixel offset, not {crop_x!r}.")
    if clip.get("captions", "burn") not in CAPTION_MODES:
        errors.append(f"{label}: captions must be one of {', '.join(CAPTION_MODES)}, "
                      f"not {clip.get('captions')!r}.")
    return errors


def plan_errors(plan: dict, duration: float) -> list[str]:
    """Problems that would make render fail or cut the wrong thing."""
    clips = plan.get("clips") if isinstance(plan, dict) else None
    if not isinstance(clips, list) or not clips:
        return ["plan.json needs a non-empty \"clips\" list."]
    errors: list[str] = []
    for index, clip in enumerate(clips, start=1):
        errors.extend(_clip_errors(clip, f"clip {index}", duration))
    return errors


def validate_plan(plan: dict, duration: float) -> list[str]:
    """Every problem: the hard errors from `plan_errors`, then softer warnings."""
    errors = plan_errors(plan, duration)
    warnings: list[str] = []
    # An explicit null is a video with no speech: Laya never ran, on purpose.
    if "laya_model" not in plan:
        warnings.append(
            "plan.json has no laya_model; thresholds cannot be traced to a checkpoint."
        )
    elif plan["laya_model"] and not plan["laya_model"].get("confidence_calibrated", True):
        buckets = ", ".join(plan["laya_model"].get("uncalibrated_buckets") or [])
        warnings.append(
            f"laya_model reports uncalibrated confidence buckets ({buckets}); "
            f"the uncertain bucket may be unreliable."
        )

    clips = plan.get("clips") if isinstance(plan.get("clips"), list) else []
    spans: list[tuple[float, float, int]] = []
    for index, clip in enumerate(clips, start=1):
        label = f"clip {index}"
        if _clip_errors(clip, label, duration):
            continue  # already reported as an error
        start, end = float(clip["in"]), float(clip["out"])
        length = min(end, duration) - start

        if end > duration:
            warnings.append(f"{label}: out-point {end:.1f}s is past the source end "
                            f"({duration:.1f}s); it will be cut there.")
        if length > SOFT_MAX:
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
    return errors + warnings


def check_plan(run: Run) -> tuple[list[str], list[str]]:
    """(errors, warnings) for the run's plan.json."""
    plan = run.read_json("plan.json")
    duration = run.read_json("source.json")["duration"]
    errors = plan_errors(plan, duration)
    return errors, [w for w in validate_plan(plan, duration) if w not in errors]
