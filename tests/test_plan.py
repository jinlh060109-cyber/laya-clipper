import pytest
from clipper.plan import LENGTH_TARGETS, validate_plan

def _plan(**clip):
    base = {"in": 100.0, "out": 130.0, "title": "T", "hook": "H",
            "description": "D", "clip_format": "story_arc"}
    return {"profile": "podcast",
            "laya_model": {"package": "0.3.5", "repo": "convaiinnovations/laya",
                           "checkpoint": "root", "revision": "abc123",
                           "device": "xpu", "dtype": "torch.bfloat16",
                           "confidence_calibrated": True, "uncalibrated_buckets": []},
            "clips": [{**base, **clip}]}

def test_length_targets_cover_every_clip_format():
    for fmt in ("cold_open", "debate", "how_to", "story_arc",
                "hot_take", "confession", "list"):
        assert fmt in LENGTH_TARGETS

def test_valid_plan_produces_no_warnings():
    assert validate_plan(_plan(), duration=5400.0) == []

def test_clip_under_ten_seconds_is_flagged():
    warnings = validate_plan(_plan(out=108.0), duration=5400.0)
    assert any("10" in w for w in warnings)

def test_clip_over_sixty_seconds_is_flagged_not_rejected():
    """Spec: flag for a second look; an exceptional segment can carry it."""
    warnings = validate_plan(_plan(out=200.0), duration=5400.0)
    assert any("60" in w for w in warnings)

def test_clip_outside_its_format_target_is_flagged():
    warnings = validate_plan(_plan(clip_format="hot_take", out=145.0), duration=5400.0)
    assert any("hot_take" in w for w in warnings)

def test_missing_title_is_flagged():
    warnings = validate_plan(_plan(title=""), duration=5400.0)
    assert any("title" in w.lower() for w in warnings)

def test_out_point_past_source_end_is_flagged():
    warnings = validate_plan(_plan(out=6000.0), duration=5400.0)
    assert any("source" in w.lower() for w in warnings)

def test_overlapping_clips_are_flagged():
    plan = _plan()
    plan["clips"].append({**plan["clips"][0], "in": 120.0, "out": 150.0})
    assert any("overlap" in w.lower() for w in validate_plan(plan, 5400.0))

def test_speaker_label_without_a_speakers_map_is_flagged():
    plan = _plan(speaker_label=True)
    assert any("speakers" in w.lower() for w in validate_plan(plan, 5400.0))

def test_missing_laya_model_is_flagged():
    plan = _plan()
    del plan["laya_model"]
    assert any("laya_model" in w for w in validate_plan(plan, 5400.0))

def test_uncalibrated_confidence_is_flagged_with_its_buckets():
    plan = _plan()
    plan["laya_model"].update(confidence_calibrated=False,
                              uncalibrated_buckets=["score:5", "noul"])
    warnings = validate_plan(plan, 5400.0)
    assert any("uncalibrated" in w and "score:5, noul" in w for w in warnings)
