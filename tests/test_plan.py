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


from clipper.plan import plan_errors  # noqa: E402


@pytest.mark.parametrize("clip, needle", [
    ({"in": None}, "in"),
    ({"out": "soon"}, "out"),
    ({"in": -3.0}, "negative"),
    ({"out": 90.0}, "not after"),
    ({"in": 6000.0, "out": 6030.0}, "beyond"),
    ({"out": 105.0}, "10s"),
    ({"crop_x": "left"}, "crop_x"),
    ({"captions": "yes"}, "captions"),
])
def test_hard_problems_are_errors_not_crashes(clip, needle):
    plan = _plan(**clip)
    if clip.get("in", 0) is None:
        del plan["clips"][0]["in"]
    errors = plan_errors(plan, duration=5400.0)
    assert any(needle in e for e in errors), errors
    validate_plan(plan, duration=5400.0)  # must not raise either


def test_a_plan_without_clips_is_an_error():
    assert plan_errors({"laya_model": {}}, 5400.0)
    assert plan_errors({"clips": "none"}, 5400.0)
    assert plan_errors({"clips": ["x"]}, 5400.0)


def test_numeric_crop_x_and_valid_captions_are_accepted():
    assert plan_errors(_plan(crop_x=300, captions="sidecar"), 5400.0) == []


def test_soft_warnings_are_not_errors():
    plan = _plan(clip_format="hot_take", out=145.0)
    assert plan_errors(plan, 5400.0) == []
    assert validate_plan(plan, 5400.0)


def test_an_explicit_null_laya_model_is_not_warned_about():
    plan = {"laya_model": None, "clips": [{"in": 0, "out": 20, "title": "t"}]}
    assert not any("laya_model" in p for p in validate_plan(plan, 100.0))


def test_action_clips_have_a_15_to_30s_target():
    plan = {"laya_model": None,
            "clips": [{"in": 0, "out": 40, "title": "t", "clip_format": "action"}]}
    assert any("action target" in p for p in validate_plan(plan, 100.0))
