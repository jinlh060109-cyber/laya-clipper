import pytest

from clipper.agent import (AgentError, checkpoint_for_language, load_agent,
                           uncalibrated_buckets, used_buckets)
from clipper.profiles.loader import load_profile


class FakeLoaded:
    """Stands in for a loaded laya.Agent."""

    def __init__(self, device="xpu", dtype="torch.bfloat16", temps=None):
        self.device = device
        self.dtype = dtype
        self.temperature_by_options_raw = temps if temps is not None else {"choice:11+": 0.1006}

    def system_one(self, state, questions):
        return {"model": "laya-rl-agent", "answers": {}, "usage": {}}


def fake_loader(recorder, **overrides):
    def _load(repo, device=None, subfolder=None, **kw):
        recorder.append({"repo": repo, "device": device, "subfolder": subfolder})
        return FakeLoaded(**overrides)
    return _load


def test_english_selects_the_repo_root():
    assert checkpoint_for_language("en") is None


def test_non_english_selects_multilingual():
    assert checkpoint_for_language("hi") == "multilingual"
    assert checkpoint_for_language("de") == "multilingual"


def test_missing_language_selects_multilingual_defensively():
    assert checkpoint_for_language(None) == "multilingual"


def test_language_region_suffix_still_counts_as_english():
    assert checkpoint_for_language("en-US") is None


def test_load_agent_passes_device_explicitly():
    """Laya's own probe cannot see Arc, so the device must be passed, not inferred."""
    calls = []
    _, meta = load_agent("en", load_profile("core"), device="xpu",
                         loader=fake_loader(calls))
    assert calls[0]["device"] == "xpu"
    assert calls[0]["subfolder"] is None
    assert meta["device"] == "xpu"


def test_load_agent_selects_multilingual_subfolder_for_non_english():
    calls = []
    _, meta = load_agent("fr", load_profile("core"), device="cpu",
                         loader=fake_loader(calls))
    assert calls[0]["subfolder"] == "multilingual"
    assert meta["checkpoint"] == "multilingual"


def test_provenance_records_repo_package_and_dtype():
    _, meta = load_agent("en", load_profile("core"), device="xpu",
                         loader=fake_loader([]))
    assert meta["repo"] == "convaiinnovations/laya"
    assert meta["checkpoint"] == "root"
    assert meta["dtype"] == "torch.bfloat16"
    assert meta["package"]


def test_silent_cpu_fallback_is_detected_and_warned():
    """Laya mutates agent.device and continues on placement failure."""
    loader = fake_loader([], device="cpu")
    with pytest.warns(RuntimeWarning, match="fell back"):
        _, meta = load_agent("en", load_profile("core"), device="xpu", loader=loader)
    assert meta["device"] == "cpu"
    assert meta["device_requested"] == "xpu"


def test_no_warning_when_the_device_is_what_was_asked_for(recwarn):
    load_agent("en", load_profile("core"), device="xpu", loader=fake_loader([]))
    assert not [w for w in recwarn if "fell back" in str(w.message)]


def test_used_buckets_reflects_the_profiles_actual_questions():
    buckets = used_buckets(load_profile("core"))
    assert "score:3-5" in buckets       # 5-level scores
    assert "choice:6-10" in buckets     # hook_type (6), clip_format (8)
    assert "noul:2" in buckets
    assert "choice:11+" not in buckets  # nothing has 11+ options


def test_the_clamped_bucket_is_not_reachable_so_calibration_holds():
    """choice:11+ ships clamped, but no question reaches it."""
    agent = FakeLoaded(temps={"choice:11+": 0.1006})
    assert uncalibrated_buckets(agent, load_profile("core")) == []


def test_a_clamped_bucket_that_is_reachable_is_reported():
    agent = FakeLoaded(temps={"choice:6-10": 0.1006})
    assert uncalibrated_buckets(agent, load_profile("core")) == ["choice:6-10"]


def test_in_range_temperatures_are_not_flagged():
    agent = FakeLoaded(temps={"choice:6-10": 1.2, "noul:2": 0.9})
    assert uncalibrated_buckets(agent, load_profile("core")) == []


def test_calibration_flag_lands_in_provenance():
    _, meta = load_agent("en", load_profile("core"), device="cpu", loader=fake_loader([]))
    assert meta["confidence_calibrated"] is True
    assert meta["uncalibrated_buckets"] == []


def test_a_loader_failure_is_reported_with_the_repo_named():
    def boom(repo, **kw):
        raise OSError("no such file")
    with pytest.raises(AgentError, match="convaiinnovations/laya"):
        load_agent("en", load_profile("core"), device="cpu", loader=boom)


@pytest.mark.model
def test_real_agent_loads_and_scores_one_window():
    from clipper.score import build_questions, score_windows

    profile = load_profile("core")
    agent, meta = load_agent("en", profile)
    assert meta["checkpoint"] == "root"
    assert meta["revision"] != "unknown"

    window = {"id": 0, "start": 0.0, "end": 30.0, "position": 0.1,
              "energy_mean": 0.4, "energy_peak": 0.8, "energy_peak_offset": 0.5,
              "preceding": "Earlier in the episode.",
              "text": "SPEAKER_01: The first version is supposed to embarrass you."}
    records = score_windows([window], profile, agent)
    assert records[0]["failed"] is False
    answers = records[0]["answers"]
    assert set(answers) == set(profile.questions)
    assert 0.0 <= answers["clipworthy"]["normalized"] <= 1.0
    assert 0.0 <= answers["open_loop"]["normalized"] <= 1.0
    assert answers["hook_type"]["value"] in profile.questions["hook_type"]["criteria"]
