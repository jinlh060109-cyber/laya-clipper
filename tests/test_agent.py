import pytest

from clipper.agent import (AgentError, checkpoint_for_language, load_agent,
                           uncalibrated_buckets, used_buckets)
from clipper.questions import FIXED as BUILTIN

QUESTIONS = {**BUILTIN, "kind": {"type": "choice", "instructions": "What kind?",
             "criteria": {k: k for k in ("tip", "joke", "story", "rant", "reveal", "other")}}}


@pytest.fixture(autouse=True)
def every_device_available(request, monkeypatch):
    """Device resolution is device.py's concern; here it must not depend on the host.

    Without this, requesting "xpu" resolves to CPU on any machine whose torch
    build lacks XPU, and the loader never sees the device the test asked for.
    Model-marked tests are exempt: they must resolve against the real hardware.
    """
    if request.node.get_closest_marker("model"):
        return
    monkeypatch.setattr("clipper.device.default_probes",
                        lambda: {"cuda": lambda: True, "xpu": lambda: True,
                                 "mps": lambda: True})


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


def test_checkpoint_edge_cases():
    """en/fr wiring is covered through load_agent below; these are the edges."""
    assert checkpoint_for_language("en-US") is None
    assert checkpoint_for_language(None) == "multilingual"


def test_load_agent_passes_device_explicitly():
    """Laya's own probe cannot see Arc, so the device must be passed, not inferred."""
    calls = []
    _, meta = load_agent("en", QUESTIONS, device="xpu",
                         loader=fake_loader(calls))
    assert calls[0]["device"] == "xpu"
    assert calls[0]["subfolder"] is None
    assert meta["device"] == "xpu"


def test_load_agent_selects_multilingual_subfolder_for_non_english():
    calls = []
    _, meta = load_agent("fr", QUESTIONS, device="cpu",
                         loader=fake_loader(calls, device="cpu"))
    assert calls[0]["subfolder"] == "multilingual"
    assert meta["checkpoint"] == "multilingual"


def test_silent_cpu_fallback_is_detected_and_warned():
    """Laya mutates agent.device and continues on placement failure."""
    loader = fake_loader([], device="cpu")
    with pytest.warns(RuntimeWarning, match="fell back"):
        _, meta = load_agent("en", QUESTIONS, device="xpu", loader=loader)
    assert meta["device"] == "cpu"
    assert meta["device_requested"] == "xpu"


def test_no_warning_when_the_device_is_what_was_asked_for(recwarn):
    load_agent("en", QUESTIONS, device="xpu", loader=fake_loader([]))
    assert not [w for w in recwarn if "fell back" in str(w.message)]


def test_used_buckets_reflects_the_actual_questions():
    buckets = used_buckets(QUESTIONS)
    assert "score:3-5" in buckets       # 5-level scores
    assert "choice:6-10" in buckets     # kind (6)
    assert "choice:2" in buckets        # yes/no questions are asked as A/B choices
    assert "noul:2" not in buckets
    assert "choice:11+" not in buckets  # nothing has 11+ options


def test_the_clamped_bucket_is_not_reachable_so_calibration_holds():
    """choice:11+ ships clamped, but no question reaches it."""
    agent = FakeLoaded(temps={"choice:11+": 0.1006})
    assert uncalibrated_buckets(agent, QUESTIONS) == []


def test_a_clamped_bucket_that_is_reachable_is_reported():
    agent = FakeLoaded(temps={"choice:6-10": 0.1006})
    assert uncalibrated_buckets(agent, QUESTIONS) == ["choice:6-10"]


def test_in_range_temperatures_are_not_flagged():
    agent = FakeLoaded(temps={"choice:6-10": 1.2, "noul:2": 0.9})
    assert uncalibrated_buckets(agent, QUESTIONS) == []


def test_a_loader_failure_is_reported_with_the_repo_named():
    def boom(repo, **kw):
        raise OSError("no such file")
    with pytest.raises(AgentError, match="convaiinnovations/laya"):
        load_agent("en", QUESTIONS, device="cpu", loader=boom)


@pytest.mark.model
def test_real_agent_loads_and_rates_one_clip():
    from clipper.questions import combined_weights
    from clipper.rate import rate_one

    agent, meta = load_agent("en", QUESTIONS)
    assert meta["device"] == meta["device_requested"], "Laya silently fell back"
    assert meta["checkpoint"] == "root"
    assert meta["revision"] != "unknown"

    clip = {"id": 0, "start": 0.0, "end": 30.0, "duration": 30.0,
            "text": "The first version is supposed to embarrass you. If it doesn't, you shipped too late."}
    rated = rate_one(clip, "talking_head", QUESTIONS, combined_weights({}, {}), agent)
    assert rated["failed"] is False
    assert set(rated["answers"]) == set(QUESTIONS)
    assert 0.0 <= rated["score"] <= 1.0
    assert rated["answers"]["kind"]["value"] in QUESTIONS["kind"]["criteria"]


def _snapshot(cache, repo, name, files, age):
    import os as _os
    snap = cache / f"models--{repo.replace('/', '--')}" / "snapshots" / name
    snap.mkdir(parents=True)
    for f in files:
        (snap / f).write_text("x")
    _os.utime(snap, (age, age))
    return snap


def test_the_newest_complete_cached_snapshot_is_found(tmp_path):
    from clipper.agent import cached_snapshot
    _snapshot(tmp_path, "org/laya", "old", ["rl_agent_config.json", "model.safetensors"], 1000)
    good = _snapshot(tmp_path, "org/laya", "mid", ["rl_agent_config.json", "model.safetensors"], 2000)
    _snapshot(tmp_path, "org/laya", "broken", ["model.safetensors"], 3000)  # half-downloaded
    assert cached_snapshot("org/laya", cache=tmp_path) == good
    assert cached_snapshot("org/other", cache=tmp_path) is None


def test_a_failed_download_falls_back_to_the_cached_copy(tmp_path, monkeypatch):
    from clipper import agent as agent_mod
    good = _snapshot(tmp_path, "org/laya", "mid", ["rl_agent_config.json", "model.safetensors"], 2000)
    monkeypatch.setattr(agent_mod, "cached_snapshot", lambda repo, subfolder=None: good)
    calls = []

    def loader(repo, device, subfolder):
        calls.append(repo)
        if repo == "org/laya":
            raise OSError("[WinError 1314] symlink privilege")
        return FakeLoaded(device=device)

    with pytest.warns(RuntimeWarning, match="cached copy"):
        agent_mod.load_agent("en", QUESTIONS, device="cpu", repo="org/laya", loader=loader)
    assert calls == ["org/laya", str(good)]
