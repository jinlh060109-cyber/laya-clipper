import pytest

from clipper import pipeline
from clipper.run import Run


@pytest.fixture
def run(tmp_path):
    return Run.create(tmp_path, "ep")


def test_a_misconfigured_ai_stops_the_run_with_the_reason(run, monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.delenv("AI_MODEL", raising=False)
    with pytest.raises(ValueError, match="AI_MODEL"):
        pipeline.default_steps().segment(run, {"prompt": ""})


def test_fill_in_only_runs_for_ticked_clips_that_asked_for_it(run, monkeypatch):
    import clipper.fillin
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    run.write_json("transcript.json", {"segments": []})
    run.write_json("selection.json", {"clips": [
        {"id": "c0", "include": True, "fill_in": True},
        {"id": "c1", "include": True, "fill_in": False},
        {"id": "c2", "include": False, "fill_in": True}]})
    monkeypatch.setattr(clipper.fillin, "fill_in",
                        lambda config, clip, transcript, notes: {"title": clip["id"]})
    fills = pipeline.default_steps().fillin(run, {"notes": ""}, None)
    assert fills == {"c0": {"title": "c0"}}
    assert run.read_json("fills.json") == fills


def test_fill_in_without_an_ai_says_how_to_set_one_up(run, monkeypatch):
    for key in ("AI_PROVIDER", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    run.write_json("selection.json", {"clips": [{"id": "c0", "include": True, "fill_in": True}]})
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        pipeline.default_steps().fillin(run, {"notes": ""}, None)


def test_rating_hands_the_gpu_memory_back(run, monkeypatch):
    import clipper.hardware
    import clipper.rate
    freed = []
    monkeypatch.setattr("clipper.device.default_probes",
                        lambda: {"cuda": lambda: False, "xpu": lambda: True, "mps": lambda: False})
    monkeypatch.setattr(clipper.rate, "run_rate", lambda r, device, progress: {"device": device})
    monkeypatch.setattr(clipper.hardware, "free_device_memory", freed.append)
    pipeline.default_steps().rate(run, {"device": "xpu"}, None)
    assert freed == ["xpu"]
