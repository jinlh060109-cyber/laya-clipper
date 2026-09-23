from pathlib import Path
import pytest
from clipper.cli import default_run_name, main

LAYA_MODEL = {"package": "0.3.5", "repo": "convaiinnovations/laya", "checkpoint": "root",
              "revision": "abc123", "device": "xpu", "dtype": "torch.bfloat16",
              "confidence_calibrated": True, "uncalibrated_buckets": []}
SUMMARY = {"scored": 4, "failed": 0, "candidates": 2, "uncertain": 1, "device": "xpu"}

def test_default_run_name_includes_date_and_stem():
    name = default_run_name(Path("/videos/Episode 47.mp4"))
    assert name.endswith("-episode-47")
    assert name[:4].isdigit()

def test_no_arguments_prints_usage_and_fails(capsys):
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()

def test_unknown_profile_exits_with_a_helpful_message(capsys, tmp_path):
    code = main(["score", str(tmp_path), "--profile", "vlog"])
    assert code == 1
    assert "podcast" in capsys.readouterr().err

def test_missing_artifact_names_the_producing_stage(capsys, tmp_path):
    from clipper.run import Run
    run = Run.create(tmp_path, "empty")
    run.write_json("source.json", {"duration": 60.0, "energy": [0.5] * 60})
    code = main(["window", str(run.root)])
    assert code == 1
    assert "transcribe" in capsys.readouterr().err

def test_plan_check_reports_warnings(capsys, tmp_path):
    from clipper.run import Run
    run = Run.create(tmp_path, "ep")
    run.write_json("source.json", {"duration": 5400.0,
                                   "video": {"width": 1920, "height": 1080}})
    run.write_json("plan.json", {"laya_model": LAYA_MODEL, "clips": [
        {"in": 10.0, "out": 15.0, "title": "T", "clip_format": "hot_take"}]})
    code = main(["plan", str(run.root), "--check"])
    assert code == 1
    assert "10" in capsys.readouterr().out

def test_plan_check_passes_a_clean_plan(capsys, tmp_path):
    from clipper.run import Run
    run = Run.create(tmp_path, "ep")
    run.write_json("source.json", {"duration": 5400.0,
                                   "video": {"width": 1920, "height": 1080}})
    run.write_json("plan.json", {"laya_model": LAYA_MODEL, "clips": [
        {"in": 100.0, "out": 120.0, "title": "T", "clip_format": "hot_take"}]})
    assert main(["plan", str(run.root), "--check"]) == 0

def test_score_passes_the_device_through_and_reports_counts(capsys, tmp_path, monkeypatch):
    seen = {}

    def fake_score(run, profile_name, device=None, agent=None):
        seen.update(profile=profile_name, device=device)
        return SUMMARY

    monkeypatch.setattr("clipper.cli.run_score", fake_score)
    (tmp_path / "ep").mkdir()
    assert main(["score", str(tmp_path / "ep"), "--profile", "podcast",
                 "--device", "xpu"]) == 0
    assert seen == {"profile": "podcast", "device": "xpu"}
    out = capsys.readouterr().out
    assert "2 candidates" in out and "1 uncertain" in out and "xpu" in out

def test_an_unloadable_model_is_reported_not_raised(capsys, tmp_path, monkeypatch):
    from clipper.agent import AgentError

    def fail(*a, **k):
        raise AgentError("Could not load Laya checkpoint")

    monkeypatch.setattr("clipper.cli.run_score", fail)
    (tmp_path / "ep").mkdir()
    assert main(["score", str(tmp_path / "ep"), "--profile", "podcast"]) == 1
    assert "Could not load" in capsys.readouterr().err

def test_all_builds_windows_before_scoring(tmp_path, monkeypatch):
    """Scoring reads windows.json; the pipeline must produce it, not assume it."""
    from clipper.run import Run
    run = Run.create(tmp_path, "ep")
    run.write_json("source.json", {"duration": 60.0, "energy": [0.5] * 60,
                                   "video": {"width": 1920, "height": 1080}})
    words = [{"word": f"w{i}", "start": float(i), "end": i + 0.5, "score": 0.9,
              "speaker": "SPEAKER_00"} for i in range(40)]
    run.write_json("transcript.json", {"language": "en", "segments": [
        {"start": 0.0, "end": 40.0, "speaker": "SPEAKER_00", "text": "x", "words": words}]})
    monkeypatch.setattr("clipper.cli._ingest_and_transcribe", lambda *a, **k: run)
    scored = {}

    def fake_score(r, profile_name, device=None, agent=None):
        scored["windows"] = r.read_json("windows.json")["windows"]
        return SUMMARY

    monkeypatch.setattr("clipper.cli.run_score", fake_score)
    assert main(["all", "video.mp4", "--profile", "podcast"]) == 0
    assert scored["windows"]


def test_web_serves_until_interrupted_and_prints_its_url(capsys, monkeypatch):
    opened = []

    class FakeServer:
        server_address = ("127.0.0.1", 9999)

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            opened.append("closed")

    monkeypatch.setattr("clipper.web.server.make_server",
                        lambda runs_dir, port=8765, runner=None: FakeServer())
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    assert main(["web", "--port", "9999"]) == 0
    assert "http://127.0.0.1:9999" in capsys.readouterr().out
    assert opened == ["http://127.0.0.1:9999", "closed"]


def test_web_no_browser_does_not_open_one(monkeypatch):
    opened = []

    class FakeServer:
        server_address = ("127.0.0.1", 8765)

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    monkeypatch.setattr("clipper.web.server.make_server",
                        lambda runs_dir, port=8765, runner=None: FakeServer())
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    assert main(["web", "--no-browser"]) == 0
    assert opened == []
