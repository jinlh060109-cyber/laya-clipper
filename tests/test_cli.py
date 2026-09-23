from pathlib import Path

import pytest

from clipper.cli import default_run_name, main
from clipper.pipeline import Steps
from clipper.run import Run

ACTION = {"silent_seconds": 600.0, "spans": 3, "candidates": 7}


def test_default_run_name_includes_date_and_stem():
    name = default_run_name(Path("/videos/Episode 47.mp4"))
    assert name.endswith("-episode-47")
    assert name[:4].isdigit()


def test_default_run_name_keeps_non_latin_names_apart():
    a = default_run_name(Path("访谈第一集.mp4"))
    assert a != default_run_name(Path("第二集.mp4")) and a.endswith("访谈第一集")


def test_no_arguments_prints_usage_and_fails(capsys):
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()


def _steps(log, **override):
    def step(name, value=None):
        def fn(*args):
            log.append((name, args))
            return value
        return fn
    steps = Steps(
        ingest=step("ingest"), transcribe=step("transcribe"),
        segment=step("segment", {"content_type": "tutorial", "ai": None, "ai_error": None,
                                 "candidates": [1, 2]}),
        rate=step("rate", {"scored": 2, "failed": 0, "device": "xpu"}),
        action=step("action", ACTION),
        select=step("select", {"clips": [
            {"id": "c0", "include": True, "start": 1.0, "end": 21.0, "score": 0.6,
             "category": "tip", "title_hint": "A tip."}]}),
        style=step("style", {"layout": "fit"}),
        fillin=step("fillin", {}),
        edit=step("edit", [{"id": "c0", "folder": "01-a-tip", "title": "A tip.",
                            "duration": 20.0, "final": "clips/01-a-tip/final.mp4"}]),
    )
    for name, fn in override.items():
        setattr(steps, name, fn)
    return steps


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr("clipper.cli.RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr("clipper.cli.check_same_source", lambda run, video: None)
    return tmp_path / "runs"


def test_analyze_runs_every_step_and_reports(runs, tmp_path, monkeypatch, capsys):
    log = []
    monkeypatch.setattr("clipper.cli.default_steps", lambda: _steps(log))
    video = tmp_path / "ep.mp4"
    video.write_bytes(b"x")
    assert main(["analyze", str(video), "--run", "ep", "--device", "xpu",
                 "--prompt", "tips", "--top", "3"]) == 0
    assert [n for n, _ in log] == ["ingest", "transcribe", "segment", "rate", "action", "select"]
    settings = dict(log)["segment"][1]
    assert settings == {"device": "xpu", "model": "large-v3", "diarize": True,
                        "prompt": "tips", "top_n": 3}
    out = capsys.readouterr().out
    assert "2 clips proposed without AI" in out and "c0" in out and "clipper make" in out


def test_analyze_reuses_what_the_run_already_has(runs, tmp_path, monkeypatch):
    log = []
    monkeypatch.setattr("clipper.cli.default_steps", lambda: _steps(log))
    run = Run.create(runs, "ep")
    run.write_json("source.json", {"duration": 60.0})
    run.write_json("transcript.json", {"segments": []})
    video = tmp_path / "ep.mp4"
    video.write_bytes(b"x")
    assert main(["analyze", str(video), "--run", "ep"]) == 0
    assert [n for n, _ in log][:1] == ["segment"]


def test_make_applies_the_choices_then_edits(runs, monkeypatch, capsys):
    log = []
    monkeypatch.setattr("clipper.cli.default_steps", lambda: _steps(log))
    run = Run.create(runs, "ep")
    run.write_json("selection.json", {"clips": [
        {"id": "c0", "include": True, "fill_in": False},
        {"id": "c1", "include": False, "fill_in": False}]})
    assert main(["make", str(run.root), "--only", "c1", "--fill-in", "--layout", "crop"]) == 0
    clips = {c["id"]: c for c in run.read_json("selection.json")["clips"]}
    assert (clips["c0"]["include"], clips["c1"]["include"], clips["c1"]["fill_in"]) == (False, True, True)
    assert [n for n, _ in log] == ["style", "fillin", "edit"]
    assert dict(log)["style"][1] == {"style": {"layout": "crop"}}
    out = capsys.readouterr().out
    assert "01-a-tip" in out and "final.mp4" in out


def test_make_with_an_unknown_clip_id_fails_cleanly(runs, monkeypatch, capsys):
    monkeypatch.setattr("clipper.cli.default_steps", lambda: _steps([]))
    run = Run.create(runs, "ep")
    run.write_json("selection.json", {"clips": [{"id": "c0", "include": True}]})
    assert main(["make", str(run.root), "--only", "c7"]) == 1
    assert "c7" in capsys.readouterr().err


def test_hardware_lists_devices_and_encoders(monkeypatch, capsys):
    monkeypatch.setattr("clipper.cli.detect", lambda: {
        "devices": [{"id": "xpu", "label": "Intel GPU (XPU)", "available": True,
                     "detail": "Arc 140T", "hint": ""},
                    {"id": "cuda", "label": "NVIDIA GPU (CUDA)", "available": False,
                     "detail": "Not found", "hint": "Install CUDA torch"}],
        "encoders": [{"id": "qsv", "label": "Intel Quick Sync", "available": True}],
        "rocm": False})
    assert main(["hardware"]) == 0
    out = capsys.readouterr().out
    assert "Intel GPU (XPU)" in out and "Arc 140T" in out and "Install CUDA torch" in out
    assert "Intel Quick Sync" in out


def test_action_prints_what_it_found(tmp_path, monkeypatch, capsys):
    run = Run.create(tmp_path, "g")
    monkeypatch.setattr("clipper.cli.run_action", lambda r, progress=None: ACTION)
    assert main(["action", str(run.root)]) == 0
    assert "7 action moments from 10.0 min without speech" in capsys.readouterr().out


def test_transcribe_takes_a_device(tmp_path, monkeypatch):
    run = Run.create(tmp_path, "ep")
    seen = []
    monkeypatch.setattr("clipper.cli.transcribe",
                        lambda wav, r, model, device=None, hf_token=None:
                        seen.append(device) or {"segments": [], "diarized": False})
    assert main(["transcribe", str(run.root), "--device", "xpu", "--no-diarize"]) == 0
    assert seen == ["xpu"]


def test_analyze_refuses_a_run_that_holds_another_video(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr("clipper.cli.RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr("clipper.cli.default_steps", lambda: _steps([]))
    old, new = tmp_path / "old.mp4", tmp_path / "new.mp4"
    old.write_bytes(b"old")
    new.write_bytes(b"new!")
    run = Run.create(tmp_path / "runs", "ep")
    run.write_json("source.json", {"path": old.resolve().as_posix(), "size": 3})
    assert main(["analyze", str(new), "--run", "ep"]) == 1
    assert "--run" in capsys.readouterr().err


def _fake_server(opened, port):
    class FakeServer:
        server_address = ("127.0.0.1", port)

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            opened.append("closed")
    return FakeServer()


def test_web_serves_until_interrupted_and_prints_its_url(capsys, monkeypatch):
    opened = []
    monkeypatch.setattr("clipper.web.server.make_server",
                        lambda runs_dir, port=8765, runner=None: _fake_server(opened, 9999))
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    assert main(["web", "--port", "9999"]) == 0
    assert "http://127.0.0.1:9999" in capsys.readouterr().out
    assert opened == ["http://127.0.0.1:9999", "closed"]


def test_web_no_browser_does_not_open_one(monkeypatch):
    opened = []
    monkeypatch.setattr("clipper.web.server.make_server",
                        lambda runs_dir, port=8765, runner=None: _fake_server(opened, 8765))
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    assert main(["web", "--no-browser"]) == 0
    assert opened == ["closed"]
