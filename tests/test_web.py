import json
import os
import socket
import struct
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import quote

import pytest

from clipper.pipeline import Steps
from clipper.web.jobs import JobRunner
from clipper.web.server import make_server

HARDWARE = {"devices": [{"id": "auto", "label": "Automatic", "available": True, "detail": "", "hint": ""},
                        {"id": "cuda", "label": "NVIDIA GPU (CUDA)", "available": False,
                         "detail": "Not found", "hint": "install"},
                        {"id": "xpu", "label": "Intel GPU (XPU)", "available": True,
                         "detail": "Arc", "hint": ""},
                        {"id": "mps", "label": "Apple GPU (Metal)", "available": False,
                         "detail": "Not found", "hint": "mac"},
                        {"id": "cpu", "label": "CPU", "available": True, "detail": "", "hint": ""}],
            "encoders": [{"id": "auto", "label": "Automatic", "available": True, "detail": "x264"}],
            "rocm": False}


def gated_steps(gate):
    def ok(value=None):
        def fn(*args):
            gate.wait(5)
            return value
        return fn

    def select(run, settings):
        gate.wait(5)
        chosen = {"rule": {}, "clips": [{"id": "c0", "start": 1.0, "end": 21.0, "include": True,
                                         "fill_in": False, "title_hint": "A tip."}]}
        run.write_json("selection.json", chosen)
        return chosen

    return Steps(ingest=ok(), transcribe=ok(),
                 segment=ok({"content_type": "tutorial", "ai": None, "ai_error": None,
                             "candidates": [1]}),
                 rate=ok({"scored": 1, "failed": 0, "device": "cpu"}),
                 action=ok({"silent_seconds": 0.0, "spans": 0, "candidates": 0}),
                 select=select, style=ok({"layout": "fit"}), fillin=ok({}),
                 edit=ok([{"id": "c0", "final": "clips/01-a-tip/final.mp4"}]))


@pytest.fixture(autouse=True)
def restore_environment():
    """Saving AI settings writes os.environ; put it back after each test."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPPER_STYLE", str(tmp_path / "style.json"))
    for key in ("AI_PROVIDER", "AI_MODEL", "AI_API_KEY", "AI_BASE_URL", "ANTHROPIC_API_KEY",
                "ALIBABA_TOKEN_PLAN_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    gate = threading.Event()
    gate.set()
    runner = JobRunner(gated_steps(gate))
    srv = make_server(tmp_path, port=0, runner=runner, hardware=lambda: HARDWARE,
                      env_path=tmp_path / ".env")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", tmp_path, runner, gate
    gate.set()
    runner.wait(5)
    srv.shutdown()
    srv.server_close()


def call(method, url, data=None, headers=None):
    request = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def jcall(method, url, data=None):
    status, body = call(method, url, data)
    return status, json.loads(body)


def upload(base, name="My Episode.mp4", data=b"\x00\x01" * 1000):
    return jcall("PUT", f"{base}/api/upload?name={quote(name)}", data)


def analyze(base, **body):
    return jcall("POST", f"{base}/api/analyze", json.dumps(body).encode())


def make(base, **body):
    return jcall("POST", f"{base}/api/make", json.dumps(body).encode())


def _analyzed(base, runner):
    _, uploaded = upload(base)
    assert analyze(base, run=uploaded["run"])[0] == 202
    runner.wait(5)
    return uploaded["run"]


def test_a_misconfigured_ai_is_reported_in_the_config(server, monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.delenv("AI_MODEL", raising=False)
    base, *_ = server
    _, config = jcall("GET", base + "/api/config")
    assert "AI_MODEL" in config["ai"]["error"]


def test_an_upload_that_is_not_a_video_is_refused(server):
    base, *_ = server
    status, body = upload(base, name="notes.txt")
    assert status == 400 and ".mp4" in body["error"]


def test_directories_in_the_upload_name_are_ignored(server):
    base, tmp_path, *_ = server
    status, body = upload(base, name="..\\..\\evil.mp4")
    assert status == 200 and body["run"].endswith("-evil")


def test_the_preview_of_an_unknown_run_is_404(server):
    base, *_ = server
    assert call("GET", f"{base}/api/selection?run=nope")[0] == 404


@pytest.mark.parametrize("field, value", [("device", "tpu"), ("model", "huge"), ("top_n", 0)])
def test_bad_analyze_settings_are_refused(server, field, value):
    base, *_ = server
    _, uploaded = upload(base)
    status, body = analyze(base, run=uploaded["run"], **{field: value})
    assert status == 400 and field.split("_")[0] in body["error"].lower()


@pytest.mark.parametrize("name", ["nope", "../outside", ""])
def test_an_unknown_run_is_refused(server, name):
    base, *_ = server
    assert analyze(base, run=name)[0] == 400


def test_a_run_without_a_video_is_refused(server):
    base, tmp_path, *_ = server
    (tmp_path / "empty").mkdir()
    status, body = analyze(base, run="empty")
    assert status == 400 and "upload" in body["error"].lower()


def test_make_with_an_unknown_clip_is_refused(server):
    base, _, runner, _ = server
    run = _analyzed(base, runner)
    status, body = make(base, run=run, choices=[{"id": "c9", "include": True}], style={})
    assert status == 400 and "c9" in body["error"]


def test_make_before_analyze_is_refused(server):
    base, *_ = server
    _, uploaded = upload(base)
    status, body = make(base, run=uploaded["run"], choices=[], style={})
    assert status == 400 and "Analyze" in body["error"]


def test_a_bad_style_is_refused(server):
    base, *_ = server
    status, body = jcall("PUT", f"{base}/api/style", json.dumps({"layout": "zoom"}).encode())
    assert status == 400 and "layout" in body["error"]


def test_style_is_saved_and_returned(server):
    base, *_ = server
    status, saved = jcall("PUT", f"{base}/api/style",
                          json.dumps({"caption_case": "upper"}).encode())
    assert status == 200 and saved["caption_case"] == "upper"
    assert jcall("GET", base + "/api/config")[1]["style"]["caption_case"] == "upper"


def test_a_second_job_and_an_upload_while_running_are_refused(server):
    base, _, runner, gate = server
    _, uploaded = upload(base)
    gate.clear()
    assert analyze(base, run=uploaded["run"])[0] == 202
    assert analyze(base, run=uploaded["run"])[0] == 409
    assert upload(base, name="other.mp4")[0] == 409
    gate.set()


def test_media_serves_run_files_with_byte_ranges(server):
    base, tmp_path, runner, _ = server
    run = _analyzed(base, runner)
    clip = tmp_path / run / "clips" / "01-a-tip"
    clip.mkdir(parents=True)
    (clip / "final.mp4").write_bytes(bytes(range(256)) * 4)
    status, body = call("GET", f"{base}/media/{run}/clips/01-a-tip/final.mp4",
                        headers={"Range": "bytes=10-19"})
    assert status == 206 and body == bytes(range(10, 20))
    status, body = call("GET", f"{base}/media/{run}/clips/01-a-tip/final.mp4")
    assert status == 200 and len(body) == 1024


@pytest.mark.parametrize("path", ["/media/../secret.txt", "/media/x/..%2F..%2Fsecret.txt",
                                  "/media/nope/video.mp4"])
def test_media_never_leaves_a_run_folder(server, path):
    base, tmp_path, *_ = server
    (tmp_path.parent / "secret.txt").write_text("no")
    assert call("GET", base + path)[0] == 404


def test_a_refused_large_upload_still_gets_its_error_message(server):
    """The server must read (discard) a refused body, or the client sees a reset, not the 400."""
    base, *_ = server
    for _ in range(3):
        status, body = upload(base, name="notes.txt", data=b"\x00" * (8 * 1024 * 1024))
        assert status == 400 and ".mp4" in body["error"]


def test_an_upload_while_busy_gets_409_not_a_reset(server):
    base, _, runner, gate = server
    _, uploaded = upload(base)
    gate.clear()
    assert analyze(base, run=uploaded["run"])[0] == 202
    status, body = upload(base, name="other.mp4", data=b"\x00" * (8 * 1024 * 1024))
    assert status == 409 and "running" in body["error"]
    gate.set()


def raw(base, request: bytes, timeout=5.0) -> bytes:
    """Send bytes on a raw socket and return whatever comes back before it closes."""
    host, port = base.removeprefix("http://").split(":")
    with socket.create_connection((host, int(port)), timeout=timeout) as sock:
        sock.sendall(request)
        chunks = []
        while True:
            try:
                data = sock.recv(65536)
            except (TimeoutError, ConnectionError):
                break
            if not data:
                break
            chunks.append(data)
        return b"".join(chunks)


@pytest.mark.parametrize("payload", [b"[1, 2]", b'"podcast"', b"null"])
def test_a_body_that_is_not_an_object_is_refused(server, payload):
    base, *_ = server
    status, body = jcall("POST", f"{base}/api/analyze", payload)
    assert status == 400 and "object" in body["error"]


def test_a_negative_content_length_is_answered_not_left_hanging(server):
    base, *_ = server
    started = time.monotonic()
    reply = raw(base, b"POST /api/analyze HTTP/1.1\r\nHost: x\r\nContent-Length: -1\r\n\r\n")
    assert reply.startswith(b"HTTP/1.0 400") or reply.startswith(b"HTTP/1.1 400")
    assert time.monotonic() - started < 4


def test_a_stalled_upload_is_dropped_quietly_and_cleaned_up(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("clipper.web.server.REQUEST_TIMEOUT", 0.5)
    srv = make_server(tmp_path, port=0, runner=JobRunner(gated_steps(threading.Event())),
                      hardware=lambda: HARDWARE)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        raw(base, b"PUT /api/upload?name=a.mp4 HTTP/1.1\r\nHost: x\r\n"
                  b"Content-Length: 1000\r\n\r\n" + b"\x00" * 10, timeout=5)
        deadline = time.monotonic() + 5
        while list(tmp_path.glob("*/video.*")) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert list(tmp_path.glob("*/video.*")) == []
    finally:
        srv.shutdown()
        srv.server_close()
    assert "Traceback" not in capsys.readouterr().err


def test_a_client_that_hangs_up_mid_upload_leaves_no_traceback(server, capsys):
    base, tmp_path, *_ = server
    host, port = base.removeprefix("http://").split(":")

    def wait_until(condition):
        deadline = time.monotonic() + 5
        while not condition() and time.monotonic() < deadline:
            time.sleep(0.02)
        return condition()

    for _ in range(3):
        with socket.create_connection((host, int(port)), timeout=5) as sock:
            sock.sendall(b"PUT /api/upload?name=a.mp4 HTTP/1.1\r\nHost: x\r\n"
                         b"Content-Length: 100000000\r\n\r\n" + b"\x00" * 100000)
            # Hang up only once the server is writing, so the check below
            # cannot pass merely because it ran before the file existed.
            assert wait_until(lambda: list(tmp_path.glob("*/video.*")))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        assert wait_until(lambda: not list(tmp_path.glob("*/video.*")))
    time.sleep(0.2)
    assert "Traceback" not in capsys.readouterr().err


def test_a_refused_make_leaves_the_saved_style_alone(server):
    base, tmp_path, runner, _ = server
    run = _analyzed(base, runner)
    make(base, run=run, choices=[{"id": "c9", "include": True}], style={"notes": "Changed."})
    assert not (tmp_path / "style.json").exists()


def put_json(base, path, body, headers=None):
    return call("PUT", base + path, json.dumps(body).encode(),
                {"Content-Type": "application/json", **(headers or {})})


def test_choosing_an_ai_provider_saves_it_and_never_echoes_the_key(server):
    base, tmp_path, *_ = server
    status, body = put_json(base, "/api/ai", {"provider": "alibaba_token_plan",
                                              "model": "qwen3.7-plus", "api_key": "sk-sp-secret"})
    assert status == 200 and b"sk-sp-secret" not in body
    saved = json.loads(body)
    assert saved["active"] == {"provider": "alibaba_token_plan", "model": "qwen3.7-plus"}
    env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "ALIBABA_TOKEN_PLAN_API_KEY=sk-sp-secret" in env and "AI_PROVIDER=alibaba_token_plan" in env
    config = jcall("GET", base + "/api/config")[1]
    assert b"sk-sp-secret" not in json.dumps(config).encode()
    token_plan = next(p for p in config["ai"]["providers"] if p["id"] == "alibaba_token_plan")
    assert token_plan["has_key"] and token_plan["active"]


def test_switching_provider_keeps_the_other_providers_key(server):
    base, tmp_path, *_ = server
    put_json(base, "/api/ai", {"provider": "anthropic", "api_key": "sk-ant-1"})
    put_json(base, "/api/ai", {"provider": "alibaba_token_plan", "api_key": "sk-sp-2"})
    status, body = put_json(base, "/api/ai", {"provider": "anthropic"})
    assert status == 200 and json.loads(body)["active"]["provider"] == "anthropic"
    env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY=sk-ant-1" in env and "ALIBABA_TOKEN_PLAN_API_KEY=sk-sp-2" in env


def test_a_provider_without_a_key_is_refused(server):
    base, *_ = server
    status, body = put_json(base, "/api/ai", {"provider": "alibaba_token_plan"})
    assert status == 400 and b"ALIBABA_TOKEN_PLAN_API_KEY" in body


def test_a_failing_ai_connection_says_why(server, monkeypatch):
    import clipper.ai
    base, *_ = server
    put_json(base, "/api/ai", {"provider": "anthropic", "api_key": "sk-ant-1"})

    def fail(*a, **k):
        raise clipper.ai.AIError("Claude rejected the API key.")

    monkeypatch.setattr(clipper.ai, "complete_json", fail)
    status, body = call("POST", base + "/api/ai/test", b"{}", {"Content-Type": "application/json"})
    assert status == 400 and b"rejected the API key" in body


@pytest.mark.parametrize("method, path", [("PUT", "/api/ai"), ("POST", "/api/analyze"),
                                          ("POST", "/api/make"), ("PUT", "/api/style")])
def test_other_websites_cannot_change_settings_or_start_jobs(server, method, path):
    """A page on any site could otherwise post to this local server, e.g. to
    point the AI at its own address and receive the user's key and transcripts."""
    base, *_ = server
    status, _ = call(method, base + path, json.dumps({"provider": "custom"}).encode(),
                     {"Content-Type": "application/json", "Origin": "http://evil.example"})
    assert status == 403


def test_the_caption_preview_is_a_jpeg_of_the_chosen_style(server, monkeypatch):
    import clipper.edit
    base, _, runner, _ = server
    run = _analyzed(base, runner)
    seen = {}

    def fake(run_, clip_id, style, ffmpeg=None):
        seen.update(clip=clip_id, style=style)
        return b"\xff\xd8fake"

    monkeypatch.setattr(clipper.edit, "preview_frame", fake)
    status, body = call("GET", f"{base}/api/preview?run={run}&clip=c0&caption_style=neon"
                               f"&layout=crop&caption_case=upper")
    assert status == 200 and body == b"\xff\xd8fake"
    assert seen["clip"] == "c0" and seen["style"]["caption_style"] == "neon"
    assert seen["style"]["layout"] == "crop" and seen["style"]["caption_case"] == "upper"


def test_a_preview_with_a_bad_style_is_refused(server):
    base, _, runner, _ = server
    run = _analyzed(base, runner)
    assert call("GET", f"{base}/api/preview?run={run}&clip=c0&caption_style=wobbly")[0] == 400
