import json
import os
import socket
import struct
import time
import threading
import urllib.error
import urllib.request
from urllib.parse import quote

import pytest

from clipper.web.jobs import JobRunner, Stages
from clipper.web.server import make_server


def gated_stages(gate):
    def ok(value=None):
        def fn(*args):
            gate.wait(5)
            return value
        return fn
    return Stages(ingest=ok(), transcribe=ok(), windows=ok(),
                  action=ok({"silent_seconds": 0.0, "spans": 0, "candidates": 0}),
                  load_agent=ok(("A", {})),
                  choose_profile=ok({"profile": "podcast", "votes": {"podcast": 1},
                                     "sampled": 1, "fallback": False}),
                  score=ok({"scored": 1, "failed": 0, "candidates": 1,
                            "uncertain": 0, "device": "cpu"}))


@pytest.fixture
def server(tmp_path):
    gate = threading.Event()
    gate.set()
    runner = JobRunner(gated_stages(gate))
    srv = make_server(tmp_path, port=0, runner=runner)
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


def start(base, **body):
    return jcall("POST", f"{base}/api/start", json.dumps(body).encode())


def test_the_page_is_served(server):
    base, *_ = server
    status, body = call("GET", base + "/")
    assert status == 200
    assert b"<title>" in body


def test_config_lists_the_choices(server, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    base, *_ = server
    status, config = jcall("GET", base + "/api/config")
    assert status == 200
    assert [p["id"] for p in config["profiles"]][0] == "auto"
    assert "gaming" in next(p["label"] for p in config["profiles"] if p["id"] == "stream")
    assert config["devices"] == ["auto", "xpu", "cpu"]
    assert config["models"][0] == "large-v3"
    assert config["hf_token"] is False


def test_an_upload_lands_on_disk_intact(server):
    base, tmp_path, *_ = server
    payload = os.urandom(3 * 1024 * 1024 + 17)
    status, body = upload(base, data=payload)
    assert status == 200
    assert body["run"].endswith("-my-episode")
    assert (tmp_path / body["run"] / "video.mp4").read_bytes() == payload


def test_an_upload_that_is_not_a_video_is_refused(server):
    base, *_ = server
    status, body = upload(base, name="notes.txt")
    assert status == 400
    assert ".mp4" in body["error"]


def test_directories_in_the_upload_name_are_ignored(server):
    base, tmp_path, *_ = server
    status, body = upload(base, name="..\\..\\evil.mp4")
    assert status == 200
    assert body["run"].endswith("-evil")
    assert (tmp_path / body["run"]).is_dir()


def test_start_runs_the_job_to_completion(server):
    base, _, runner, _ = server
    _, uploaded = upload(base)
    status, body = start(base, run=uploaded["run"], profile="podcast")
    assert status == 202
    assert body["state"] == "running"
    runner.wait(5)
    _, final = jcall("GET", base + "/api/status")
    assert final["state"] == "done"
    assert final["result"]["candidates"] == 1


def test_status_is_idle_before_any_job(server):
    base, *_ = server
    assert jcall("GET", base + "/api/status") == (200, {"state": "idle"})


def test_an_unknown_profile_is_refused(server):
    base, *_ = server
    _, uploaded = upload(base)
    status, body = start(base, run=uploaded["run"], profile="vlog")
    assert status == 400
    assert "profile" in body["error"].lower()


@pytest.mark.parametrize("name", ["nope", "../outside", ""])
def test_an_unknown_run_is_refused(server, name):
    base, *_ = server
    status, _ = start(base, run=name, profile="podcast")
    assert status == 400


def test_a_run_without_a_video_is_refused(server):
    base, tmp_path, *_ = server
    (tmp_path / "empty").mkdir()
    status, body = start(base, run="empty", profile="podcast")
    assert status == 400
    assert "upload" in body["error"].lower()


def test_a_second_start_and_an_upload_while_running_are_refused(server):
    base, _, runner, gate = server
    _, uploaded = upload(base)
    gate.clear()
    assert start(base, run=uploaded["run"], profile="podcast")[0] == 202
    assert start(base, run=uploaded["run"], profile="podcast")[0] == 409
    assert upload(base, name="other.mp4")[0] == 409
    gate.set()


def test_unknown_routes_are_404(server):
    base, *_ = server
    assert call("GET", base + "/nope")[0] == 404


def test_the_page_calls_every_api_route_and_names_every_stage(server):
    """A cheap guard that the page and the API have not drifted apart."""
    from clipper.web.jobs import STAGES
    base, *_ = server
    _, body = call("GET", base + "/")
    page = body.decode("utf-8")
    for route in ("/api/config", "/api/upload", "/api/start", "/api/status"):
        assert route in page
    for stage in STAGES:
        assert f'"{stage}"' in page


def test_a_refused_large_upload_still_gets_its_error_message(server):
    """The server must read (discard) a refused body, or the client sees a reset, not the 400."""
    base, *_ = server
    for _ in range(3):
        status, body = upload(base, name="notes.txt", data=b"\x00" * (8 * 1024 * 1024))
        assert status == 400
        assert ".mp4" in body["error"]


def test_an_upload_while_busy_gets_409_not_a_reset(server):
    base, _, runner, gate = server
    _, uploaded = upload(base)
    gate.clear()
    assert start(base, run=uploaded["run"], profile="podcast")[0] == 202
    status, body = upload(base, name="other.mp4", data=b"\x00" * (8 * 1024 * 1024))
    assert status == 409
    assert "running" in body["error"]
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
def test_a_start_body_that_is_not_an_object_is_refused(server, payload):
    base, *_ = server
    status, body = jcall("POST", f"{base}/api/start", payload)
    assert status == 400
    assert "object" in body["error"]


def test_a_negative_content_length_is_answered_not_left_hanging(server):
    base, *_ = server
    started = time.monotonic()
    reply = raw(base, b"POST /api/start HTTP/1.1\r\nHost: x\r\nContent-Length: -1\r\n\r\n")
    assert reply.startswith(b"HTTP/1.0 400") or reply.startswith(b"HTTP/1.1 400")
    assert time.monotonic() - started < 4


def test_a_stalled_upload_is_dropped_quietly_and_cleaned_up(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("clipper.web.server.REQUEST_TIMEOUT", 0.5)
    srv = make_server(tmp_path, port=0, runner=JobRunner(gated_stages(threading.Event())))
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
    for _ in range(3):
        with socket.create_connection((host, int(port)), timeout=5) as sock:
            sock.sendall(b"PUT /api/upload?name=a.mp4 HTTP/1.1\r\nHost: x\r\n"
                         b"Content-Length: 100000000\r\n\r\n" + b"\x00" * 100000)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    deadline = time.monotonic() + 5
    while list(tmp_path.glob("*/video.*")) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert list(tmp_path.glob("*/video.*")) == []
    time.sleep(0.2)
    assert "Traceback" not in capsys.readouterr().err
