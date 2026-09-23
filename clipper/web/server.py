from __future__ import annotations

import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from clipper.run import Run, default_run_name
from clipper.web.jobs import JobBusy, JobRunner

CHUNK = 1 << 20
PROFILES: list[tuple[str, str]] = [
    ("auto", "Auto (Laya decides)"),
    ("podcast", "Podcast"),
    ("talking_head", "Talking head"),
    ("lecture", "Lecture"),
    ("stream", "Stream (gaming, just chatting)"),
]
DEVICES = ["auto", "xpu", "cpu"]
MODELS = ["large-v3", "medium", "small"]
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
INDEX = Path(__file__).with_name("index.html")


class App:
    """Request logic, kept apart from HTTP plumbing."""

    def __init__(self, runs_dir: Path, runner: JobRunner) -> None:
        self.runs_dir = runs_dir
        self.runner = runner

    def config(self) -> dict:
        return {"profiles": [{"id": pid, "label": label} for pid, label in PROFILES],
                "devices": DEVICES, "models": MODELS,
                "hf_token": bool(os.environ.get("HF_TOKEN"))}

    def receive_upload(self, filename: str, stream, length: int) -> str:
        base = re.split(r"[\\/]", filename)[-1].strip()
        suffix = Path(base).suffix.lower()
        if suffix not in VIDEO_EXTS:
            raise ValueError(f"Not a supported video file. Use one of: {', '.join(VIDEO_EXTS)}.")
        if length <= 0:
            raise ValueError("No file data was received.")
        if self.runner.busy():
            raise JobBusy("A job is running. Upload again when it has finished.")

        run = Run.create(self.runs_dir, default_run_name(Path(base)))
        for old in run.root.glob("video.*"):
            old.unlink()
        target = run.path("video" + suffix)
        remaining = length
        try:
            with target.open("wb") as out:
                while remaining > 0:
                    chunk = stream.read(min(CHUNK, remaining))
                    if not chunk:
                        raise ValueError("The upload ended before the whole file arrived.")
                    out.write(chunk)
                    remaining -= len(chunk)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return run.root.name

    def start(self, body: dict) -> dict:
        name = str(body.get("run") or "")
        run_dir = self.runs_dir / name
        if not name or Path(name).name != name or name in (".", "..") or not run_dir.is_dir():
            raise ValueError("Unknown run. Upload a video first.")
        video = next(iter(sorted(run_dir.glob("video.*"))), None)
        if video is None:
            raise ValueError("This run has no video. Upload a file first.")
        self.runner.start(Run(run_dir), video, validate_settings(body))
        return self.runner.status()


def validate_settings(body: dict) -> dict:
    profile = body.get("profile", "auto")
    if profile not in dict(PROFILES):
        raise ValueError(f"Unknown profile {profile!r}.")
    device = body.get("device", "auto")
    if device not in DEVICES:
        raise ValueError(f"Unknown device {device!r}.")
    model = body.get("model", MODELS[0])
    if model not in MODELS:
        raise ValueError(f"Unknown Whisper model {model!r}.")
    return {"profile": profile, "device": device, "model": model,
            "diarize": bool(body.get("diarize", True)),
            "vertical": bool(body.get("vertical", True)),
            "prompt": str(body.get("prompt") or "").strip()[:2000]}


class _CountingReader:
    """Wraps the request body stream and remembers how much has been read."""

    def __init__(self, stream) -> None:
        self.stream = stream
        self.consumed = 0

    def read(self, size: int = -1) -> bytes:
        data = self.stream.read(size)
        self.consumed += len(data)
        return data


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002 - quiet console
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data: dict) -> None:
            self._send(status, json.dumps(data).encode("utf-8"), "application/json")

        def _length(self) -> int:
            try:
                return int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return 0

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/config":
                self._json(200, app.config())
            elif path == "/api/status":
                self._json(200, app.runner.status())
            else:
                self._json(404, {"error": "Not found."})

        def _refuse(self, status: int, message: str, body: _CountingReader) -> None:
            # Read and discard whatever of the body is still in flight before
            # answering. Closing a socket with unread data makes Windows send
            # a reset, and the client then sees "connection aborted" instead
            # of this message.
            remaining = self._length() - body.consumed
            while remaining > 0:
                chunk = body.read(min(CHUNK, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
            self.close_connection = True
            self._json(status, {"error": message})

        def do_PUT(self) -> None:
            url = urlparse(self.path)
            body = _CountingReader(self.rfile)
            if url.path != "/api/upload":
                self._refuse(404, "Not found.", body)
                return
            name = parse_qs(url.query).get("name", [""])[0]
            try:
                run = app.receive_upload(name, body, self._length())
            except JobBusy as error:
                self._refuse(409, str(error), body)
                return
            except ValueError as error:
                self._refuse(400, str(error), body)
                return
            except OSError as error:
                self._refuse(500, f"Could not save the upload: {error}", body)
                return
            self._json(200, {"run": run})

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/start":
                self._json(404, {"error": "Not found."})
                return
            try:
                body = json.loads(self.rfile.read(self._length()) or b"{}")
                snapshot = app.start(body)
            except JobBusy as error:
                self._json(409, {"error": str(error)})
                return
            except (ValueError, json.JSONDecodeError) as error:
                self._json(400, {"error": str(error)})
                return
            self._json(202, snapshot)

    return Handler


def make_server(runs_dir: Path, port: int = 8765,
                runner: JobRunner | None = None) -> ThreadingHTTPServer:
    runs_dir.mkdir(parents=True, exist_ok=True)
    app = App(runs_dir, runner or JobRunner())
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
