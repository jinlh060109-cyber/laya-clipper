from __future__ import annotations

import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, unquote, urlparse

from clipper import ai, style
from clipper.run import Run, default_run_name
from clipper.selection import apply_choices
from clipper.web.jobs import JobBusy, JobRunner

CHUNK = 1 << 20
# Seconds a client may go silent mid-request before its connection is dropped.
REQUEST_TIMEOUT = 60.0
DEVICES = ("auto", "cuda", "xpu", "mps", "cpu")
MODELS = ["large-v3", "medium", "small"]
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
MEDIA_TYPES = {".mp4": "video/mp4", ".mov": "video/quicktime", ".mkv": "video/x-matroska",
               ".webm": "video/webm", ".avi": "video/x-msvideo", ".m4v": "video/mp4",
               ".md": "text/markdown; charset=utf-8", ".json": "application/json",
               ".srt": "text/plain; charset=utf-8", ".jpg": "image/jpeg"}
INDEX = Path(__file__).with_name("index.html")


class NotFound(LookupError):
    pass


def _hardware_probe() -> dict:
    from clipper.hardware import detect

    return detect()


class App:
    """Request logic, kept apart from HTTP plumbing."""

    def __init__(self, runs_dir: Path, runner: JobRunner,
                 hardware: Callable[[], dict] = _hardware_probe) -> None:
        self.runs_dir = runs_dir
        self.runner = runner
        self._hardware_probe = hardware
        self._hardware: dict | None = None

    def hardware(self) -> dict:
        # Probing torch and test-encoding with ffmpeg takes seconds; once is enough.
        if self._hardware is None:
            self._hardware = self._hardware_probe()
        return self._hardware

    def config(self) -> dict:
        try:
            active, error = ai.config_from_env(), None
        except ValueError as exc:
            active, error = None, str(exc)
        return {"hardware": self.hardware(), "models": MODELS,
                "ai": {"providers": ai.available_providers(),
                       "active": active.describe() if active else None, "error": error},
                "hf_token": bool(os.environ.get("HF_TOKEN")), "style": style.load()}

    def run_dir(self, name: str) -> Path:
        run_dir = self.runs_dir / name
        if not name or Path(name).name != name or name in (".", "..") or not run_dir.is_dir():
            raise ValueError("Unknown run. Upload a video first.")
        return run_dir

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

    def analyze(self, body: dict) -> dict:
        run_dir = self.run_dir(str(body.get("run") or ""))
        video = next(iter(sorted(run_dir.glob("video.*"))), None)
        if video is None:
            raise ValueError("This run has no video. Upload a file first.")
        self.runner.start(Run(run_dir), "analyze", validate_settings(body), video=video)
        return self.runner.status()

    def selection(self, name: str) -> dict:
        try:
            run = Run(self.run_dir(name))
        except ValueError as exc:
            raise NotFound(str(exc)) from exc
        if not run.exists("selection.json"):
            raise NotFound("This run has not been analyzed yet.")
        segments = run.read_json("segments.json") if run.exists("segments.json") else {}
        video = next(iter(sorted(run.root.glob("video.*"))), None)
        return {"run": name, "clips": run.read_json("selection.json")["clips"],
                "video": video.name if video else None,
                "content_type": segments.get("content_type"),
                "summary": segments.get("summary", ""),
                "ai": segments.get("ai"), "ai_error": segments.get("ai_error"),
                "questions": [{"id": qid, "type": q["type"], "instructions": q["instructions"]}
                              for qid, q in (segments.get("questions") or {}).items()],
                "made": run.read_json("clips.json")["clips"] if run.exists("clips.json") else []}

    def make(self, body: dict) -> dict:
        run = Run(self.run_dir(str(body.get("run") or "")))
        if not run.exists("selection.json"):
            raise ValueError("Analyze this video first; there is no clip selection yet.")
        if self.runner.busy():
            raise JobBusy("A job is already running. Wait for it to finish.")
        chosen_style = style.save(body.get("style") or {})
        choices = body.get("choices") or []
        if not isinstance(choices, list):
            raise ValueError("choices must be a list.")
        apply_choices(run, choices)
        self.runner.start(run, "make", {"style": chosen_style})
        return self.runner.status()

    def media(self, path: str) -> Path:
        """A file inside a run folder, and never anything outside one."""
        name, _, rel = unquote(path).partition("/")
        try:
            run_dir = self.run_dir(name).resolve()
        except ValueError as exc:
            raise NotFound("No such file.") from exc
        target = (run_dir / rel).resolve()
        if run_dir not in target.parents or not target.is_file():
            raise NotFound("No such file.")
        return target


def validate_settings(body: dict) -> dict:
    device = body.get("device", "auto")
    if device not in DEVICES:
        raise ValueError(f"Unknown device {device!r}.")
    model = body.get("model", MODELS[0])
    if model not in MODELS:
        raise ValueError(f"Unknown Whisper model {model!r}.")
    top_n = body.get("top_n", 5)
    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 30:
        raise ValueError("top_n (clips to keep) must be a whole number from 1 to 30.")
    return {"device": device, "model": model, "diarize": bool(body.get("diarize", True)),
            "prompt": str(body.get("prompt") or "").strip()[:2000], "top_n": top_n}


class _CountingReader:
    """Wraps the request body stream and remembers how much has been read."""

    def __init__(self, stream) -> None:
        self.stream = stream
        self.consumed = 0

    def read(self, size: int = -1) -> bytes:
        data = self.stream.read(size)
        self.consumed += len(data)
        return data


def _byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", (header or "").strip())
    if not match or not (match.group(1) or match.group(2)):
        return None
    if match.group(1):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else size - 1
    else:
        start, end = max(0, size - int(match.group(2))), size - 1
    if start >= size or start > end:
        return None
    return start, min(end, size - 1)


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        timeout = REQUEST_TIMEOUT

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
                return max(0, int(self.headers.get("Content-Length") or 0))
            except ValueError:
                return 0

        def _body(self) -> dict:
            body = json.loads(self.rfile.read(self._length()) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("Send the settings as a JSON object.")
            return body

        def _file(self, target: Path) -> None:
            size = target.stat().st_size
            wanted = _byte_range(self.headers.get("Range"), size)
            start, end = wanted if wanted else (0, size - 1)
            self.send_response(206 if wanted else 200)
            self.send_header("Content-Type", MEDIA_TYPES.get(target.suffix.lower(),
                                                             "application/octet-stream"))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if wanted:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with target.open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = handle.read(min(CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            try:
                if url.path == "/":
                    self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/config":
                    self._json(200, app.config())
                elif url.path == "/api/status":
                    self._json(200, app.runner.status())
                elif url.path == "/api/selection":
                    self._json(200, app.selection(parse_qs(url.query).get("run", [""])[0]))
                elif url.path.startswith("/media/"):
                    self._file(app.media(url.path[len("/media/"):]))
                else:
                    self._json(404, {"error": "Not found."})
            except NotFound as error:
                self._json(404, {"error": str(error)})
            except (ConnectionError, TimeoutError):
                self.close_connection = True

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
            if url.path == "/api/style":
                try:
                    self._json(200, style.save(self._body()))
                except (ValueError, json.JSONDecodeError) as error:
                    self._json(400, {"error": str(error)})
                return
            body = _CountingReader(self.rfile)
            if url.path != "/api/upload":
                self._refuse(404, "Not found.", body)
                return
            name = parse_qs(url.query).get("name", [""])[0]
            try:
                run = app.receive_upload(name, body, self._length())
            except (ConnectionError, TimeoutError):
                # The client hung up or went silent; nobody is left to answer.
                self.close_connection = True
                return
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
            routes = {"/api/analyze": app.analyze, "/api/make": app.make}
            handler = routes.get(urlparse(self.path).path)
            if handler is None:
                self._json(404, {"error": "Not found."})
                return
            try:
                snapshot = handler(self._body())
            except JobBusy as error:
                self._json(409, {"error": str(error)})
                return
            except (ValueError, json.JSONDecodeError) as error:
                self._json(400, {"error": str(error)})
                return
            self._json(202, snapshot)

    return Handler


class _Server(ThreadingHTTPServer):
    def handle_error(self, request, client_address) -> None:
        # A browser tab closed mid-request is routine, not worth a traceback.
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def make_server(runs_dir: Path, port: int = 8765, runner: JobRunner | None = None,
                hardware: Callable[[], dict] | None = None) -> ThreadingHTTPServer:
    runs_dir.mkdir(parents=True, exist_ok=True)
    app = App(runs_dir, runner or JobRunner(), hardware or _hardware_probe)
    return _Server(("127.0.0.1", port), make_handler(app))
