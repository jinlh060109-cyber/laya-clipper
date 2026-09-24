"""End-to-end fixtures: a real `clipper web` process, a real browser, real
ffmpeg, Whisper and Laya. Nothing is faked.

Run with the GPU environment, which has the models installed:

    .venv-gpu/Scripts/python.exe -m pytest -m e2e tests/e2e

The source video is the 99 s camping story (speech, 1920x1080). Point
CLIPPER_E2E_VIDEO at another file to use that instead. Tests that need a
real AI read ALIBABA_TOKEN_PLAN_API_KEY (or CLIPPER_E2E_AI_PROVIDER plus that
provider's key) and are skipped without one.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = ROOT / "runs" / "2026-09-23-camping-story" / "video.mp4"


def ffprobe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,width,height:format=duration", "-of", "json",
                          str(path)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@pytest.fixture(scope="session")
def source_video() -> Path:
    video = Path(os.environ.get("CLIPPER_E2E_VIDEO") or DEFAULT_VIDEO)
    if not video.exists():
        pytest.skip(f"No source video at {video}")
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not on PATH")
    return video


class Server:
    def __init__(self, workdir: Path, env: dict) -> None:
        self.workdir = workdir
        self.log: list[str] = []
        self.env = env
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "clipper.cli", "web", "--port", "0", "--no-browser"],
            cwd=workdir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace")
        for line in self.proc.stdout:
            self.log.append(line)
            match = re.search(r"(http://127\.0\.0\.1:\d+)", line)
            if match:
                self.url = match.group(1)
                break
        else:
            raise RuntimeError("clipper web did not start:\n" + "".join(self.log))
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        for line in self.proc.stdout:
            self.log.append(line)

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    @property
    def runs(self) -> Path:
        return self.workdir / "runs"


def _clean_env(workdir: Path, extra: dict | None = None) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("AI_") or k.endswith("_API_KEY"))}
    env.update(CLIPPER_STYLE=str(workdir / "style.json"), PYTHONUTF8="1",
               AI_PROVIDER="none", **(extra or {}))
    return env


@pytest.fixture
def start_server(tmp_path):
    servers: list[Server] = []

    def start(extra_env: dict | None = None, workdir: Path | None = None) -> Server:
        where = workdir or tmp_path / "app"
        where.mkdir(exist_ok=True)
        server = Server(where, _clean_env(where, extra_env))
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.stop()


@pytest.fixture(scope="session")
def browser():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as p:
        instance = p.chromium.launch()
        yield instance
        instance.close()


@pytest.fixture
def page(browser):
    context = browser.new_context(viewport={"width": 1200, "height": 1000})
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.js_errors = errors
    yield page
    context.close()
    assert not errors, f"JavaScript errors on the page: {errors}"


def ai_env() -> dict | None:
    """Real AI credentials from the environment, or None to skip."""
    provider = os.environ.get("CLIPPER_E2E_AI_PROVIDER", "alibaba_token_plan")
    from clipper.ai import PROVIDERS
    key_env = PROVIDERS[provider]["key_env"]
    key = os.environ.get(key_env) if key_env else "unused"
    if not key:
        return None
    return {"provider": provider, "key_env": key_env, "key": key,
            "model": os.environ.get("CLIPPER_E2E_AI_MODEL", PROVIDERS[provider]["default_model"])}
