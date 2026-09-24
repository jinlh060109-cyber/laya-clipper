"""The whole product, as a user drives it: the web page in Chromium against a
real `clipper web` process, then the CLI on the run the page produced."""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

from .conftest import ai_env, ffprobe

pytestmark = pytest.mark.e2e

ANALYZE_TIMEOUT = 15 * 60 * 1000  # ms: Whisper + Laya on a CPU can be slow
MAKE_TIMEOUT = 10 * 60 * 1000


def _finish(page, selector, timeout):
    """Wait for `selector` to show, but stop at once if the job fails."""
    page.wait_for_selector(f"{selector}:not([hidden]), #error:not([hidden])", timeout=timeout)
    assert page.locator("#error").is_hidden(), page.inner_text("#error")


def _open(page, server):
    page.goto(server.url)
    page.wait_for_selector("body[data-ready]")


def _analyze(page, server, video, prompt=""):
    _open(page, server)
    page.set_input_files("#file", str(video))
    page.select_option("#model", "small")
    if prompt:
        page.fill("#prompt", prompt)
    page.fill("#top", "2")
    page.click("#analyze")
    _finish(page, "#preview", ANALYZE_TIMEOUT)
    return [p for p in server.runs.iterdir() if p.is_dir()][0]


def _preview_png(page, look):
    """Show the preview for `look` and return the JPEG bytes and its size."""
    page.click("#preview-btn")
    page.wait_for_function(
        "(look) => { const i = document.querySelector('#caption-preview');"
        " return !i.hidden && (i.dataset.look || '').startsWith(look + '|')"
        " && i.complete && i.naturalWidth > 0; }",
        arg=look, timeout=60_000)
    size = page.evaluate("[document.querySelector('#caption-preview').naturalWidth,"
                         " document.querySelector('#caption-preview').naturalHeight]")
    return page.evaluate("""async () => {
        const r = await fetch(document.querySelector('#caption-preview').src);
        return Array.from(new Uint8Array(await r.arrayBuffer()));
    }"""), size


def test_no_ai_flow_from_upload_to_captioned_clips(page, start_server, source_video):
    server = start_server()
    run = _analyze(page, server, source_video)
    summary = page.inner_text("#preview-summary")
    assert "No AI" in summary
    rows = page.locator("#rows tr")
    assert rows.count() >= 1

    # Every caption look renders a real, distinct frame in the browser.
    looks = [o["value"] for o in page.eval_on_selector_all(
        "#caption_style option", "els => els.map(e => ({value: e.value}))")]
    assert set(looks) >= {"classic", "bold", "boxed", "minimal", "one_word", "neon"}
    frames = {}
    for look in looks:
        page.select_option("#caption_style", look)
        data, size = _preview_png(page, look)
        assert data[:2] == [0xFF, 0xD8], look  # JPEG
        assert size == [540, 960], (look, size)  # half-size 9:16
        frames[look] = bytes(data)
    assert len(set(frames.values())) == len(looks), "two looks rendered the same frame"

    # Make only the first clip, in the bold look, and check the file.
    page.select_option("#caption_style", "bold")
    boxes = page.locator("#rows .include")
    for i in range(boxes.count()):
        boxes.nth(i).set_checked(i == 0)
    page.click("#make")
    _finish(page, "#done", MAKE_TIMEOUT)
    assert page.locator("#results video").count() == 1
    ready = page.wait_for_function(
        "() => { const v = document.querySelector('#results video');"
        " return v && v.readyState >= 1 && v.videoHeight; }", timeout=30_000)
    assert ready

    finals = list((run / "clips").glob("*/final.mp4"))
    assert len(finals) == 1
    streams = ffprobe(finals[0])["streams"]
    video = next(s for s in streams if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1080, 1920)
    assert any(s["codec_type"] == "audio" for s in streams)
    spec = json.loads((finals[0].parent / "edit.json").read_text(encoding="utf-8"))
    assert spec["caption_style"] == "bold"
    assert "bold look" in (finals[0].parent / "prompt.md").read_text(encoding="utf-8")
    assert (finals[0].parent / "captions.srt").read_text(encoding="utf-8").strip()

    # The style chosen on the page is remembered for the next visit.
    page.reload()
    page.wait_for_selector("body[data-ready]")
    assert page.input_value("#caption_style") == "bold"

    # The CLI can re-make the same run in another look.
    out = subprocess.run([sys.executable, "-m", "clipper.cli", "make", str(run),
                          "--caption-style", "one_word", "--captions", "burn"],
                         cwd=server.workdir, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", env=server.env)
    assert out.returncode == 0, out.stderr
    spec = json.loads(next((run / "clips").glob("*/edit.json")).read_text(encoding="utf-8"))
    assert spec["caption_style"] == "one_word"


def test_ai_provider_is_chosen_on_the_page_and_drives_analysis(page, start_server, source_video):
    creds = ai_env()
    if creds is None:
        pytest.skip("No AI key in the environment")
    server = start_server()
    _open(page, server)
    page.click("#ai-panel summary")
    assert page.inner_text("#ai-summary").startswith("none")

    # A wrong key is caught by the connection test, before any analysis.
    page.select_option("#ai-provider", creds["provider"])
    page.fill("#ai-model", creds["model"])
    page.fill("#ai-key", "sk-sp-definitely-wrong")
    page.click("#ai-save")
    page.wait_for_selector("#ai-result.ok")
    page.click("#ai-test")
    page.wait_for_selector("#ai-result.warn", timeout=120_000)

    page.fill("#ai-key", creds["key"])
    page.click("#ai-save")
    page.wait_for_selector("#ai-result.ok")
    page.click("#ai-test")
    page.wait_for_function("() => document.querySelector('#ai-result').textContent.startsWith('Works')",
                           timeout=120_000)
    env_text = (server.workdir / ".env").read_text(encoding="utf-8")
    assert f"{creds['key_env']}={creds['key']}" in env_text
    page_html = page.content()
    assert creds["key"] not in page_html  # the key never comes back to the page

    run = _analyze(page, server, source_video, prompt="the funniest moment")
    assert "Clips proposed by" in page.inner_text("#preview-summary")

    # AI fill-in on one clip writes the title into the edit.
    boxes, fills = page.locator("#rows .include"), page.locator("#rows .fill")
    for i in range(boxes.count()):
        boxes.nth(i).set_checked(i == 0)
        fills.nth(i).set_checked(i == 0)
    page.select_option("#caption_style", "neon")
    page.click("#make")
    _finish(page, "#done", MAKE_TIMEOUT)
    spec = json.loads(next((run / "clips").glob("*/edit.json")).read_text(encoding="utf-8"))
    assert spec["fill_in_used"] and spec["title"] and spec["caption_style"] == "neon"
    made = json.loads((run / "clips.json").read_text(encoding="utf-8"))["clips"][0]
    assert made["punch_ins"] == len(spec["punch_ins"])
    final = run / made["final"]
    duration = float(ffprobe(final)["format"]["duration"])
    assert abs(duration - spec["duration"]) < 0.5  # zooms never change the length

    # Switching to "no AI" keeps the saved key.
    page.evaluate("document.querySelector('#ai-panel').open = true")
    page.select_option("#ai-provider", "none")
    page.click("#ai-save")
    page.wait_for_selector("#ai-result.ok")
    assert page.inner_text("#ai-summary").startswith("none")
    assert creds["key"] in (server.workdir / ".env").read_text(encoding="utf-8")


def test_another_site_cannot_drive_the_local_server(start_server):
    server = start_server()
    request = urllib.request.Request(server.url + "/api/ai", method="PUT",
                                     data=b'{"provider": "custom", "base_url": "http://evil/v1"}',
                                     headers={"Origin": "http://evil.example",
                                              "Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=10)
    assert caught.value.code == 403
    assert not (server.workdir / ".env").exists()


def test_cli_lists_providers(start_server):
    out = subprocess.run([sys.executable, "-m", "clipper.cli", "providers"],
                         capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0
    for pid in ("anthropic", "alibaba_token_plan", "ollama"):
        assert pid in out.stdout
