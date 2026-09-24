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


def _analyze(page, server, video, prompt="", reload=True):
    if reload:
        _open(page, server)
    page.set_input_files("#file", str(video))
    page.select_option("#model", "small")
    if prompt:
        page.fill("#prompt", prompt)
    page.fill("#top", "2")
    page.click("#analyze")
    _finish(page, "#preview", ANALYZE_TIMEOUT)
    return [p for p in server.runs.iterdir() if p.is_dir()][0]


def _preview_png(page):
    """Wait until the preview shows the look now chosen on the page (no click:
    choosing a look is enough) and return the JPEG bytes and its size."""
    page.wait_for_function(
        "() => { const i = document.querySelector('#caption-preview');"
        " const v = (id) => document.getElementById(id).value;"
        " const look = [v('caption_style'), v('caption_case'), v('layout'), v('captions'),"
        "   document.getElementById('vertical').checked].join('|');"
        " return !i.hidden && i.dataset.look === look && !i.classList.contains('stale')"
        " && i.complete && i.naturalWidth > 0; }",
        timeout=60_000)
    size = page.evaluate("[document.querySelector('#caption-preview').naturalWidth,"
                         " document.querySelector('#caption-preview').naturalHeight]")
    return page.evaluate("""async () => {
        const r = await fetch(document.querySelector('#caption-preview').src);
        return Array.from(new Uint8Array(await r.arrayBuffer()));
    }"""), size


def test_no_ai_flow_from_upload_to_captioned_clips(artifacts, page, start_server, source_video):
    server = start_server()
    run = _analyze(page, server, source_video)
    artifacts.watch(run)
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
        data, size = _preview_png(page)
        assert data[:2] == [0xFF, 0xD8], look  # JPEG
        assert size == [540, 960], (look, size)  # half-size 9:16
        frames[look] = bytes(data)
        artifacts.save_bytes(f"previews/{look}.jpg", frames[look])
    assert len(set(frames.values())) == len(looks), "two looks rendered the same frame"

    # Every layout shows at once on choosing it, each a different 9:16 frame.
    layouts = page.eval_on_selector_all("#layout option", "els => els.map(e => e.value)")
    assert set(layouts) >= {"fit", "crop", "black", "square", "split"}
    shots = {}
    for layout in layouts:
        page.select_option("#layout", layout)
        data, size = _preview_png(page)
        assert size == [540, 960], (layout, size)
        shots[layout] = bytes(data)
        artifacts.save_bytes(f"layouts/{layout}.jpg", shots[layout])
    assert len(set(shots.values())) == len(layouts), "two layouts rendered the same frame"
    page.select_option("#layout", "fit")

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
    artifacts.note(looks=looks, preview_size=size, made_in="bold")

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
    artifacts.note(cli_remade_in="one_word")


def test_ai_provider_is_chosen_on_the_page_and_drives_analysis(artifacts, page, start_server,
                                                               source_video):
    creds = ai_env()
    if creds is None:
        pytest.skip("No AI key in the environment")
    server = start_server()
    _open(page, server)
    page.click("#ai-panel summary")
    assert page.inner_text("#ai-summary").startswith("none")

    # The user picks a provider, types a key and tests it straight away,
    # without saving first. A wrong key is caught before any analysis.
    page.select_option("#ai-provider", creds["provider"])
    page.select_option("#ai-model-pick", creds["model"])
    assert not page.is_visible("#ai-model")  # one model field: the list
    page.fill("#ai-key", "sk-sp-" + "x" * 24)
    assert page.is_visible("#ai-pending")
    page.click("#ai-test")
    page.wait_for_selector("#ai-result.warn", timeout=120_000)
    assert "No AI" not in page.inner_text("#ai-result"), page.inner_text("#ai-result")

    page.fill("#ai-key", creds["key"])
    # With the real key, the list comes from the provider itself.
    page.dispatch_event("#ai-key", "change")
    page.wait_for_function("() => /models available/.test(document.querySelector('#ai-models-state').textContent)",
                           timeout=60_000)
    listed = page.eval_on_selector_all("#ai-model-pick option", "els => els.map(e => e.value)")
    assert creds["model"] in listed, listed
    artifacts.save_bytes("provider-models.txt", "\n".join(listed).encode())
    page.select_option("#ai-model-pick", creds["model"])
    page.click("#ai-test")
    page.wait_for_function("() => document.querySelector('#ai-result').textContent.startsWith('Works')",
                           timeout=120_000)
    assert not (server.workdir / ".env").exists()  # tested, not saved
    assert page.inner_text("#ai-summary").endswith("(not saved)")

    # Analyzing uses what the form shows, still without "Use this AI".
    run = _analyze(page, server, source_video, prompt="the funniest moment", reload=False)
    artifacts.watch(run)
    assert "Clips proposed by" in page.inner_text("#preview-summary")
    env_text = (server.workdir / ".env").read_text(encoding="utf-8")
    assert f"{creds['key_env']}={creds['key']}" in env_text
    assert page.is_hidden("#ai-pending")
    assert creds["key"] not in page.content()  # the key never comes back to the page

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
    ass = (final.parent / "captions.ass").read_text(encoding="utf-8")
    assert spec["hook"] and ",Title," in ass  # the AI's hook is on screen first
    artifacts.note(provider=creds["provider"], model=creds["model"], title=spec["title"],
                   hook=spec["hook"], punch_ins=[p["at"] for p in spec["punch_ins"]],
                   timings=json.loads((run / "job.json").read_text(encoding="utf-8")).get("timings"))

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
