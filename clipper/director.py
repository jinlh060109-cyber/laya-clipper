"""Step 11 (optional): an AI director adds motion graphics to each clip.

For every made clip the chosen AI runs as an agent with tools (ToolChat, so
any provider works). It reads only that clip's script, looks at frames of
the finished clip with its vision model, and places graphics from a fixed
Remotion kit (remotion/src/graphics.tsx): kinetic text, counters, lower
thirds, pointers, stickers, a progress bar. It previews its layer on real
frames and fixes what it sees before it renders. The AI never writes code:
it fills in props, checked here, and Remotion renders them as a transparent
layer that ffmpeg lays over final.mp4. `final_plain.mp4` keeps the clip
without graphics.

Free: Remotion's own licence is free for individuals and teams of up to
three; Node, Chrome Headless Shell and ffmpeg are free.
"""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

from clipper.ai import AIConfig, AIError, ToolChat
from clipper.captions import picture_band, reserved_bands
from clipper.fillin import clip_slice
from clipper.run import Run
from clipper.skills import body as skill

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT / "remotion"
MAX_TURNS = 14
MAX_ITEMS = 8
FRAME_WIDTH = 432
POSITIONS = ("top", "upper", "center", "lower")

_TEXT = {"type": "string"}
_WHEN = {"from": {"type": "number", "description": "Start, seconds from the clip's start."},
         "to": {"type": "number", "description": "End, seconds from the clip's start."}}
# The Remotion components and their props, as the AI sees them.
GRAPHICS: dict[str, dict] = {
    "kinetic_text": {
        "description": "Big animated words: a key number, claim or punchline, 1-5 words.",
        "props": {"text": _TEXT,
                  "animation": {"type": "string", "enum": ["pop", "slide", "typewriter", "punch"]},
                  "position": {"type": "string", "enum": list(POSITIONS)}},
        "required": ["text"],
    },
    "counter": {
        "description": "A number that counts up from 0, with an optional label under it.",
        "props": {"value": {"type": "number"}, "prefix": _TEXT, "suffix": _TEXT, "label": _TEXT,
                  "decimals": {"type": "integer"},
                  "position": {"type": "string", "enum": list(POSITIONS)}},
        "required": ["value"],
    },
    "lower_third": {
        "description": "A name or topic bar that slides in just above the captions.",
        "props": {"title": _TEXT, "subtitle": _TEXT},
        "required": ["title"],
    },
    "pointer": {
        "description": ("A ring around something in the picture with a short label. x and y "
                        "are 0-1 across and down the frame; find them by looking."),
        "props": {"x": {"type": "number"}, "y": {"type": "number"}, "label": _TEXT},
        "required": ["x", "y"],
    },
    "sticker": {
        "description": "A tilted badge with one or two words, for a reaction beat.",
        "props": {"text": _TEXT, "x": {"type": "number"}, "y": {"type": "number"}},
        "required": ["text"],
    },
    "progress_bar": {
        "description": "A thin bar across the top that fills as the clip plays. Place 0 to the end.",
        "props": {"position": {"type": "string", "enum": ["top", "bottom"]}},
        "required": [],
    },
}
LIMITS = {"text": 40, "label": 24, "title": 32, "subtitle": 40, "prefix": 4, "suffix": 12}


def available() -> str | None:
    """Why the director cannot run here, or None."""
    if shutil.which("node") is None:
        return "Remotion needs Node.js 18+ (https://nodejs.org). Install it and restart."
    if not (PROJECT / "node_modules" / "@remotion" / "renderer").exists():
        return f"Remotion is not installed. Run `npm install` in {PROJECT}."
    return None


# -- the plan the tools build ----------------------------------------------

def check_item(item: dict, duration: float) -> dict:
    """A graphic, cleaned; ValueError with a reason the AI can act on."""
    component = item.get("component")
    if component not in GRAPHICS:
        raise ValueError(f"Unknown component {component!r}. Use one of: {', '.join(GRAPHICS)}.")
    spec = GRAPHICS[component]
    try:
        start, end = float(item.get("from", 0)), float(item.get("to", duration))
    except (TypeError, ValueError):
        raise ValueError("from and to must be seconds.") from None
    start, end = max(0.0, start), min(duration, end)
    if component != "progress_bar" and not 0.8 <= end - start <= 8:
        raise ValueError(f"{component} would be on screen {end - start:.1f} s; keep it 0.8-8 s.")
    props = {}
    for key, value in (item.get("props") or {}).items():
        if key not in spec["props"]:
            continue
        kind = spec["props"][key]["type"]
        if kind == "string":
            value = str(value).strip()
            if key in LIMITS and len(value) > LIMITS[key]:
                raise ValueError(f"{key} is {len(value)} characters; keep it under {LIMITS[key]}.")
            allowed = spec["props"][key].get("enum")
            if allowed and value not in allowed:
                raise ValueError(f"{key} must be one of {', '.join(allowed)}.")
        else:
            try:
                value = int(value) if kind == "integer" else float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{key} must be a number.") from None
            if key in ("x", "y") and not 0 <= value <= 1:
                raise ValueError(f"{key} is a fraction of the frame, 0 to 1.")
        props[key] = value
    missing = [k for k in spec["required"] if k not in props or props[k] in ("", None)]
    if missing:
        raise ValueError(f"{component} needs {', '.join(missing)}.")
    return {"component": component, "from": round(start, 2), "to": round(end, 2), "props": props}


def _overlaps(a: dict, b: dict) -> bool:
    return a["from"] < b["to"] and b["from"] < a["to"]


def add_items(plan: list[dict], items: list[dict], duration: float) -> tuple[list[dict], list[str]]:
    """The plan with the items that passed, and a line per item on what happened."""
    out, notes = list(plan), []
    for raw in items:
        try:
            item = check_item(raw, duration)
        except ValueError as exc:
            notes.append(f"refused {raw.get('component')!r}: {exc}")
            continue
        text_like = {"kinetic_text", "counter"}
        clash = next((p for p in out if p["component"] in text_like and item["component"] in text_like
                      and _overlaps(p, item)
                      and p["props"].get("position", "center") == item["props"].get("position", "center")),
                     None)
        if clash:
            notes.append(f"refused {item['component']} at {item['from']}-{item['to']} s: it would sit "
                         f"on the {clash['component']} already there. Change the time or position.")
            continue
        if len(out) >= MAX_ITEMS:
            notes.append(f"refused {item['component']}: at most {MAX_ITEMS} graphics per clip.")
            continue
        out.append(item)
        notes.append(f"added #{len(out) - 1} {item['component']} {item['from']}-{item['to']} s")
    return out, notes


# -- rendering -------------------------------------------------------------

def slots(bands: dict, picture: tuple[int, int], height: int) -> dict:
    """Where each `position` puts a graphic (centre y, px), and what it covers.
    With a picture band (fit/black/square), top and lower use the empty space
    above and below the picture; otherwise all four are on the picture,
    between the hook title and the captions."""
    top, bottom = picture
    cap_top, cap_bottom, title = bands["caption_top"], bands["caption_bottom"], bands["title_bottom"]
    framed = top > 120
    if framed:
        below = max(cap_bottom, bottom) + 40
        out = {"top": (max(40, top // 2), "the empty band above the picture (the hook title is "
                                          "there for the first 2.8 s)"),
               "upper": (top + (bottom - top) // 4, "the upper part of the picture"),
               "center": ((top + bottom) // 2, "the middle of the picture"),
               "lower": ((below + height) // 2, "the empty band below the captions")}
    else:
        span = cap_top - title
        out = {"top": (title + span // 8, "the picture, just under the hook title"),
               "upper": (title + span // 3, "the upper part of the picture"),
               "center": (title + span // 2, "the middle of the picture"),
               "lower": (title + span * 5 // 6, "the picture, just above the captions")}
    return {name: {"y": int(y), "covers": what} for name, (y, what) in out.items()}


def overlay_props(plan: list[dict], width: int, height: int, fps: float, duration: float,
                  bands: dict, accent: str, picture: tuple[int, int] | None = None) -> dict:
    picture = picture or (0, height)
    places = slots(bands, picture, height)
    return {"width": width, "height": height, "fps": round(fps, 3), "duration": duration,
            "captionTop": bands["caption_top"], "captionBottom": bands["caption_bottom"],
            "titleBottom": bands["title_bottom"], "slots": {k: v["y"] for k, v in places.items()},
            "lowerThirdTop": (places["lower"]["y"] - 60 if picture[0] > 120
                              else bands["caption_top"] - 200),
            "accent": accent, "items": plan}


def _node(args: list[str], what: str, timeout: float = 900) -> str:
    result = subprocess.run(["node", str(PROJECT / "render.mjs"), *args], cwd=PROJECT,
                            stdin=subprocess.DEVNULL, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=timeout, check=False)
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout).strip().splitlines()[-8:])
        raise RuntimeError(f"Remotion could not {what}:\n{tail}")
    return result.stdout


def _ffmpeg(ffmpeg: str, args: list[str], what: str) -> bytes:
    result = subprocess.run([ffmpeg, "-v", "error", "-nostdin", *args], stdin=subprocess.DEVNULL,
                            capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Could not {what}: {result.stderr.decode(errors='replace')[-400:]}")
    return result.stdout


def frame_jpeg(ffmpeg: str, video: Path, at: float, overlay_png: Path | None = None) -> bytes:
    """One frame of the clip, with the planned graphics drawn on when given."""
    args = ["-ss", f"{at:.3f}", "-i", str(video)]
    if overlay_png is not None:
        args += ["-i", str(overlay_png), "-filter_complex",
                 f"[0:v][1:v]overlay=0:0,scale={FRAME_WIDTH}:-2"]
    else:
        args += ["-vf", f"scale={FRAME_WIDTH}:-2"]
    return _ffmpeg(ffmpeg, [*args, "-frames:v", "1", "-f", "image2pipe", "-c:v", "mjpeg",
                            "-q:v", "4", "-"], f"take the frame at {at:.1f} s")


def render_over(ffmpeg: str, folder: Path, props: dict, encoder: list[str]) -> None:
    """Renders the layer and lays it over final.mp4 (the plain clip is kept)."""
    plain, final = folder / "final_plain.mp4", folder / "final.mp4"
    if not plain.exists():
        final.replace(plain)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "props.json").write_text(json.dumps(props), encoding="utf-8")
        layer = Path(tmp) / "layer.mov"
        _node(["video", str(Path(tmp) / "props.json"), str(layer)], "render the graphics")
        out = Path(tmp) / "final.mp4"
        _ffmpeg(ffmpeg, ["-y", "-i", str(plain), "-i", str(layer), "-filter_complex",
                         "[0:v][1:v]overlay=0:0:eof_action=pass:format=auto,format=yuv420p[v]",
                         "-map", "[v]", "-map", "0:a?", *encoder, "-c:a", "copy",
                         "-movflags", "+faststart", str(out)], "lay the graphics over the clip")
        shutil.move(str(out), final)


# -- the agent ---------------------------------------------------------------

def _tools() -> list[dict]:
    item = {"type": "object", "properties": {
        "component": {"type": "string", "enum": list(GRAPHICS)}, **_WHEN,
        "props": {"type": "object", "description": "The component's props (see its description)."}},
        "required": ["component", "from", "to", "props"]}
    kit = "; ".join(f"{name}: {g['description']} Props: "
                    + ", ".join(k + ("*" if k in g["required"] else "") for k in g["props"])
                    for name, g in GRAPHICS.items())
    return [
        {"name": "look", "description": ("See frames of the finished clip (captions and hook title "
                                         "already burned in) at the given seconds. Up to 6 at once."),
         "parameters": {"type": "object", "properties": {
             "times": {"type": "array", "items": {"type": "number"}}}, "required": ["times"]}},
        {"name": "add_graphics", "description": ("Add graphics to the plan. Components (* = "
                                                 "required prop): " + kit),
         "parameters": {"type": "object", "properties": {
             "items": {"type": "array", "items": item}}, "required": ["items"]}},
        {"name": "remove_graphic", "description": "Remove a graphic from the plan by its # number.",
         "parameters": {"type": "object", "properties": {"index": {"type": "integer"}},
                        "required": ["index"]}},
        {"name": "preview", "description": ("See the clip with the planned graphics drawn on, at "
                                            "the given seconds (rendered by Remotion). Up to 4."),
         "parameters": {"type": "object", "properties": {
             "times": {"type": "array", "items": {"type": "number"}}}, "required": ["times"]}},
        {"name": "render", "description": ("Render the plan onto the clip. Call once, last. With "
                                           "an empty plan the clip stays as it is."),
         "parameters": {"type": "object", "properties": {"summary": {
             "type": "string", "description": "One sentence: what you added and why."}},
             "required": ["summary"]}},
    ]


def direct(config: AIConfig, folder: Path, script: str, clip: dict, notes: str, *,
           ffmpeg: str, width: int, height: int, fps: float, bands: dict, encoder: list[str],
           picture: tuple[int, int] | None = None, accent: str = "#FFD400",
           chat_factory=ToolChat) -> dict:
    """One clip: the agent looks, plans, previews and renders. The log is also
    written to director.json in the clip folder."""
    video = folder / "final.mp4"
    duration = float(clip["duration"])
    plan: list[dict] = []
    log: list[dict] = []
    rendered, summary = False, ""
    picture = picture or (0, height)
    places = slots(bands, picture, height)
    chat = chat_factory(config, skill("director"), _tools())
    where = "\n".join(f"- {name}: {v['covers']}" for name, v in places.items())
    chat.say(f"Clip: {clip.get('title') or clip['id']} ({duration:.1f} s, {width}x{height}).\n"
             f"Positions for kinetic_text and counter:\n{where}\n"
             f"lower_third sits {'in the band below the captions' if picture[0] > 120 else 'just above the captions'}. "
             f"Graphics never cover the captions or the hook title.\n"
             f"You have {MAX_TURNS} turns; render before they run out.\n"
             f"Creator's style notes: {notes.strip() or '(none)'}\n\n"
             f"Script (seconds from the clip's start):\n{script or '(no speech)'}")
    began = time.time()

    def stills(times: list[float]) -> dict[float, Path]:
        tmp = Path(tempfile.mkdtemp(dir=folder, prefix=".preview-"))
        props = overlay_props(plan, width, height, fps, duration, bands, accent, picture)
        (tmp / "props.json").write_text(json.dumps(props), encoding="utf-8")
        _node(["stills", str(tmp / "props.json"), str(tmp), ",".join(f"{t:.2f}" for t in times)],
              "draw the preview")
        return {t: tmp / f"{t:.2f}.png" for t in times}

    def clamp_times(raw, most: int) -> list[float]:
        times = []
        for t in (raw or [])[:most]:
            try:
                times.append(round(min(max(0.0, float(t)), max(0.0, duration - 0.05)), 2))
            except (TypeError, ValueError):
                continue
        return times or [round(duration / 2, 2)]

    for turn in range(MAX_TURNS):
        if turn == MAX_TURNS - 3 and not rendered:
            chat.say("Three turns left: settle the plan and call render.")
        text, calls = chat.step(3000)
        if not calls:
            if rendered or not plan:
                break
            chat.say("Call render to finish (or remove the graphics you do not want first).")
            continue
        results = []
        for call in calls:
            name, args = call["name"], call.get("arguments") or {}
            entry = {"tool": name, "arguments": args}
            try:
                if name == "look":
                    times = clamp_times(args.get("times"), 6)
                    blocks = [{"type": "text", "text": f"Frames at {', '.join(f'{t} s' for t in times)}:"}]
                    blocks += [{"type": "image", "jpeg": base64.b64encode(
                        frame_jpeg(ffmpeg, video, t)).decode()} for t in times]
                    results.append((call["id"], blocks))
                    entry["result"] = f"{len(times)} frames"
                elif name == "add_graphics":
                    plan, lines = add_items(plan, list(args.get("items") or []), duration)
                    reply = "\n".join(lines) + "\nPlan now:\n" + _describe(plan)
                    results.append((call["id"], reply))
                    entry["result"] = reply
                elif name == "remove_graphic":
                    index = int(args.get("index", -1))
                    if not 0 <= index < len(plan):
                        raise ValueError(f"There is no graphic #{index}.")
                    plan.pop(index)
                    reply = "Removed. Plan now:\n" + _describe(plan)
                    results.append((call["id"], reply))
                    entry["result"] = reply
                elif name == "preview":
                    times = clamp_times(args.get("times"), 4)
                    pngs = stills(times) if plan else {t: None for t in times}
                    blocks = [{"type": "text", "text": "With the planned graphics at "
                               + ", ".join(f"{t} s" for t in times) + ":"}]
                    blocks += [{"type": "image", "jpeg": base64.b64encode(
                        frame_jpeg(ffmpeg, video, t, png)).decode()} for t, png in pngs.items()]
                    for png in pngs.values():
                        if png is not None:
                            shutil.rmtree(png.parent, ignore_errors=True)
                            break
                    results.append((call["id"], blocks))
                    entry["result"] = f"{len(times)} previews"
                elif name == "render":
                    summary = str(args.get("summary") or "")
                    if plan:
                        render_over(ffmpeg, folder, overlay_props(plan, width, height, fps, duration,
                                                                  bands, accent, picture), encoder)
                    rendered = True
                    reply = f"Rendered {len(plan)} graphics onto final.mp4." if plan else "Nothing to render."
                    results.append((call["id"], reply))
                    entry["result"] = reply
                else:
                    raise ValueError(f"There is no tool {name!r}.")
            except (ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                results.append((call["id"], f"Error: {exc}"))
                entry["result"] = f"Error: {exc}"
            log.append(entry)
        chat.reply(results)
        if rendered:
            break

    for leftover in folder.glob(".preview-*"):
        shutil.rmtree(leftover, ignore_errors=True)
    record = {"model": f"{config.provider}/{config.model}", "vision": getattr(chat, "vision", None),
              "graphics": plan, "rendered": rendered and bool(plan), "summary": summary,
              "seconds": round(time.time() - began, 1), "calls": log}
    (folder / "director.json").write_text(json.dumps(record, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    return record


def _describe(plan: list[dict]) -> str:
    if not plan:
        return "(empty)"
    return "\n".join(f"#{k} {p['component']} {p['from']}-{p['to']} s {json.dumps(p['props'], ensure_ascii=False)}"
                     for k, p in enumerate(plan))


def run_director(run: Run, style: dict, done: list[dict], config: AIConfig | None,
                 progress: Callable[[int, int], None] | None = None, ffmpeg=None) -> list[dict]:
    """Every made clip through the director. A clip the director fails on
    keeps its plain edit, and the reason goes on the clip."""
    if config is None:
        raise ValueError("The AI director needs an AI. Pick a provider in the AI panel, or switch "
                         "the director off.")
    why = available()
    if why:
        raise ValueError(why)
    from clipper.edit import display_size
    from clipper.hardware import encoder_args, pick_encoder, probe_encoders
    from clipper.preflight import preflight

    ffmpeg = str(ffmpeg or preflight(require_subtitles=False).ffmpeg)
    meta = run.read_json("source.json")
    width, height = (1080, 1920) if style["vertical"] else display_size(meta["video"])
    fps = float(meta["video"].get("fps") or 30.0)
    encoder = [*encoder_args(pick_encoder(style["encoder"], probe_encoders(ffmpeg))),
               "-pix_fmt", "yuv420p"]
    bands = reserved_bands(style["layout"], height, style.get("caption_style", "classic"))
    picture = picture_band(style["layout"], height) if style["vertical"] else (0, height)
    if style["captions"] != "burn":
        bands = {"caption_top": height - 40, "caption_bottom": height, "title_bottom": 0}
    elif not style.get("hook_title", True):
        bands = {**bands, "title_bottom": 0}
    from clipper.silence import speech_only
    transcript = speech_only(run.read_json("transcript.json"))
    selected = {c["id"]: c for c in run.read_json("selection.json")["clips"]}
    out = []
    for index, made in enumerate(done, start=1):
        folder = run.clips_dir() / made["folder"]
        source = selected.get(made["id"], {})
        start = float(source.get("start", 0.0))
        script = clip_slice(transcript, start, start + float(made["duration"]))
        try:
            record = direct(config, folder, script, {**made, "id": made["id"]},
                            style.get("notes", ""), ffmpeg=ffmpeg, width=width, height=height,
                            fps=fps, bands=bands, encoder=encoder, picture=picture)
            made = {**made, "director": {"graphics": len(record["graphics"]),
                                         "rendered": record["rendered"],
                                         "summary": record["summary"], "vision": record["vision"]}}
        except (AIError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            made = {**made, "director": {"error": str(exc)}}
        out.append(made)
        if progress is not None:
            progress(index, len(done))
    run.write_json("clips.json", {"clips": out})
    return out
