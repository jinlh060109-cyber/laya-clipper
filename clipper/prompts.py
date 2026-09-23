"""Step 9: one edit instruction per clip.

`edit.json` is the structured half, which the default editor applies;
`prompt.md` is the same plus the style notes and the clip's words, written
for an AI editor (the clipper skill, or later an MCP video tool).
"""
from __future__ import annotations


def _clock(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def edit_spec(clip: dict, style: dict, fill: dict | None, content_type: str,
              stem: str) -> dict:
    fill = fill or {}
    return {
        "id": clip["id"], "folder": stem,
        "source_range": {"in": clip["start"], "out": clip["end"]},
        "duration": clip["duration"], "category": clip.get("category", "clip"),
        "content_type": content_type, "source": clip.get("source", "ai"),
        "laya": {"score": clip.get("score"), "confidence": clip.get("confidence"),
                 "uncertain": clip.get("uncertain", False)},
        "title": fill.get("title") or clip.get("title_hint") or "",
        "hook": fill.get("hook", ""), "description": fill.get("description", ""),
        "caption_quote": fill.get("caption_quote", ""),
        "punch_ins": fill.get("punch_ins", []),
        "fill_in_used": bool(fill),
        **{key: style[key] for key in ("layout", "vertical", "captions",
                                       "caption_case", "encoder")},
    }


def _rating(spec: dict) -> str:
    score = spec["laya"]["score"] or 0.0
    if spec.get("source") == "action":
        # Nobody speaks, so there was nothing for Laya to read.
        return (f"- Action score {score:.2f} (loudness, motion and scene cuts; "
                f"there is no speech to rate).")
    return (f"- Laya score {score:.2f}, confidence {spec['laya']['confidence'] or 0.0:.2f}"
            + (" (Laya was unsure)." if spec["laya"]["uncertain"] else "."))


def render_prompt(spec: dict, style_notes: str, transcript_slice: str) -> str:
    rng = spec["source_range"]
    lines = [
        f"# Edit: {spec['title'] or spec['id']}",
        "",
        "## Style (from the creator, apply to every clip)",
        style_notes.strip() or "No style notes were given. Keep the edit clean and simple.",
        "",
        "## This clip",
        f"- Cut from the original video at {_clock(rng['in'])}-{_clock(rng['out'])} "
        f"({spec['duration']:.1f} s); already cut as `cut.mp4`.",
        f"- Kind of video: {spec['content_type']}; kind of moment: {spec['category']}.",
        _rating(spec),
        f"- Layout: {spec['layout']}, {'vertical 9:16' if spec['vertical'] else 'original shape'}; "
        f"captions: {spec['captions']} ({spec['caption_case']} case).",
    ]
    if spec["title"]:
        lines.append(f"- Title: {spec['title']}")
    if spec["hook"]:
        lines.append(f"- On-screen hook: {spec['hook']}")
    if spec["caption_quote"]:
        lines.append(f"- Quote to feature: \"{spec['caption_quote']}\"")
    if spec["description"]:
        lines.append(f"- Post description: {spec['description']}")
    for punch in spec["punch_ins"]:
        lines.append(f"- Punch-in at {punch['at']:.2f} s: {punch['reason']}")
    lines += [
        "",
        "## What is already done",
        "`final.mp4` is the default edit: the layout above and burned captions. "
        "Apply what it could not (punch-ins, pacing, anything in the style notes) "
        "and write the result to `edited.mp4` in this folder.",
        "",
        "## Transcript (seconds from the clip's start)",
        transcript_slice or "(no speech in this clip)",
        "",
    ]
    return "\n".join(lines)
