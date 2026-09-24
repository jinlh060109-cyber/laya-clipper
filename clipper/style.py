"""Step 7: the user's style, written once and reused for every clip.

Structured fields drive the default editor; `notes` (tone, pacing, brand
rules) are passed verbatim to whoever edits the clip. No AI rewrites this.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from clipper.captions import CAPTION_STYLES
from clipper.filters import LAYOUTS
from clipper.hardware import ENCODERS

DEFAULT = {"layout": "fit", "vertical": True, "captions": "burn",
           "caption_style": "classic", "caption_case": "sentence", "encoder": "auto",
           "punch_ins": True, "notes": ""}
CHOICES = {"layout": LAYOUTS, "captions": ("burn", "sidecar", "none"),
           "caption_style": tuple(CAPTION_STYLES),
           "caption_case": ("sentence", "upper"),
           "encoder": ("auto", *(eid for eid, _, _ in ENCODERS))}
MAX_NOTES = 4000


def path() -> Path:
    return Path(os.environ.get("CLIPPER_STYLE") or "style.json")


def load() -> dict:
    try:
        data = json.loads(path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(DEFAULT)
    if not isinstance(data, dict):
        return dict(DEFAULT)
    try:
        return validate({**DEFAULT, **data})
    except ValueError:
        return dict(DEFAULT)


def validate(data: dict) -> dict:
    out = dict(DEFAULT)
    for key in DEFAULT:
        if key in data:
            out[key] = data[key]
    for key, allowed in CHOICES.items():
        if out[key] not in allowed:
            raise ValueError(f"Style {key} must be one of {', '.join(allowed)}, "
                             f"not {out[key]!r}.")
    out["vertical"] = bool(out["vertical"])
    out["punch_ins"] = bool(out["punch_ins"])
    out["notes"] = str(out["notes"] or "")
    if len(out["notes"]) > MAX_NOTES:
        raise ValueError(f"Style notes are limited to {MAX_NOTES} characters.")
    return out


def save(data: dict) -> dict:
    style = validate({**load(), **data})
    target = path()
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(style, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)
    return style
