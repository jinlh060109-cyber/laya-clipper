from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class PreflightError(RuntimeError):
    """A required external tool is missing or unusable."""


@dataclass(frozen=True)
class Preflight:
    ffmpeg: Path
    ffprobe: Path


def find_binary(name: str, env_var: str) -> Path:
    override = os.environ.get(env_var)
    if override:
        candidate = Path(override)
        if candidate.exists():
            return candidate
    found = shutil.which(name)
    if found:
        return Path(found)
    raise PreflightError(
        f"{name} not found. Set {env_var} or install ffmpeg and put it on PATH. "
        f"Windows: download a build from https://www.gyan.dev/ffmpeg/builds/ "
        f"(the 'full' build includes libass)."
    )


def _run_filters(ffmpeg: Path) -> str:
    result = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout + result.stderr


def has_subtitles_filter(ffmpeg: Path) -> bool:
    return re.search(r"^\s*\S+\s+subtitles\s", _run_filters(ffmpeg), re.MULTILINE) is not None


def preflight(require_subtitles: bool = True) -> Preflight:
    ffmpeg = find_binary("ffmpeg", "FFMPEG_PATH")
    ffprobe = find_binary("ffprobe", "FFPROBE_PATH")
    if require_subtitles and not has_subtitles_filter(ffmpeg):
        raise PreflightError(
            "This ffmpeg build has no 'subtitles' filter, so captions cannot be "
            "burned in. It was built without libass. Install a full build "
            "(Windows: gyan.dev 'full'; macOS: brew install ffmpeg-full) or run "
            "with --captions sidecar."
        )
    return Preflight(ffmpeg=ffmpeg, ffprobe=ffprobe)
