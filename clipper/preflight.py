from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class PreflightError(RuntimeError):
    """A required external tool is missing or unusable."""


class _FfmpegRunError(RuntimeError):
    """Internal signal: the binary ran but exited non-zero (not a libass issue)."""

    def __init__(self, returncode: int, output: str) -> None:
        super().__init__(f"ffmpeg exited with code {returncode}")
        self.returncode = returncode
        self.output = output


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
        raise PreflightError(
            f"{env_var} is set to '{override}' but that path does not exist. "
            f"Fix {env_var} to point at your {name} binary, or unset it to fall "
            f"back to PATH."
        )
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
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise _FfmpegRunError(result.returncode, output)
    return output


def has_subtitles_filter(ffmpeg: Path) -> bool:
    try:
        output = _run_filters(ffmpeg)
    except _FfmpegRunError as exc:
        excerpt = exc.output.strip()[-500:]
        raise PreflightError(
            f"{ffmpeg} exited with code {exc.returncode} while listing filters. "
            f"The binary itself failed to run; check that it is valid and "
            f"executable for this platform. Output excerpt: {excerpt!r}"
        ) from exc
    return re.search(r"^\s*\S+\s+subtitles\s", output, re.MULTILINE) is not None


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
