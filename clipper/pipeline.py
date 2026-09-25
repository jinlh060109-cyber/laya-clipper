"""The two jobs the app runs, as ordered steps.

Analyze (steps 1-6): read the video, transcribe it, let the AI propose clips
and Laya questions, let Laya rate them, find moments without speech, and
pick what to show in the preview.
Make clips (steps 7-11): fix the style, optionally let the AI fill in each
clip, cut and edit every ticked clip, then optionally let the AI director
add Remotion motion graphics to each (clipper.director).

Steps are injectable so the job runner is tested without any model or ffmpeg.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from clipper.run import Run

ANALYZE = ("ingest", "transcribe", "segment", "rate", "action", "select")
MAKE = ("style", "fillin", "edit", "director")
Progress = Callable[[int, int], None]


@dataclass
class Steps:
    ingest: Callable[[Run, Path], Any]
    transcribe: Callable[[Run, dict], Any]
    segment: Callable[[Run, dict], dict]
    rate: Callable[[Run, dict, Progress], dict]
    action: Callable[[Run, Progress], dict]
    select: Callable[[Run, dict], dict]
    style: Callable[[Run, dict], dict]
    fillin: Callable[[Run, dict, Progress], dict]
    edit: Callable[[Run, dict, dict, Progress], list]
    director: Callable[[Run, dict, list, Progress], list] = (
        lambda run, style, done, progress: done)


def default_steps() -> Steps:
    """The real steps. Imports stay inside so importing this module is cheap."""

    def ingest(run: Run, video: Path) -> None:
        from clipper.ingest import ingest as read_video
        from clipper.preflight import preflight

        tools = preflight(require_subtitles=False)
        read_video(tools.ffmpeg, tools.ffprobe, video, run)

    def transcribe(run: Run, settings: dict) -> None:
        from clipper import transcribe as whisper

        token = os.environ.get("HF_TOKEN") if settings.get("diarize") else None
        whisper.transcribe(run.path("audio.wav"), run, model=settings["model"],
                           device=settings["device"], hf_token=token)

    def segment(run: Run, settings: dict) -> dict:
        from clipper import segment as seg
        from clipper.ai import config_from_env

        return seg.run_segment(run, config_from_env(), settings.get("prompt", ""))

    def rate(run: Run, settings: dict, progress: Progress) -> dict:
        from clipper import hardware, rate as laya
        from clipper.device import resolve_device

        device = resolve_device(settings["device"])
        try:
            return laya.run_rate(run, device=device, progress=progress)
        finally:
            hardware.free_device_memory(device)

    def action(run: Run, progress: Progress) -> dict:
        from clipper.stages.action import run_action

        return run_action(run, progress=progress)

    def select(run: Run, settings: dict) -> dict:
        from clipper.selection import run_select

        return run_select(run, top_n=int(settings.get("top_n", 5)))

    def style(run: Run, settings: dict) -> dict:
        from clipper import style as styles

        chosen = styles.validate({**styles.load(), **(settings.get("style") or {})})
        run.write_json("style.json", chosen)
        return chosen

    def fillin(run: Run, chosen_style: dict, progress: Progress) -> dict:
        from clipper import fillin as filler
        from clipper.ai import config_from_env

        wanted = [c for c in run.read_json("selection.json")["clips"]
                  if c.get("include") and c.get("fill_in")]
        fills: dict = {}
        if wanted:
            config = config_from_env()
            if config is None:
                raise ValueError("AI fill-in needs an AI. Set ANTHROPIC_API_KEY (or "
                                 "AI_PROVIDER and AI_MODEL) in .env, or switch fill-in off.")
            transcript = run.read_json("transcript.json")
            for index, clip in enumerate(wanted, start=1):
                fills[clip["id"]] = filler.fill_in(config, clip, transcript,
                                                   chosen_style.get("notes", ""))
                if progress is not None:
                    progress(index, len(wanted))
        run.write_json("fills.json", fills)
        return fills

    def edit(run: Run, chosen_style: dict, fills: dict, progress: Progress) -> list:
        from clipper.edit import run_edit

        return run_edit(run, chosen_style, fills=fills, progress=progress)

    def director(run: Run, chosen_style: dict, done: list, progress: Progress) -> list:
        from clipper import director as graphics
        from clipper.ai import config_from_env

        return graphics.run_director(run, chosen_style, done, config_from_env(), progress)

    return Steps(ingest=ingest, transcribe=transcribe, segment=segment, rate=rate,
                 action=action, select=select, style=style, fillin=fillin, edit=edit,
                 director=director)
