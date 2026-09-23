from __future__ import annotations

from pathlib import Path

from clipper.action import THRESHOLD, WEIGHTS, action_windows, choose_action
from clipper.frames import contact_sheet
from clipper.motion import measure_motion
from clipper.preflight import preflight
from clipper.run import Run
from clipper.silence import silent_spans

MIN_GAP = 8.0


def run_action(run: Run, progress=None) -> dict:
    """Find moments without speech from sound and picture; write action.json.

    The video is decoded for motion only when there is silence to judge, and
    only once per run (motion.json is reused on re-runs).
    """
    source = run.read_json("source.json")
    spans = silent_spans(run.read_json("transcript.json"), source["duration"], MIN_GAP)
    frames = run.path("frames")
    if frames.exists():
        for old in frames.glob("action-*.jpg"):
            old.unlink()

    candidates: list[dict] = []
    if spans:
        ffmpeg = preflight(require_subtitles=False).ffmpeg
        video = Path(source["path"])
        energy = source["energy"]
        if run.exists("motion.json"):
            motion = run.read_json("motion.json")
        else:
            motion = measure_motion(ffmpeg, video, len(energy), progress)
            run.write_json("motion.json", motion)
        windows = action_windows(spans, energy, motion["motion"],
                                 motion["motion_raw"], motion["cuts"])
        candidates = choose_action(windows)
        for candidate in candidates:
            name = f"action-{candidate['id']:02d}.jpg"
            candidate["sheet"] = f"frames/{name}"
            candidate["sheet_times"] = contact_sheet(ffmpeg, video, candidate["start"],
                                                     candidate["end"], frames / name)

    silent = round(sum(end - start for start, end in spans), 1)
    run.write_json("action.json", {"min_gap": MIN_GAP, "silent_seconds": silent,
                                   "spans": spans, "weights": WEIGHTS,
                                   "threshold": THRESHOLD, "candidates": candidates})
    return {"silent_seconds": silent, "spans": len(spans), "candidates": len(candidates)}
