from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import date
from pathlib import Path

from clipper.agent import AgentError
from clipper.device import VALID as DEVICES
from clipper.ingest import ingest
from clipper.plan import check_plan
from clipper.preflight import PreflightError, preflight
from clipper.profiles.loader import ProfileError
from clipper.render import run_render
from clipper.run import MissingArtifact, Run
from clipper.stages.score import run_score
from clipper.transcribe import transcribe
from clipper.window import write_windows

RUNS_DIR = Path("runs")

DEVICE_HELP = ("Device for Laya. Default: CLIPPER_LAYA_DEVICE, else auto "
               "(cuda -> xpu -> mps -> cpu). Intel Arc needs xpu; Laya's own "
               "auto-detect cannot see it.")


def default_run_name(video: Path) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", video.stem.lower()).strip("-") or "run"
    return f"{date.today().isoformat()}-{stem}"


def _load_dotenv() -> None:
    env = Path(".env")
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            # An empty value in .env means "unset", not "set to empty": an empty
            # CLIPPER_LAYA_DEVICE would otherwise mask the auto default.
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clipper")
    sub = parser.add_subparsers(dest="command")

    # --profile is validated by load_profile, not argparse `choices`, so an
    # unknown name exits 1 with the list of available profiles.
    p = sub.add_parser("ingest"); p.add_argument("video"); p.add_argument("--run")
    p = sub.add_parser("transcribe"); p.add_argument("run")
    p.add_argument("--model", default="large-v3"); p.add_argument("--no-diarize", action="store_true")
    p = sub.add_parser("window"); p.add_argument("run")
    p = sub.add_parser("score"); p.add_argument("run")
    p.add_argument("--profile", required=True)
    p.add_argument("--device", default=None, choices=DEVICES, help=DEVICE_HELP)
    p = sub.add_parser("plan"); p.add_argument("run"); p.add_argument("--check", action="store_true")
    p = sub.add_parser("render"); p.add_argument("run")
    p.add_argument("--vertical", action="store_true")
    p.add_argument("--captions", default="burn", choices=["burn", "sidecar", "none"])
    p = sub.add_parser("all"); p.add_argument("video")
    p.add_argument("--profile", required=True)
    p.add_argument("--device", default=None, choices=DEVICES, help=DEVICE_HELP)
    p.add_argument("--run"); p.add_argument("--vertical", action="store_true")
    p.add_argument("--model", default="large-v3")
    return parser


def _ingest_and_transcribe(video: Path, name: str | None, model: str,
                           diarize: bool = True) -> Run:
    tools = preflight(require_subtitles=False)
    run = Run.create(RUNS_DIR, name or default_run_name(video))
    if not run.exists("source.json"):
        ingest(tools.ffmpeg, tools.ffprobe, video, run)
    if not run.exists("transcript.json"):
        transcribe(run.path("audio.wav"), run, model=model,
                   hf_token=os.environ.get("HF_TOKEN") if diarize else None)
    return run


def _report_score(result: dict) -> None:
    print(f"{result['candidates']} candidates, {result['uncertain']} uncertain, "
          f"{result['failed']} failed windows (scored on {result['device']})")


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_usage()
        return 2

    try:
        if args.command == "ingest":
            tools = preflight(require_subtitles=False)
            video = Path(args.video)
            run = Run.create(RUNS_DIR, args.run or default_run_name(video))
            source = ingest(tools.ffmpeg, tools.ffprobe, video, run)
            print(f"{run.root}: {source['duration']:.1f}s, "
                  f"{source['video']['width']}x{source['video']['height']}")

        elif args.command == "transcribe":
            run = Run.open(Path(args.run))
            transcript = transcribe(
                run.path("audio.wav"), run, model=args.model,
                hf_token=None if args.no_diarize else os.environ.get("HF_TOKEN"))
            print(f"{len(transcript['segments'])} segments, "
                  f"diarized={transcript['diarized']}")

        elif args.command == "window":
            run = Run.open(Path(args.run))
            print(f"{len(write_windows(run))} windows")

        elif args.command == "score":
            run = Run.open(Path(args.run))
            _report_score(run_score(run, args.profile, device=args.device))

        elif args.command == "plan":
            run = Run.open(Path(args.run))
            if not args.check:
                print("The plan stage is performed by Claude. "
                      "See .claude/skills/clipper/SKILL.md, then re-run with --check.")
                return 0
            warnings = check_plan(run)
            for warning in warnings:
                print(warning)
            if warnings:
                return 1
            print("plan.json looks good.")

        elif args.command == "render":
            run = Run.open(Path(args.run))
            written = run_render(run, vertical=args.vertical, captions=args.captions)
            for clip in written:
                print(f"{clip['file']}  {clip['duration']}s  {clip.get('title', '')}")

        elif args.command == "all":
            video = Path(args.video)
            run = _ingest_and_transcribe(video, args.run, args.model)
            if not run.exists("windows.json"):
                write_windows(run)
            _report_score(run_score(run, args.profile, device=args.device))
            print(f"Artifacts in {run.root}.")
            print("Next: ask Claude to run the plan stage "
                  "(.claude/skills/clipper/SKILL.md), then `clipper render`.")

    except (ProfileError, MissingArtifact, PreflightError, AgentError,
            ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
