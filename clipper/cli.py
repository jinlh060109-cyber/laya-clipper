from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from clipper.agent import AgentError
from clipper.device import VALID as DEVICES
from clipper.ingest import check_same_source, ingest
from clipper.plan import check_plan
from clipper.preflight import PreflightError, preflight
from clipper.profiles.loader import ProfileError
from clipper.render import run_render
from clipper.run import MissingArtifact, Run, default_run_name
from clipper.stages.action import run_action
from clipper.stages.score import run_score
from clipper.transcribe import transcribe
from clipper.window import write_windows

RUNS_DIR = Path("runs")

DEVICE_HELP = ("Device for Laya. Default: CLIPPER_LAYA_DEVICE, else auto "
               "(cuda -> xpu -> mps -> cpu). Intel Arc needs xpu; Laya's own "
               "auto-detect cannot see it.")


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
    p = sub.add_parser("action"); p.add_argument("run")
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
    p.add_argument("--run"); p.add_argument("--model", default="large-v3")
    p = sub.add_parser("web")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    return parser


def _ingest_and_transcribe(video: Path, name: str | None, model: str,
                           diarize: bool = True) -> Run:
    tools = preflight(require_subtitles=False)
    run = Run.create(RUNS_DIR, name or default_run_name(video))
    check_same_source(run, video)
    if not run.exists("source.json"):
        ingest(tools.ffmpeg, tools.ffprobe, video, run)
    if not run.exists("transcript.json"):
        transcribe(run.path("audio.wav"), run, model=model,
                   hf_token=os.environ.get("HF_TOKEN") if diarize else None)
    return run


def _printer(label: str):
    """One stderr line, rewritten in place, at most every 1% of the work."""
    def show(done: int, total: int) -> None:
        if done == total or done % max(1, total // 100) == 0:
            end = "\n" if done == total else ""
            print(f"\r{label}: {done}/{total}", end=end, file=sys.stderr, flush=True)
    return show


_print_progress = _printer("Scoring windows")
_print_motion = _printer("Measuring motion (s)")


def _report_action(result: dict) -> None:
    if not result["spans"]:
        print("No stretch of 8 s or more without speech; no action moments.")
    else:
        print(f"{result['candidates']} action moments from "
              f"{result['silent_seconds'] / 60:.1f} min without speech")


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

    if args.command == "web":
        import webbrowser

        from clipper.web import server as web_server

        srv = web_server.make_server(RUNS_DIR, port=args.port)
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        print(f"Clipper is running at {url}  (Ctrl+C to stop)")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            srv.server_close()
        return 0

    try:
        if args.command == "ingest":
            tools = preflight(require_subtitles=False)
            video = Path(args.video)
            run = Run.create(RUNS_DIR, args.run or default_run_name(video))
            check_same_source(run, video)
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

        elif args.command == "action":
            run = Run.open(Path(args.run))
            _report_action(run_action(run, progress=_print_motion))

        elif args.command == "score":
            run = Run.open(Path(args.run))
            _report_score(run_score(run, args.profile, device=args.device,
                                    progress=_print_progress))

        elif args.command == "plan":
            run = Run.open(Path(args.run))
            if not args.check:
                print("The plan stage is performed by Claude. "
                      "See .claude/skills/clipper/SKILL.md, then re-run with --check.")
                return 0
            errors, warnings = check_plan(run)
            for error in errors:
                print(f"error: {error}")
            for warning in warnings:
                print(f"warning: {warning}")
            if errors:
                return 1
            print("plan.json looks good." if not warnings
                  else "plan.json will render; review the warnings above.")

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
            _report_action(run_action(run, progress=_print_motion))
            _report_score(run_score(run, args.profile, device=args.device,
                                    progress=_print_progress))
            print(f"Artifacts in {run.root}.")
            print("Next: ask Claude to run the plan stage "
                  "(.claude/skills/clipper/SKILL.md), then `clipper render`.")

    except (ProfileError, MissingArtifact, PreflightError, AgentError,
            ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
