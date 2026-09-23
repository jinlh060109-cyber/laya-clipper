from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from clipper.agent import AgentError
from clipper.ai import AIError
from clipper.device import VALID as DEVICES
from clipper.hardware import detect
from clipper.ingest import check_same_source, ingest
from clipper.pipeline import default_steps
from clipper.preflight import PreflightError, preflight
from clipper.run import MissingArtifact, Run, default_run_name
from clipper.selection import apply_choices
from clipper.stages.action import run_action
from clipper.style import CHOICES
from clipper.transcribe import transcribe

RUNS_DIR = Path("runs")
MODELS = ("large-v3", "medium", "small")
DEVICE_HELP = ("Device for Whisper and Laya. Default: CLIPPER_LAYA_DEVICE, else auto "
               "(cuda -> xpu -> mps -> cpu). `clipper hardware` shows what this machine has.")


def _load_dotenv() -> None:
    env = Path(".env")
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            # An empty value in .env means "unset", not "set to empty".
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clipper")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("analyze", help="Steps 1-6: from a video to a clip selection.")
    p.add_argument("video"); p.add_argument("--run")
    p.add_argument("--device", default=None, choices=DEVICES, help=DEVICE_HELP)
    p.add_argument("--model", default="large-v3", choices=MODELS)
    p.add_argument("--no-diarize", action="store_true")
    p.add_argument("--prompt", default="", help="What you are looking for, e.g. 'useful tips'.")
    p.add_argument("--top", type=int, default=5, help="How many clips to tick.")

    p = sub.add_parser("make", help="Steps 7-10: style, optional AI fill-in, edit.")
    p.add_argument("run")
    p.add_argument("--only", help="Comma-separated clip ids to make, e.g. c1,c3,a0.")
    p.add_argument("--fill-in", action="store_true", help="Let the AI fill in each clip.")
    for key in ("layout", "captions", "caption_case", "encoder"):
        p.add_argument(f"--{key.replace('_', '-')}", dest=key, choices=CHOICES[key])

    p = sub.add_parser("hardware", help="Show the GPUs and video encoders found.")

    p = sub.add_parser("ingest"); p.add_argument("video"); p.add_argument("--run")
    p = sub.add_parser("transcribe"); p.add_argument("run")
    p.add_argument("--model", default="large-v3", choices=MODELS)
    p.add_argument("--no-diarize", action="store_true")
    p.add_argument("--device", default=None, choices=DEVICES, help=DEVICE_HELP)
    p = sub.add_parser("action"); p.add_argument("run")

    p = sub.add_parser("web")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    return parser


def _printer(label: str):
    """One stderr line, rewritten in place, at most every 1% of the work."""
    def show(done: int, total: int) -> None:
        if done == total or done % max(1, total // 100) == 0:
            end = "\n" if done == total else ""
            print(f"\r{label}: {done}/{total}", end=end, file=sys.stderr, flush=True)
    return show


def _report_action(result: dict) -> None:
    if not result["spans"]:
        print("No stretch of 8 s or more without speech; no action moments.")
    else:
        print(f"{result['candidates']} action moments from "
              f"{result['silent_seconds'] / 60:.1f} min without speech")


def _analyze(args) -> int:
    video = Path(args.video)
    run = Run.create(RUNS_DIR, args.run or default_run_name(video))
    check_same_source(run, video)
    steps = default_steps()
    settings = {"device": args.device or "auto", "model": args.model,
                "diarize": not args.no_diarize, "prompt": args.prompt, "top_n": args.top}
    run.write_json("settings.json", settings)
    if not run.exists("source.json"):
        steps.ingest(run, video)
    if not run.exists("transcript.json"):
        steps.transcribe(run, settings)
    segments = steps.segment(run, settings)
    if segments.get("ai_error"):
        print(f"The AI step failed, so clips were cut into chunks: {segments['ai_error']}",
              file=sys.stderr)
    rated = steps.rate(run, settings, _printer("Laya rating clips"))
    _report_action(steps.action(run, _printer("Measuring motion (s)")))
    chosen = steps.select(run, settings)

    ai = segments.get("ai")
    how = f"by {ai['provider']} ({ai['model']})" if ai else "without AI"
    print(f"{len(segments.get('candidates') or [])} clips proposed {how}; "
          f"{rated.get('scored', 0)} rated by Laya on {rated.get('device')}.")
    for clip in chosen.get("clips") or []:
        mark = "[x]" if clip.get("include") else "[ ]"
        print(f"  {mark} {clip['id']:>4}  {clip['start']:7.1f}-{clip['end']:7.1f}s  "
              f"score {clip.get('score', 0):.2f}  {clip.get('category', '')}: "
              f"{clip.get('title_hint', '')}")
    print(f"Artifacts in {run.root}.\nNext: clipper make {run.root} "
          f"[--only c1,c3] [--fill-in]")
    return 0


def _make(args) -> int:
    run = Run.open(Path(args.run))
    selection = run.read_json("selection.json")
    if args.only or args.fill_in:
        wanted = [cid.strip() for cid in args.only.split(",")] if args.only else None
        choices = [{"id": clip["id"],
                    "include": clip["id"] in wanted if wanted is not None else clip["include"],
                    "fill_in": bool(args.fill_in)}
                   for clip in selection["clips"]]
        if wanted is not None:
            known = {clip["id"] for clip in selection["clips"]}
            unknown = [cid for cid in wanted if cid not in known]
            if unknown:
                raise ValueError(f"Unknown clip id(s): {', '.join(unknown)}.")
        apply_choices(run, choices)
    overrides = {key: getattr(args, key) for key in ("layout", "captions",
                                                     "caption_case", "encoder")
                 if getattr(args, key)}
    steps = default_steps()
    chosen = steps.style(run, {"style": overrides})
    fills = steps.fillin(run, chosen, _printer("AI filling in clips"))
    for clip in steps.edit(run, chosen, fills, _printer("Editing clips")):
        print(f"{run.root / clip['final']}  {clip['duration']}s  {clip['title']}")
    print("Each clip folder also has prompt.md and edit.json for an AI editor.")
    return 0


def _hardware() -> int:
    info = detect()
    print("Devices for Whisper and Laya:")
    for device in info["devices"]:
        mark = "yes" if device["available"] else "no "
        print(f"  [{mark}] {device['id']:<5} {device['label']}: {device['detail']}")
        if device.get("hint"):
            print(f"          {device['hint']}")
    print("Video encoders:")
    for encoder in info["encoders"]:
        mark = "yes" if encoder["available"] else "no "
        print(f"  [{mark}] {encoder['id']:<12} {encoder['label']}"
              + (f": {encoder['detail']}" if encoder.get("detail") else ""))
    return 0


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
        if args.command == "analyze":
            return _analyze(args)
        if args.command == "make":
            return _make(args)
        if args.command == "hardware":
            return _hardware()
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
                run.path("audio.wav"), run, model=args.model, device=args.device,
                hf_token=None if args.no_diarize else os.environ.get("HF_TOKEN"))
            print(f"{len(transcript['segments'])} segments, "
                  f"diarized={transcript['diarized']}")
        elif args.command == "action":
            run = Run.open(Path(args.run))
            _report_action(run_action(run, progress=_printer("Measuring motion (s)")))
    except (MissingArtifact, PreflightError, AgentError, AIError,
            ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
