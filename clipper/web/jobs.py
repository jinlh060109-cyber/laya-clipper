from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from clipper.run import Run

STAGES: tuple[str, ...] = ("upload", "ingest", "transcribe", "windows", "profile", "score")


class JobBusy(RuntimeError):
    """A job is already running; the GPU and the run directory are in use."""


class _Stopped(Exception):
    """Internal: a stage failed and the job has already been marked."""


@dataclass
class Stages:
    """The pipeline's stage functions, injected so tests never load real tools."""
    ingest: Callable[[Run, Path], Any]
    transcribe: Callable[[Run, dict], Any]
    windows: Callable[[Run], Any]
    load_agent: Callable[[Run, dict], tuple]
    choose_profile: Callable[[Run, Any], dict]
    score: Callable[[Run, str, dict, "tuple | None", Callable[[int, int], None]], dict]


def default_stages() -> Stages:
    """The real stages. Imports are local so importing this module stays cheap."""
    from clipper.agent import load_agent
    from clipper.classify import choose_profile
    from clipper.ingest import ingest
    from clipper.preflight import preflight
    from clipper.profiles.loader import load_profile
    from clipper.stages.score import run_score
    from clipper.transcribe import transcribe
    from clipper.window import write_windows

    def ingest_stage(run: Run, video: Path) -> None:
        tools = preflight(require_subtitles=False)
        ingest(tools.ffmpeg, tools.ffprobe, video, run)

    def transcribe_stage(run: Run, settings: dict) -> None:
        token = os.environ.get("HF_TOKEN") if settings["diarize"] else None
        transcribe(run.path("audio.wav"), run, model=settings["model"], hf_token=token)

    def load_stage(run: Run, settings: dict) -> tuple:
        language = run.read_json("transcript.json").get("language")
        return load_agent(language, load_profile("core"), device=settings["device"])

    def choose_stage(run: Run, agent) -> dict:
        return choose_profile(run.read_json("windows.json")["windows"], agent)

    def score_stage(run: Run, profile: str, settings: dict, agent, progress) -> dict:
        return run_score(run, profile, device=settings["device"], agent=agent,
                         progress=progress)

    return Stages(ingest_stage, transcribe_stage, write_windows,
                  load_stage, choose_stage, score_stage)


@dataclass
class Job:
    run: Run
    video: Path
    settings: dict
    stages: dict = field(default_factory=lambda: {
        name: ("done" if name == "upload" else "pending") for name in STAGES})
    state: str = "running"
    error: str | None = None
    failed_stage: str | None = None
    profile: str | None = None
    votes: dict | None = None
    fallback: bool = False
    result: dict | None = None
    progress: dict | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None

    def snapshot(self) -> dict:
        return {"state": self.state, "run": self.run.root.name,
                "run_dir": str(self.run.root), "stages": dict(self.stages),
                "error": self.error, "failed_stage": self.failed_stage,
                "profile": self.profile, "votes": self.votes,
                "fallback": self.fallback, "result": self.result,
                "progress": self.progress,
                "started": self.started, "finished": self.finished}


class JobRunner:
    """Runs one pipeline job at a time in a background thread."""

    def __init__(self, stages: Stages | None = None) -> None:
        self._stages = stages
        self._lock = threading.Lock()
        # A separate lock for the job.json write itself: `status()`/`busy()`
        # only ever need `_lock` briefly, but two threads persisting the same
        # run's job.json (a job finishing while the next one's `start()` does
        # its initial write) must not race through the same `.tmp` file.
        self._write_lock = threading.Lock()
        self._job: Job | None = None
        self._thread: threading.Thread | None = None

    @property
    def stages(self) -> Stages:
        if self._stages is None:
            self._stages = default_stages()
        return self._stages

    def busy(self) -> bool:
        with self._lock:
            return self._job is not None and self._job.state == "running"

    def status(self) -> dict:
        with self._lock:
            return self._job.snapshot() if self._job else {"state": "idle"}

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def start(self, run: Run, video: Path, settings: dict) -> None:
        with self._lock:
            if self._job is not None and self._job.state == "running":
                raise JobBusy("A job is already running. Wait for it to finish.")
            previous = self._job
            job = Job(run=run, video=video, settings=dict(settings))
            self._job = job
        try:
            run.write_json("settings.json", job.settings)
            self._persist(job)
        except Exception:
            # The job never actually started: don't leave a phantom "running"
            # job behind (every later start()/upload would then raise
            # JobBusy forever). Restore whatever was there before and let the
            # caller see the error.
            with self._lock:
                if self._job is job:
                    self._job = previous
            raise
        self._thread = threading.Thread(target=self._execute, args=(job,), daemon=True)
        self._thread.start()

    # -- internals ---------------------------------------------------------

    def _persist(self, job: Job) -> None:
        # Hold the write lock across the snapshot *and* the write so two
        # concurrent persists (e.g. a job finishing while the next start()
        # writes its own initial job.json) can't interleave through the same
        # job.json.tmp file.
        with self._write_lock:
            with self._lock:
                snapshot = job.snapshot()
            job.run.write_json("job.json", snapshot)

    def _update(self, job: Job, **changes) -> None:
        with self._lock:
            stage_changes = changes.pop("stage", None)
            if stage_changes:
                job.stages[stage_changes[0]] = stage_changes[1]
            for key, value in changes.items():
                setattr(job, key, value)
        # The in-memory job (read by status()/busy()) is the source of
        # truth and has already been updated above; persisting it to disk
        # is best-effort and must not crash the worker thread. A raised
        # persist failure here still propagates to the caller (_step, or
        # _execute's final "done" update), which is caught by _execute's
        # catch-all and turned into a proper "failed" job via _fail.
        self._persist(job)

    def _fail(self, job: Job, exc: Exception) -> None:
        """Best-effort terminal failure marker: this must never itself raise."""
        with self._lock:
            running = next((s for s, state in job.stages.items() if state == "running"), None)
            job.state = "failed"
            job.error = str(exc) or type(exc).__name__
            if job.failed_stage is None:
                job.failed_stage = running
            job.finished = time.time()
        try:
            self._persist(job)
        except Exception:
            pass

    def _step(self, job: Job, name: str, fn: Callable[[], Any]) -> Any:
        self._update(job, stage=(name, "running"))
        try:
            value = fn()
        except Exception as exc:  # noqa: BLE001 - every failure is reported, not raised
            self._update(job, stage=(name, "failed"), state="failed",
                         error=str(exc) or type(exc).__name__, failed_stage=name,
                         finished=time.time())
            raise _Stopped from exc
        self._update(job, stage=(name, "done"))
        return value

    def _execute(self, job: Job) -> None:
        run, settings = job.run, job.settings
        agent = None
        try:
            # Building the real stages (importing torch etc.) can itself
            # fail, so it must be inside this try -- see the catch-all below.
            stages = self.stages

            self._step(job, "ingest", lambda: stages.ingest(run, job.video))
            self._step(job, "transcribe", lambda: stages.transcribe(run, settings))
            self._step(job, "windows", lambda: stages.windows(run))

            def pick() -> dict:
                nonlocal agent
                if settings["profile"] != "auto":
                    return {"profile": settings["profile"], "votes": None,
                            "fallback": False}
                agent = stages.load_agent(run, settings)
                return stages.choose_profile(run, agent[0])

            picked = self._step(job, "profile", pick)
            self._update(job, profile=picked["profile"], votes=picked.get("votes"),
                         fallback=bool(picked.get("fallback")))
            recorded = {**settings, "profile_chosen": picked["profile"]}
            if settings["profile"] == "auto":
                recorded["profile_votes"] = picked.get("votes")
                recorded["profile_fallback"] = bool(picked.get("fallback"))
            run.write_json("settings.json", recorded)

            def report(done: int, total: int) -> None:
                # In memory only: status() is polled every second, and writing
                # job.json once per window would cost more than the update.
                with self._lock:
                    job.progress = {"stage": "score", "done": done, "total": total}

            result = self._step(job, "score",
                                lambda: stages.score(run, picked["profile"], settings,
                                                     agent, report))
            # Marking "done" is part of the same protected body: if this
            # persist fails too, the catch-all below still ends the job
            # rather than leaving it stuck "running".
            self._update(job, state="done", result=result, finished=time.time())
        except _Stopped:
            return
        except Exception as exc:  # noqa: BLE001 - any escape ends the job, never leaves it "running"
            self._fail(job, exc)
