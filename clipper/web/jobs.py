from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from clipper.pipeline import ANALYZE, MAKE, Steps, default_steps
from clipper.run import Run

KINDS = {"analyze": ("upload", *ANALYZE), "make": MAKE}


class JobBusy(RuntimeError):
    """A job is already running; the GPU and the run directory are in use."""


class _Stopped(Exception):
    """Internal: a stage failed and the job has already been marked."""


@dataclass
class Job:
    run: Run
    kind: str
    settings: dict
    video: Path | None = None
    stages: dict = field(default_factory=dict)
    state: str = "running"
    error: str | None = None
    failed_stage: str | None = None
    result: dict | None = None
    progress: dict | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None

    def snapshot(self) -> dict:
        return {"state": self.state, "kind": self.kind, "run": self.run.root.name,
                "run_dir": str(self.run.root), "stages": dict(self.stages),
                "error": self.error, "failed_stage": self.failed_stage,
                "result": self.result, "progress": self.progress,
                "started": self.started, "finished": self.finished}


class JobRunner:
    """Runs one job at a time (analyze or make clips) in a background thread."""

    def __init__(self, steps: Steps | None = None) -> None:
        self._steps = steps
        self._lock = threading.Lock()
        # A separate lock for the job.json write itself, so two threads
        # persisting the same run (a job finishing while the next starts)
        # never race through the same .tmp file.
        self._write_lock = threading.Lock()
        self._job: Job | None = None
        self._thread: threading.Thread | None = None

    @property
    def steps(self) -> Steps:
        if self._steps is None:
            self._steps = default_steps()
        return self._steps

    def busy(self) -> bool:
        with self._lock:
            return self._job is not None and self._job.state == "running"

    def status(self) -> dict:
        with self._lock:
            return self._job.snapshot() if self._job else {"state": "idle"}

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def start(self, run: Run, kind: str, settings: dict, video: Path | None = None) -> None:
        if kind not in KINDS:
            raise ValueError(f"Unknown job kind {kind!r}.")
        if kind == "make" and not run.exists("selection.json"):
            raise ValueError("Analyze this video first; there is no clip selection yet.")
        with self._lock:
            if self._job is not None and self._job.state == "running":
                raise JobBusy("A job is already running. Wait for it to finish.")
            previous = self._job
            stages = {name: "pending" for name in KINDS[kind]}
            if kind == "analyze":
                stages["upload"] = "done"
            job = Job(run=run, kind=kind, settings=dict(settings), video=video, stages=stages)
            self._job = job
        try:
            if kind == "analyze":
                run.write_json("settings.json", job.settings)
            self._persist(job)
        except Exception:
            # The job never started: restore what was there so later starts
            # are not refused by a phantom "running" job.
            with self._lock:
                if self._job is job:
                    self._job = previous
            raise
        self._thread = threading.Thread(target=self._execute, args=(job,), daemon=True)
        self._thread.start()

    # -- internals ---------------------------------------------------------

    def _persist(self, job: Job) -> None:
        with self._write_lock:
            with self._lock:
                snapshot = job.snapshot()
            job.run.write_json("job.json", snapshot)

    def _update(self, job: Job, **changes) -> None:
        with self._lock:
            stage = changes.pop("stage", None)
            if stage:
                job.stages[stage[0]] = stage[1]
            for key, value in changes.items():
                setattr(job, key, value)
        # The in-memory job is the source of truth; a failed persist here
        # propagates to _execute's catch-all, which ends the job as failed.
        self._persist(job)

    def _fail(self, job: Job, exc: Exception) -> None:
        """Best-effort terminal failure marker: this must never itself raise."""
        with self._lock:
            running = next((s for s, st in job.stages.items() if st == "running"), None)
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

    def _reporter(self, job: Job, stage: str) -> Callable[[int, int], None]:
        def report(done: int, total: int) -> None:
            # In memory only: status() is polled every second.
            with self._lock:
                job.progress = {"stage": stage, "done": done, "total": total}
        return report

    def _execute(self, job: Job) -> None:
        try:
            # Building the real steps (importing torch etc.) can itself fail,
            # so it sits inside this try -- see the catch-all below.
            steps = self.steps
            run = job.run
            if job.kind == "analyze":
                result = self._analyze(job, steps, run)
            else:
                result = self._make(job, steps, run)
            self._update(job, state="done", result=result, finished=time.time())
        except _Stopped:
            return
        except Exception as exc:  # noqa: BLE001 - never leave a job "running"
            self._fail(job, exc)

    def _analyze(self, job: Job, steps: Steps, run: Run) -> dict:
        settings = job.settings
        self._step(job, "ingest", lambda: steps.ingest(run, job.video))
        self._step(job, "transcribe", lambda: steps.transcribe(run, settings))
        segments = self._step(job, "segment", lambda: steps.segment(run, settings)) or {}
        rated = self._step(job, "rate", lambda: steps.rate(
            run, settings, self._reporter(job, "rate"))) or {}
        found = self._step(job, "action", lambda: steps.action(
            run, self._reporter(job, "action"))) or {}
        chosen = self._step(job, "select", lambda: steps.select(run, settings)) or {}
        return {"content_type": segments.get("content_type"), "ai": segments.get("ai"),
                "ai_error": segments.get("ai_error"),
                "candidates": len(segments.get("candidates") or []),
                "scored": rated.get("scored", 0), "failed": rated.get("failed", 0),
                "device": rated.get("device"),
                "action_candidates": found.get("candidates", 0),
                "silent_seconds": found.get("silent_seconds", 0.0),
                "kept": sum(1 for c in chosen.get("clips") or [] if c.get("include"))}

    def _make(self, job: Job, steps: Steps, run: Run) -> dict:
        chosen = self._step(job, "style", lambda: steps.style(run, job.settings))
        fills = self._step(job, "fillin", lambda: steps.fillin(
            run, chosen, self._reporter(job, "fillin")))
        done = self._step(job, "edit", lambda: steps.edit(
            run, chosen, fills, self._reporter(job, "edit")))
        return {"clips": done}
