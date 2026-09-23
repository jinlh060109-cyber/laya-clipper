from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path

PRODUCED_BY = {
    "source.json": "ingest",
    "audio.wav": "ingest",
    "transcript.json": "transcribe",
    "windows.json": "window",
    "scores.json": "score",
    "candidates.json": "score",
    "plan.json": "plan",
}


class MissingArtifact(FileNotFoundError):
    """A required artifact has not been produced yet."""


class Run:
    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def create(cls, base: Path, name: str) -> "Run":
        root = base / name
        root.mkdir(parents=True, exist_ok=True)
        return cls(root)

    @classmethod
    def open(cls, path: Path) -> "Run":
        if not path.is_dir():
            raise MissingArtifact(
                f"No run directory at {path}. Run `clipper ingest <video>` first to create one."
            )
        return cls(path)

    def path(self, name: str) -> Path:
        return self.root / name

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def read_json(self, name: str) -> dict:
        target = self.path(name)
        if not target.exists():
            stage = PRODUCED_BY.get(name, "an earlier stage")
            raise MissingArtifact(
                f"{name} not found in {self.root}. Run `clipper {stage} {self.root}` first."
            )
        with target.open(encoding="utf-8") as handle:
            return json.load(handle)

    def write_json(self, name: str, data: dict) -> None:
        target = self.path(name)
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False)
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)

    def clips_dir(self) -> Path:
        target = self.root / "clips"
        target.mkdir(exist_ok=True)
        return target


def default_run_name(video: Path) -> str:
    # \w keeps letters of every script, so non-Latin names stay distinct
    # instead of all collapsing to "run".
    stem = re.sub(r"[\W_]+", "-", video.stem.lower()).strip("-") or "run"
    return f"{date.today().isoformat()}-{stem}"
