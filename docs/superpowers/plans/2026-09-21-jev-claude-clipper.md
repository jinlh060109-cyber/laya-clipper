# JEV Claude Clipper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a local video file into short, upload-ready clips by transcribing it, scoring every part of the transcript with TypeSafe Jev, letting Claude select and trim, and rendering with ffmpeg.

**Architecture:** Six stages, each a CLI subcommand that reads JSON artifacts from a run directory and writes JSON artifacts back — `ingest → transcribe → window → score → plan → render`. Nothing passes between stages in memory. The `plan` stage is performed by Claude with user editing skills; every other stage is deterministic Python. Almost all logic is pure functions over small JSON, so the test suite runs offline with no API key and no committed media.

**Tech Stack:** Python 3.11+, WhisperX (faster-whisper + wav2vec2 + pyannote 3.x), `typesafe-sdk` (Jev), ffmpeg/ffprobe with libass, PyYAML, `httpx` (LLM adapter), pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-jev-claude-clipper-design.md`

## Global Constraints

- Python 3.11 or later. Type hints on every public function.
- Every stage reads and writes only through the `Run` artifact layer (Task 2). No stage opens another stage's file by raw path.
- Window defaults: **30.0s window, 10.0s step, 20.0s preceding context**. Candidate cap **60.0s**. Clips shorter than **10.0s** are rejected.
- Jev: model `jev-latest` by default; the resolved `response.model` is recorded in `scores.json`. Rate limit ceiling 1,200 req/min; concurrency semaphore of 16.
- Score rubric levels: 5 per `Score` question, normalized via the response `legend`, never hardcoded integers.
- Energy normalization is against a **rolling 300-second baseline**, never whole-file.
- Subtitle files are always staged into a space-free temp directory before any ffmpeg `subtitles=` filter references them. Quoting and backslash escaping do not work; do not attempt them.
- `runs/` and `.env` are gitignored. No API key is ever written into a run artifact.
- Commit after every task. Conventional commit prefixes (`feat:`, `test:`, `fix:`, `chore:`).

## Reference implementation

`https://github.com/op7418/Youtube-clipper-skill` is prior art for the render stage only. Read `scripts/clip_video.py`, `scripts/burn_subtitles.py` and `TECHNICAL_NOTES.md` before Task 15. Check its LICENSE before copying any code verbatim; the gotchas below are facts, not code, and need no attribution:

- `ffmpeg` may lack libass entirely. Verify the *filter*: `ffmpeg -filters | grep subtitles`.
- The `subtitles=` filter truncates paths at the first space. Double quotes, single quotes and backslash escaping all fail. A space-free temp directory is the only fix.
- Subtitle timestamps must be rebased per clip: `adjusted = original - clip_start`, clamped at 0.
- Font size by resolution: 720p → 20, 1080p → 24, 4K → 48.

---

## File Structure

| File | Responsibility |
|---|---|
| `clipper/preflight.py` | Locate ffmpeg/ffprobe; verify the `subtitles` filter exists. |
| `clipper/run.py` | Run directory creation, artifact read/write. The only file that touches run paths. |
| `clipper/energy.py` | Per-second RMS and rolling-baseline normalization. Pure. |
| `clipper/ingest.py` | ffprobe parsing, audio extraction, wiring energy into `source.json`. |
| `clipper/transcribe.py` | WhisperX invocation and transcript normalization. |
| `clipper/window.py` | Transcript + energy → overlapping windows. Pure. |
| `clipper/profiles/loader.py` | Profile YAML schema, `extends` resolution, validation. |
| `clipper/profiles/*.yaml` | `core`, `podcast`, `talking_head`, `lecture`, `stream`. |
| `clipper/jev.py` | Profile questions → SDK question objects; async scoring client. |
| `clipper/rank.py` | Composite score, gates, uncertainty. Pure. |
| `clipper/merge.py` | Window scores → candidates, backward extension. Pure. |
| `clipper/captions.py` | Words → cues, rebasing, SRT and ASS rendering. Pure. |
| `clipper/render.py` | Crop math, filter chain, ffmpeg invocation, temp staging. |
| `clipper/llm.py` | OpenAI-compatible writer adapter. |
| `clipper/cli.py` | Argument parsing, stage dispatch, `all`. |
| `.claude/skills/clipper/SKILL.md` | How Claude performs the `plan` stage. |

---

### Task 1: Project scaffold and ffmpeg preflight

**Files:**
- Create: `pyproject.toml`, `clipper/__init__.py`, `clipper/preflight.py`, `.env.example`
- Test: `tests/test_preflight.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `find_binary(name: str, env_var: str) -> Path`, `has_subtitles_filter(ffmpeg: Path) -> bool`, `preflight() -> Preflight` where `Preflight` is a frozen dataclass with `ffmpeg: Path` and `ffprobe: Path`. Raises `PreflightError` with an actionable message.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "clipper"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "typesafe-sdk",
    "whisperx",
    "pyyaml",
    "httpx",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio"]

[project.scripts]
clipper = "clipper.cli:main"

[tool.pytest.ini_options]
markers = ["live: hits a real API; deselected by default"]
addopts = "-m 'not live'"
asyncio_mode = "auto"
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_preflight.py
import pytest
from pathlib import Path
from clipper.preflight import (
    PreflightError, find_binary, has_subtitles_filter,
)

def test_find_binary_prefers_env_var(tmp_path, monkeypatch):
    fake = tmp_path / "ffmpeg.exe"
    fake.write_text("")
    monkeypatch.setenv("FFMPEG_PATH", str(fake))
    assert find_binary("ffmpeg", "FFMPEG_PATH") == fake

def test_find_binary_raises_actionable_error(monkeypatch):
    monkeypatch.setenv("FFMPEG_PATH", "/nonexistent/ffmpeg")
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(PreflightError) as err:
        find_binary("ffmpeg", "FFMPEG_PATH")
    assert "install" in str(err.value).lower()

def test_has_subtitles_filter_detects_presence(monkeypatch):
    monkeypatch.setattr(
        "clipper.preflight._run_filters",
        lambda _: "... T.. subtitles         V->V       Render text subtitles ...",
    )
    assert has_subtitles_filter(Path("ffmpeg")) is True

def test_has_subtitles_filter_detects_absence(monkeypatch):
    monkeypatch.setattr("clipper.preflight._run_filters", lambda _: "... scale ...")
    assert has_subtitles_filter(Path("ffmpeg")) is False
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_preflight.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clipper.preflight'`

- [ ] **Step 4: Implement `clipper/preflight.py`**

```python
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
```

- [ ] **Step 5: Create `.env.example`**

```
TYPESAFE_API_KEY=sk-...
HF_TOKEN=
WRITER_BASE_URL=
WRITER_API_KEY=
WRITER_MODEL=
FFMPEG_PATH=
FFPROBE_PATH=
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_preflight.py -v`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml clipper/ tests/ .env.example
git commit -m "feat: project scaffold and ffmpeg preflight"
```

---

### Task 2: Run directory and artifact layer

**Files:**
- Create: `clipper/run.py`
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Run` with `Run.create(base: Path, name: str) -> Run`, `Run.open(path: Path) -> Run`, `.root: Path`, `.path(name: str) -> Path`, `.read_json(name: str) -> dict`, `.write_json(name: str, data: dict) -> None`, `.exists(name: str) -> bool`, `.clips_dir() -> Path`. Raises `MissingArtifact` naming the stage that produces it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run.py
import pytest
from clipper.run import Run, MissingArtifact

def test_create_makes_run_directory(tmp_path):
    run = Run.create(tmp_path, "ep47")
    assert run.root == tmp_path / "ep47"
    assert run.root.is_dir()

def test_write_then_read_roundtrips(tmp_path):
    run = Run.create(tmp_path, "ep47")
    run.write_json("source.json", {"duration": 5400.0})
    assert run.read_json("source.json") == {"duration": 5400.0}

def test_missing_artifact_names_producing_stage(tmp_path):
    run = Run.create(tmp_path, "ep47")
    with pytest.raises(MissingArtifact) as err:
        run.read_json("transcript.json")
    assert "transcribe" in str(err.value)

def test_write_is_atomic_leaving_no_partial_file(tmp_path, monkeypatch):
    run = Run.create(tmp_path, "ep47")
    run.write_json("source.json", {"ok": True})
    monkeypatch.setattr(
        "json.dump", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom"))
    )
    with pytest.raises(ValueError):
        run.write_json("source.json", {"ok": False})
    assert run.read_json("source.json") == {"ok": True}

def test_clips_dir_is_created_on_demand(tmp_path):
    run = Run.create(tmp_path, "ep47")
    assert run.clips_dir().is_dir()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clipper.run'`

- [ ] **Step 3: Implement `clipper/run.py`**

```python
from __future__ import annotations

import json
import os
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
            raise MissingArtifact(f"No run directory at {path}")
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_run.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add clipper/run.py tests/test_run.py
git commit -m "feat: run directory and atomic artifact layer"
```

---

### Task 3: Rolling-baseline energy normalization

**Files:**
- Create: `clipper/energy.py`
- Test: `tests/test_energy.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `rolling_baseline(rms: list[float], baseline_seconds: int = 300) -> list[float]` returning values clamped to 0.0–1.0, same length as input.

Spec §4: absolute loudness is a poor signal because a consistently loud speaker saturates everywhere while a spike inside a quiet stretch fails to register. Normalize locally.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_energy.py
import pytest
from clipper.energy import rolling_baseline

def test_output_length_matches_input():
    assert len(rolling_baseline([0.1] * 100, baseline_seconds=10)) == 100

def test_flat_signal_normalizes_to_midpoint():
    out = rolling_baseline([0.5] * 100, baseline_seconds=10)
    assert all(abs(v - 0.5) < 1e-6 for v in out)

def test_spike_in_quiet_stretch_scores_high():
    rms = [0.05] * 60 + [0.9] + [0.05] * 60
    out = rolling_baseline(rms, baseline_seconds=30)
    assert out[60] > 0.9

def test_loud_speaker_does_not_saturate():
    """A uniformly loud signal must not read as one long highlight."""
    out = rolling_baseline([0.95] * 200, baseline_seconds=30)
    assert max(out) < 0.6

def test_values_are_clamped_to_unit_range():
    rms = [0.0] * 50 + [1000.0] + [0.0] * 50
    out = rolling_baseline(rms, baseline_seconds=20)
    assert all(0.0 <= v <= 1.0 for v in out)

def test_empty_input_returns_empty():
    assert rolling_baseline([], baseline_seconds=30) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_energy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clipper.energy'`

- [ ] **Step 3: Implement `clipper/energy.py`**

```python
from __future__ import annotations

import statistics


def rolling_baseline(rms: list[float], baseline_seconds: int = 300) -> list[float]:
    """Normalize per-second RMS against a local window, as a clamped z-score.

    Returns values in 0..1 where 0.5 is "typical for this part of the file".
    """
    if not rms:
        return []
    half = max(1, baseline_seconds // 2)
    out: list[float] = []
    for i in range(len(rms)):
        lo = max(0, i - half)
        hi = min(len(rms), i + half + 1)
        local = rms[lo:hi]
        mean = statistics.fmean(local)
        stdev = statistics.pstdev(local) if len(local) > 1 else 0.0
        if stdev < 1e-9:
            out.append(0.5)
            continue
        z = (rms[i] - mean) / stdev
        out.append(min(1.0, max(0.0, 0.5 + z / 6.0)))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_energy.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add clipper/energy.py tests/test_energy.py
git commit -m "feat: rolling-baseline audio energy normalization"
```

---

### Task 4: Ingest stage

**Files:**
- Create: `clipper/ingest.py`
- Test: `tests/test_ingest.py`, `tests/fixtures/ffprobe_1080p.json`

**Interfaces:**
- Consumes: `preflight()` (Task 1), `Run` (Task 2), `rolling_baseline` (Task 3).
- Produces: `parse_probe(probe: dict, source: Path) -> dict` (pure), `extract_audio(ffmpeg: Path, video: Path, dest: Path) -> None`, `per_second_rms(ffmpeg: Path, wav: Path, duration: float) -> list[float]`, `ingest(video: Path, run: Run) -> dict`.

`source.json` shape: `{"path", "duration", "container", "video": {"codec","width","height","fps","rotation"}, "audio": {"codec","sample_rate","channels"}, "energy": [...]}`.

- [ ] **Step 1: Create the ffprobe fixture**

```json
{
  "format": { "duration": "5400.120000", "format_name": "mov,mp4,m4a,3gp,3g2,mj2" },
  "streams": [
    { "codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080,
      "avg_frame_rate": "30000/1001",
      "side_data_list": [ { "side_data_type": "Display Matrix", "rotation": -90 } ] },
    { "codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2 }
  ]
}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_ingest.py
import json
from pathlib import Path
import pytest
from clipper.ingest import parse_probe

FIXTURE = Path(__file__).parent / "fixtures" / "ffprobe_1080p.json"

def test_parse_probe_extracts_duration_and_streams():
    probe = json.loads(FIXTURE.read_text())
    out = parse_probe(probe, Path("/videos/ep47.mp4"))
    assert out["duration"] == pytest.approx(5400.12)
    assert out["video"]["width"] == 1920
    assert out["video"]["height"] == 1080
    assert out["audio"]["sample_rate"] == 48000
    assert out["path"] == "/videos/ep47.mp4"

def test_parse_probe_converts_fractional_frame_rate():
    probe = json.loads(FIXTURE.read_text())
    assert parse_probe(probe, Path("x.mp4"))["video"]["fps"] == pytest.approx(29.97, abs=0.01)

def test_parse_probe_reads_rotation_side_data():
    """Portrait phone footage must not be cropped sideways at render time."""
    probe = json.loads(FIXTURE.read_text())
    assert parse_probe(probe, Path("x.mp4"))["video"]["rotation"] == -90

def test_parse_probe_defaults_rotation_to_zero():
    probe = {"format": {"duration": "10.0", "format_name": "mp4"},
             "streams": [{"codec_type": "video", "codec_name": "h264",
                          "width": 640, "height": 480, "avg_frame_rate": "25/1"},
                         {"codec_type": "audio", "codec_name": "aac",
                          "sample_rate": "44100", "channels": 1}]}
    assert parse_probe(probe, Path("x.mp4"))["video"]["rotation"] == 0

def test_parse_probe_rejects_file_with_no_audio():
    probe = {"format": {"duration": "10.0", "format_name": "mp4"},
             "streams": [{"codec_type": "video", "codec_name": "h264",
                          "width": 640, "height": 480, "avg_frame_rate": "25/1"}]}
    with pytest.raises(ValueError, match="no audio"):
        parse_probe(probe, Path("x.mp4"))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clipper.ingest'`

- [ ] **Step 4: Implement `clipper/ingest.py`**

```python
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from clipper.energy import rolling_baseline
from clipper.run import Run


def _fps(rate: str) -> float:
    if "/" in rate:
        num, den = rate.split("/", 1)
        return float(num) / float(den) if float(den) else 0.0
    return float(rate)


def _rotation(stream: dict) -> int:
    for side in stream.get("side_data_list", []):
        if "rotation" in side:
            return int(side["rotation"])
    return 0


def parse_probe(probe: dict, source: Path) -> dict:
    streams = probe.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise ValueError(f"{source} has no video stream.")
    if audio is None:
        raise ValueError(f"{source} has no audio stream; there is nothing to transcribe.")
    return {
        "path": source.as_posix(),
        "duration": float(probe["format"]["duration"]),
        "container": probe["format"]["format_name"],
        "video": {
            "codec": video["codec_name"],
            "width": int(video["width"]),
            "height": int(video["height"]),
            "fps": _fps(video.get("avg_frame_rate", "0/1")),
            "rotation": _rotation(video),
        },
        "audio": {
            "codec": audio["codec_name"],
            "sample_rate": int(audio["sample_rate"]),
            "channels": int(audio["channels"]),
        },
    }


def probe_source(ffprobe: Path, video: Path) -> dict:
    result = subprocess.run(
        [str(ffprobe), "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(video)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video}:\n{result.stderr}")
    return parse_probe(json.loads(result.stdout), video)


def extract_audio(ffmpeg: Path, video: Path, dest: Path) -> None:
    result = subprocess.run(
        [str(ffmpeg), "-y", "-i", str(video), "-vn",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed:\n{result.stderr}")


def per_second_rms(ffmpeg: Path, wav: Path, duration: float) -> list[float]:
    """One RMS value per second, via ffmpeg's astats filter."""
    result = subprocess.run(
        [str(ffmpeg), "-v", "info", "-i", str(wav),
         "-af", "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level",
         "-f", "null", "-"],
        capture_output=True, text=True, check=False,
    )
    levels: list[float] = []
    for line in (result.stderr + result.stdout).splitlines():
        if "RMS_level=" in line:
            raw = line.rsplit("=", 1)[1].strip()
            db = -90.0 if raw in {"-inf", "-nan", "nan"} else float(raw)
            levels.append(10 ** (max(db, -90.0) / 20.0))
    expected = max(1, int(duration))
    if len(levels) < expected:
        levels.extend([levels[-1] if levels else 0.0] * (expected - len(levels)))
    return levels[:expected]


def ingest(ffmpeg: Path, ffprobe: Path, video: Path, run: Run) -> dict:
    source = probe_source(ffprobe, video)
    wav = run.path("audio.wav")
    extract_audio(ffmpeg, video, wav)
    raw = per_second_rms(ffmpeg, wav, source["duration"])
    source["energy"] = rolling_baseline(raw)
    run.write_json("source.json", source)
    return source
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_ingest.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add clipper/ingest.py tests/test_ingest.py tests/fixtures/ffprobe_1080p.json
git commit -m "feat: ingest stage with probe parsing and energy extraction"
```

---

### Task 5: Transcribe stage

**Files:**
- Create: `clipper/transcribe.py`
- Test: `tests/test_transcribe.py`, `tests/fixtures/whisperx_result.json`

**Interfaces:**
- Consumes: `Run` (Task 2).
- Produces: `normalize_transcript(result: dict, model: str, diarized: bool, language: str) -> dict` (pure), `transcribe(wav: Path, run: Run, model: str = "large-v3", device: str | None = None, hf_token: str | None = None) -> dict`.

`transcript.json` shape per spec §5: `{"language","model","diarized","segments":[{"start","end","speaker","text","words":[{"word","start","end","score","speaker"}]}]}`.

- [ ] **Step 1: Create the WhisperX fixture**

Word entries deliberately include one with missing timings — WhisperX omits them for words it cannot align, and downstream code must not crash on it.

```json
{
  "segments": [
    { "start": 872.41, "end": 878.90, "text": " So the thing nobody says",
      "speaker": "SPEAKER_01",
      "words": [
        { "word": "So", "start": 872.41, "end": 872.58, "score": 0.91, "speaker": "SPEAKER_01" },
        { "word": "the", "start": 872.60, "end": 872.72, "score": 0.88, "speaker": "SPEAKER_01" },
        { "word": "thing" },
        { "word": "nobody", "start": 873.10, "end": 873.55, "score": 0.93, "speaker": "SPEAKER_01" },
        { "word": "says", "start": 873.58, "end": 873.99, "score": 0.90, "speaker": "SPEAKER_01" }
      ] }
  ]
}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_transcribe.py
import json
from pathlib import Path
import pytest
from clipper.transcribe import normalize_transcript

FIXTURE = Path(__file__).parent / "fixtures" / "whisperx_result.json"

def _result():
    return json.loads(FIXTURE.read_text())

def test_normalize_sets_metadata():
    out = normalize_transcript(_result(), model="large-v3", diarized=True, language="en")
    assert out["model"] == "large-v3"
    assert out["diarized"] is True
    assert out["language"] == "en"

def test_normalize_strips_leading_segment_whitespace():
    out = normalize_transcript(_result(), "large-v3", True, "en")
    assert out["segments"][0]["text"] == "So the thing nobody says"

def test_unaligned_words_are_interpolated_not_dropped():
    """WhisperX omits timings for words it cannot align. Dropping them would
    corrupt caption text; leaving them untimed would crash the ASS renderer."""
    out = normalize_transcript(_result(), "large-v3", True, "en")
    words = out["segments"][0]["words"]
    assert [w["word"] for w in words] == ["So", "the", "thing", "nobody", "says"]
    thing = words[2]
    assert 872.72 <= thing["start"] <= thing["end"] <= 873.10
    assert thing["score"] == 0.0

def test_missing_speaker_defaults_to_speaker_zero():
    result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hi",
                            "words": [{"word": "hi", "start": 0.0, "end": 1.0, "score": 0.9}]}]}
    out = normalize_transcript(result, "large-v3", diarized=False, language="en")
    assert out["segments"][0]["speaker"] == "SPEAKER_00"
    assert out["segments"][0]["words"][0]["speaker"] == "SPEAKER_00"

def test_empty_segments_raise():
    with pytest.raises(ValueError, match="no speech"):
        normalize_transcript({"segments": []}, "large-v3", False, "en")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_transcribe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clipper.transcribe'`

- [ ] **Step 4: Implement `clipper/transcribe.py`**

```python
from __future__ import annotations

import os
from pathlib import Path

from clipper.run import Run

DEFAULT_SPEAKER = "SPEAKER_00"


def _fill_timings(words: list[dict], seg_start: float, seg_end: float) -> list[dict]:
    """Give every word a start and end, interpolating across unaligned runs."""
    filled = [dict(w) for w in words]
    known = [i for i, w in enumerate(filled) if "start" in w and "end" in w]
    if not known:
        span = (seg_end - seg_start) / max(1, len(filled))
        for i, w in enumerate(filled):
            w["start"] = seg_start + i * span
            w["end"] = seg_start + (i + 1) * span
            w["score"] = 0.0
        return filled
    for i, word in enumerate(filled):
        if "start" in word and "end" in word:
            continue
        prev = max((k for k in known if k < i), default=None)
        nxt = min((k for k in known if k > i), default=None)
        lo = filled[prev]["end"] if prev is not None else seg_start
        hi = filled[nxt]["start"] if nxt is not None else seg_end
        gap = [j for j in range(len(filled)) if "start" not in filled[j]
               and (prev is None or j > prev) and (nxt is None or j < nxt)]
        share = (hi - lo) / max(1, len(gap))
        slot = gap.index(i)
        word["start"] = lo + slot * share
        word["end"] = lo + (slot + 1) * share
        word["score"] = 0.0
    return filled


def normalize_transcript(result: dict, model: str, diarized: bool, language: str) -> dict:
    segments = result.get("segments") or []
    if not segments:
        raise ValueError(
            "Transcription produced no speech. The audio may be silent or music only."
        )
    out_segments = []
    for seg in segments:
        speaker = seg.get("speaker", DEFAULT_SPEAKER)
        words = _fill_timings(seg.get("words", []), float(seg["start"]), float(seg["end"]))
        out_segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "speaker": speaker,
            "text": seg.get("text", "").strip(),
            "words": [{
                "word": w["word"],
                "start": float(w["start"]),
                "end": float(w["end"]),
                "score": float(w.get("score", 0.0)),
                "speaker": w.get("speaker", speaker),
            } for w in words],
        })
    return {"language": language, "model": model,
            "diarized": diarized, "segments": out_segments}


def transcribe(wav: Path, run: Run, model: str = "large-v3",
               device: str | None = None, hf_token: str | None = None) -> dict:
    import torch
    import whisperx

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = "float16" if device == "cuda" else "int8"

    audio = whisperx.load_audio(str(wav))
    asr = whisperx.load_model(model, device, compute_type=compute_type)
    result = asr.transcribe(audio, batch_size=16)
    language = result["language"]

    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(result["segments"], align_model, metadata,
                            audio, device, return_char_alignments=False)

    token = hf_token or os.environ.get("HF_TOKEN")
    diarized = False
    if token:
        from whisperx.diarize import DiarizationPipeline
        pipeline = DiarizationPipeline(use_auth_token=token, device=device)
        result = whisperx.assign_word_speakers(pipeline(audio), result)
        diarized = True
    else:
        print("HF_TOKEN not set; skipping diarization. All speech labelled SPEAKER_00.")

    transcript = normalize_transcript(result, model, diarized, language)
    run.write_json("transcript.json", transcript)
    return transcript
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_transcribe.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add clipper/transcribe.py tests/test_transcribe.py tests/fixtures/whisperx_result.json
git commit -m "feat: transcribe stage with WhisperX and transcript normalization"
```

---

### Task 6: Windowing

**Files:**
- Create: `clipper/window.py`
- Test: `tests/test_window.py`

**Interfaces:**
- Consumes: `transcript.json` (Task 5), `source.json` (Task 4).
- Produces: `build_windows(transcript: dict, energy: list[float], duration: float, window_seconds: float = 30.0, step_seconds: float = 10.0, context_seconds: float = 20.0) -> list[dict]`.

Window dict keys: `id, start, end, text, preceding, position, energy_mean, energy_peak, energy_peak_offset`.

Spec §6: `energy_peak_offset` is the fraction of the window at which peak energy falls. Merging uses it to extend candidates *backwards*, because reactions lag their cause.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_window.py
import pytest
from clipper.window import build_windows

def _transcript(n_words: int = 300, start: float = 0.0, wps: float = 3.0) -> dict:
    words = [{"word": f"w{i}", "start": start + i / wps, "end": start + (i + 0.9) / wps,
              "score": 0.9, "speaker": "SPEAKER_00" if i % 60 < 30 else "SPEAKER_01"}
             for i in range(n_words)]
    return {"language": "en", "model": "large-v3", "diarized": True,
            "segments": [{"start": words[0]["start"], "end": words[-1]["end"],
                          "speaker": "SPEAKER_00", "text": " ".join(w["word"] for w in words),
                          "words": words}]}

def test_windows_step_by_step_seconds():
    w = build_windows(_transcript(), [0.5] * 100, duration=100.0)
    assert w[1]["start"] - w[0]["start"] == pytest.approx(10.0, abs=1.5)

def test_windows_are_about_window_seconds_long():
    w = build_windows(_transcript(), [0.5] * 100, duration=100.0)
    assert all(20.0 <= x["end"] - x["start"] <= 34.0 for x in w)

def test_window_never_starts_mid_word():
    t = _transcript()
    starts = {round(word["start"], 3)
              for word in t["segments"][0]["words"]}
    w = build_windows(t, [0.5] * 100, duration=100.0)
    assert all(round(x["start"], 3) in starts for x in w)

def test_preceding_context_is_populated_after_the_first_window():
    w = build_windows(_transcript(), [0.5] * 100, duration=100.0)
    assert w[0]["preceding"] == ""
    assert len(w[3]["preceding"]) > 0

def test_text_carries_speaker_labels():
    w = build_windows(_transcript(), [0.5] * 100, duration=100.0)
    assert "SPEAKER_00:" in w[0]["text"]

def test_energy_peak_offset_locates_the_spike():
    """A spike late in the window must report an offset near 1.0, so merging
    can extend the candidate backwards to the line that caused the reaction."""
    energy = [0.1] * 100
    energy[28] = 0.99
    w = build_windows(_transcript(), energy, duration=100.0)
    first = w[0]
    assert first["energy_peak"] == pytest.approx(0.99)
    assert first["energy_peak_offset"] > 0.85

def test_position_is_fractional_through_the_source():
    w = build_windows(_transcript(n_words=900), [0.5] * 300, duration=300.0)
    assert w[0]["position"] < 0.1
    assert w[-1]["position"] > 0.7

def test_ids_are_stable_and_sequential():
    w = build_windows(_transcript(), [0.5] * 100, duration=100.0)
    assert [x["id"] for x in w] == list(range(len(w)))

def test_short_source_yields_one_window():
    t = _transcript(n_words=20)
    assert len(build_windows(t, [0.5] * 10, duration=7.0)) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_window.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clipper.window'`

- [ ] **Step 3: Implement `clipper/window.py`**

```python
from __future__ import annotations


def _flatten(transcript: dict) -> list[dict]:
    words: list[dict] = []
    for seg in transcript["segments"]:
        words.extend(seg["words"])
    return words


def _render(words: list[dict]) -> str:
    """Words to speaker-labelled text, emitting a label only on speaker change."""
    lines: list[str] = []
    current: str | None = None
    buffer: list[str] = []
    for word in words:
        if word["speaker"] != current:
            if buffer:
                lines.append(f"{current}: {' '.join(buffer)}")
            current = word["speaker"]
            buffer = []
        buffer.append(word["word"])
    if buffer:
        lines.append(f"{current}: {' '.join(buffer)}")
    return "\n".join(lines)


def build_windows(transcript: dict, energy: list[float], duration: float,
                  window_seconds: float = 30.0, step_seconds: float = 10.0,
                  context_seconds: float = 20.0) -> list[dict]:
    words = _flatten(transcript)
    if not words:
        return []

    windows: list[dict] = []
    cursor = 0
    index = 0
    while cursor < len(words):
        anchor = words[cursor]["start"]
        inside = [w for w in words if anchor <= w["start"] < anchor + window_seconds]
        if not inside:
            break
        start, end = inside[0]["start"], inside[-1]["end"]
        context = [w for w in words if anchor - context_seconds <= w["start"] < anchor]

        lo, hi = int(start), max(int(start) + 1, int(end))
        slice_ = energy[lo:hi] or [0.0]
        peak = max(slice_)
        span = max(1e-6, end - start)
        offset = (lo + slice_.index(peak) - start) / span

        windows.append({
            "id": index,
            "start": start,
            "end": end,
            "text": _render(inside),
            "preceding": _render(context),
            "position": start / duration if duration else 0.0,
            "energy_mean": sum(slice_) / len(slice_),
            "energy_peak": peak,
            "energy_peak_offset": min(1.0, max(0.0, offset)),
        })
        index += 1

        if end >= words[-1]["end"]:
            break
        nxt = next((i for i, w in enumerate(words) if w["start"] >= anchor + step_seconds), None)
        if nxt is None or nxt <= cursor:
            break
        cursor = nxt
    return windows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_window.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add clipper/window.py tests/test_window.py
git commit -m "feat: transcript windowing with energy peak offsets"
```

---

### Task 7: Profile schema, loader, and the five profile files

**Files:**
- Create: `clipper/profiles/__init__.py`, `clipper/profiles/loader.py`, `clipper/profiles/core.yaml`, `clipper/profiles/podcast.yaml`, `clipper/profiles/talking_head.yaml`, `clipper/profiles/lecture.yaml`, `clipper/profiles/stream.yaml`
- Test: `tests/test_profiles.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Profile` frozen dataclass with `name: str`, `questions: dict[str, dict]`, `weights: dict[str, float]`, `penalties: dict[str, float]`, `gates: dict[str, float]`, `thresholds: dict[str, float]`, `merge: dict[str, float]`, `window: dict[str, float]`. Loader: `load_profile(name: str) -> Profile`, `available_profiles() -> list[str]`. Raises `ProfileError`.

Question entries are `{"type": "score"|"choice"|"noul", "instructions": str, "criteria": ...}` — list for score, dict for choice, `{"true","false"}` dict for noul.

- [ ] **Step 1: Write `clipper/profiles/core.yaml`**

The Hook / Flow / Value bundle plus failure modes from spec §7.

```yaml
name: core
window:
  seconds: 30.0
  step: 10.0
  context: 20.0
questions:
  clipworthy:
    type: score
    instructions: >
      How likely a stranger scrolling past would stop and watch this segment,
      judged on the segment itself rather than the surrounding episode.
    criteria:
      - "Filler. Logistics, small talk, or a thought going nowhere."
      - "Mildly interesting to an existing fan, dull to anyone else."
      - "Solid. One clear idea or moment worth hearing."
      - "Strong. A stranger would finish it and might share it."
      - "Exceptional. A stranger would stop scrolling within a second."
  hook_strength:
    type: score
    instructions: >
      How well the FIRST sentence of this segment works as the opening line of
      a short video. Judge only the opening, not the whole segment.
    criteria:
      - "Dead open. Filler words, logistics, or mid-admin chatter."
      - "Slow. Understandable but gives no reason to keep watching."
      - "Adequate. States something concrete."
      - "Strong. A claim, question, or image that demands the next sentence."
      - "Arresting. Impossible to scroll past."
  hook_type:
    type: choice
    instructions: What kind of opening this segment's first sentence uses.
    criteria:
      pattern_interrupt: "Something unexpected or contradictory that breaks attention."
      direct_promise: "States a specific outcome the viewer will get."
      question: "Poses a question the viewer wants answered."
      contradiction: "Contradicts common wisdom, then promises the explanation."
      story_open: "Begins a narrative with a clear situation and stakes."
      none: "No recognisable hook; the segment simply starts."
  open_loop:
    type: noul
    instructions: The opening raises a question the viewer needs resolved.
    criteria:
      true: "A curiosity gap is opened and not immediately closed."
      false: "Everything is stated flatly with nothing left pending."
  self_contained:
    type: noul
    instructions: This segment is understandable to someone who has heard nothing before it.
    criteria:
      true: "A new listener follows it completely."
      false: "A new listener would be lost or confused."
  needs_context:
    type: noul
    instructions: >
      The segment refers to something discussed earlier that is not present in
      the segment itself.
    criteria:
      true: "Contains unexplained references such as 'that thing we mentioned'."
      false: "Every reference is explained inside the segment."
  ends_cleanly:
    type: noul
    instructions: The thought reaches a natural conclusion before the segment ends.
    criteria:
      true: "The point lands and completes."
      false: "It cuts off mid-thought or trails away."
  payoff:
    type: noul
    instructions: The segment delivers on what its opening implies.
    criteria:
      true: "The promise made at the start is met by the end."
      false: "It sets something up and never resolves it."
  poster_line:
    type: noul
    instructions: >
      The segment contains at least one sentence you could put on a poster:
      quotable, self-standing, and memorable on its own.
    criteria:
      true: "There is a line worth quoting verbatim."
      false: "Nothing here would survive being pulled out of context."
  concrete_specifics:
    type: noul
    instructions: The segment contains a specific number, name, date, or worked example.
    criteria:
      true: "Contains concrete specifics rather than generalities."
      false: "Entirely abstract or generic."
  audible_reaction:
    type: noul
    instructions: >
      Someone in the recording reacts: laughter, an interjection, a sharp
      agreement or disagreement, an audible pause of surprise.
    criteria:
      true: "A visible human reaction occurs in the transcript."
      false: "Even, uninterrupted delivery."
  emotional_intensity:
    type: score
    instructions: The emotional charge of the segment, whatever its direction.
    criteria:
      - "Flat and procedural."
      - "Mild interest or warmth."
      - "Clearly engaged."
      - "Strong feeling: excitement, anger, grief, delight."
      - "Peak intensity."
  clip_format:
    type: choice
    instructions: Which short-form clip archetype this segment best fits.
    criteria:
      cold_open: "Opens on a provocative line with no setup."
      debate: "Tight back-and-forth disagreement between two people."
      how_to: "A specific, useful method explained."
      story_arc: "Setup, turn and payoff."
      hot_take: "A single bold claim stated plainly."
      confession: "An honest or vulnerable admission."
      list: "A numbered or enumerated run of points."
      filler: "None of the above; not clip material."
  opens_with_windup:
    type: noul
    instructions: >
      The first sentence is setup, backstory, throat-clearing, or a greeting
      rather than the point itself.
    criteria:
      true: "Begins with a wind-up such as 'so, um, basically' or a long preamble."
      false: "Begins directly on substance."
  buried_lede:
    type: noul
    instructions: >
      The most interesting moment arrives well after the segment starts, so the
      opening should be trimmed later.
    criteria:
      true: "The best line is in the second half."
      false: "The best material is at or near the start."
weights:
  clipworthy: 0.45
  hook_strength: 0.30
  poster_line: 0.08
  open_loop: 0.06
  payoff: 0.05
  concrete_specifics: 0.03
  audible_reaction: 0.03
penalties:
  needs_context: 0.20
  opens_with_windup: 0.15
gates:
  self_contained: 0.35
thresholds:
  candidate: 0.55
  uncertain_confidence: 0.55
merge:
  max_seconds: 60.0
  backward_extend_seconds: 8.0
  backward_extend_offset: 0.6
```

- [ ] **Step 2: Write the four content profiles**

```yaml
# clipper/profiles/podcast.yaml
name: podcast
extends: core
questions:
  disagreement:
    type: noul
    instructions: Two speakers genuinely disagree, rather than politely agreeing.
    criteria:
      true: "A real difference of view is voiced."
      false: "Agreement, or one speaker only."
  personal_story:
    type: noul
    instructions: A speaker recounts something that happened to them personally.
    criteria:
      true: "A first-hand anecdote with specifics."
      false: "Abstract discussion."
  needs_speaker_id:
    type: noul
    instructions: >
      Following this segment requires knowing who is speaking, so an on-screen
      speaker label is needed.
    criteria:
      true: "Two or more voices alternate and identity matters."
      false: "A single voice, or identity is irrelevant."
weights:
  disagreement: 0.06
  personal_story: 0.05
```

```yaml
# clipper/profiles/talking_head.yaml
name: talking_head
extends: core
questions:
  hot_take:
    type: noul
    instructions: The speaker states an opinion that some of the audience would argue with.
    criteria:
      true: "A contestable claim stated plainly."
      false: "Uncontroversial or purely descriptive."
  direct_address:
    type: noul
    instructions: The speaker addresses the viewer directly, as 'you'.
    criteria:
      true: "Speaks to the viewer."
      false: "Speaks about a subject in the abstract."
weights:
  hot_take: 0.08
  direct_address: 0.04
```

```yaml
# clipper/profiles/lecture.yaml
name: lecture
extends: core
questions:
  complete_explanation:
    type: noul
    instructions: The segment explains one idea from beginning to end.
    criteria:
      true: "Setup, explanation and conclusion are all present."
      false: "Only part of an explanation."
  one_concept:
    type: noul
    instructions: The segment covers exactly one concept rather than several partial ones.
    criteria:
      true: "A single idea, fully handled."
      false: "Two or more ideas, none finished."
  actionable:
    type: noul
    instructions: A viewer could do something differently after watching this.
    criteria:
      true: "Contains a usable instruction or technique."
      false: "Purely conceptual."
  requires_visual:
    type: noul
    instructions: >
      The segment is incomprehensible without seeing slides, a screen, or a
      diagram referred to in the speech.
    criteria:
      true: "Refers to something on screen, such as 'as you can see here'."
      false: "Works as audio alone."
weights:
  complete_explanation: 0.08
  one_concept: 0.06
  actionable: 0.05
penalties:
  requires_visual: 0.12
```

```yaml
# clipper/profiles/stream.yaml
name: stream
extends: core
questions:
  reaction_spike:
    type: noul
    instructions: Something happens that the streamer reacts to strongly.
    criteria:
      true: "A shout, a sudden shift in delivery, an exclamation."
      false: "Even, steady commentary."
  narratable:
    type: noul
    instructions: >
      What happened is clear from the speech alone, without needing to see the
      game state.
    criteria:
      true: "The speech describes what occurred."
      false: "Only makes sense if you can see the screen."
  needs_gameplay_context:
    type: noul
    instructions: Understanding requires knowing the game situation beforehand.
    criteria:
      true: "Depends on prior game state."
      false: "Self-explanatory."
weights:
  reaction_spike: 0.10
  narratable: 0.06
penalties:
  needs_gameplay_context: 0.12
```

- [ ] **Step 3: Write the failing test**

```python
# tests/test_profiles.py
import pytest
from clipper.profiles.loader import (
    Profile, ProfileError, available_profiles, load_profile,
)

def test_all_four_content_profiles_are_available():
    assert set(available_profiles()) == {"podcast", "talking_head", "lecture", "stream"}

def test_core_is_not_offered_as_a_content_profile():
    """core.yaml is a base to extend, not something you run."""
    assert "core" not in available_profiles()

def test_profile_inherits_core_questions():
    p = load_profile("podcast")
    assert "clipworthy" in p.questions
    assert "disagreement" in p.questions

def test_profile_merges_weights_over_core():
    p = load_profile("podcast")
    assert p.weights["clipworthy"] == pytest.approx(0.45)
    assert p.weights["disagreement"] == pytest.approx(0.06)

def test_profile_merges_penalties_over_core():
    p = load_profile("lecture")
    assert p.penalties["needs_context"] == pytest.approx(0.20)
    assert p.penalties["requires_visual"] == pytest.approx(0.12)

def test_window_defaults_match_spec():
    p = load_profile("podcast")
    assert p.window["seconds"] == 30.0
    assert p.window["step"] == 10.0
    assert p.window["context"] == 20.0

def test_every_score_question_has_five_levels():
    for name in available_profiles():
        for key, q in load_profile(name).questions.items():
            if q["type"] == "score":
                assert len(q["criteria"]) == 5, f"{name}.{key}"

def test_every_weighted_key_is_a_real_question():
    for name in available_profiles():
        p = load_profile(name)
        for key in list(p.weights) + list(p.penalties) + list(p.gates):
            assert key in p.questions, f"{name}: {key} is weighted but never asked"

def test_choice_criteria_stay_under_cardinality_limit():
    for name in available_profiles():
        for q in load_profile(name).questions.values():
            if q["type"] == "choice":
                assert len(q["criteria"]) <= 255

def test_unknown_profile_lists_the_valid_ones():
    with pytest.raises(ProfileError) as err:
        load_profile("vlog")
    assert "podcast" in str(err.value)

def test_unknown_question_type_is_rejected(tmp_path, monkeypatch):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\nquestions:\n  x:\n    type: ordinal\n    instructions: hi\n")
    monkeypatch.setattr("clipper.profiles.loader.PROFILE_DIR", tmp_path)
    with pytest.raises(ProfileError, match="ordinal"):
        load_profile("bad")
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_profiles.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clipper.profiles.loader'`

- [ ] **Step 5: Implement `clipper/profiles/loader.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROFILE_DIR = Path(__file__).parent
BASE_ONLY = {"core"}
VALID_TYPES = {"score", "choice", "noul"}


class ProfileError(ValueError):
    """A profile file is missing or malformed."""


@dataclass(frozen=True)
class Profile:
    name: str
    questions: dict[str, dict]
    weights: dict[str, float] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)
    gates: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    merge: dict[str, float] = field(default_factory=dict)
    window: dict[str, float] = field(default_factory=dict)


def available_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILE_DIR.glob("*.yaml") if p.stem not in BASE_ONLY)


def _read(name: str) -> dict:
    path = PROFILE_DIR / f"{name}.yaml"
    if not path.exists():
        raise ProfileError(
            f"Unknown profile '{name}'. Available: {', '.join(available_profiles())}."
        )
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _validate(name: str, questions: dict[str, dict]) -> None:
    for key, question in questions.items():
        qtype = question.get("type")
        if qtype not in VALID_TYPES:
            raise ProfileError(
                f"{name}.{key}: unknown question type '{qtype}'. "
                f"Expected one of {', '.join(sorted(VALID_TYPES))}."
            )
        criteria = question.get("criteria")
        if qtype == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ProfileError(f"{name}.{key}: score criteria must be a list of 2-10 levels.")
        elif qtype == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise ProfileError(f"{name}.{key}: choice criteria must be a non-empty mapping.")
            if len(criteria) > 255:
                raise ProfileError(f"{name}.{key}: choice exceeds Jev's 255-option limit.")


def load_profile(name: str) -> Profile:
    data = _read(name)
    parent = data.get("extends")
    base = _read(parent) if parent else {}

    def merged(key: str) -> dict:
        out = dict(base.get(key) or {})
        out.update(data.get(key) or {})
        return out

    questions = merged("questions")
    _validate(name, questions)
    return Profile(
        name=data.get("name", name),
        questions=questions,
        weights=merged("weights"),
        penalties=merged("penalties"),
        gates=merged("gates"),
        thresholds=merged("thresholds"),
        merge=merged("merge"),
        window=merged("window"),
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_profiles.py -v`
Expected: 11 passed

- [ ] **Step 7: Commit**

```bash
git add clipper/profiles/ tests/test_profiles.py
git commit -m "feat: profile schema, loader, and four content profiles"
```

---

### Task 8: Jev question building and async scoring client

**Files:**
- Create: `clipper/jev.py`
- Test: `tests/test_jev.py`

**Interfaces:**
- Consumes: `Profile` (Task 7), window dicts (Task 6).
- Produces: `build_questions(profile: Profile) -> dict[str, object]`, `window_state(window: dict, profile_name: str) -> dict`, `answers_to_dict(response) -> dict[str, dict]`, `async score_windows(windows, profile, concurrency=16, client=None) -> tuple[list[dict], str]` returning per-window score records and the resolved model string.

Score record: `{"id", "answers": {...}, "failed": bool}`. Each answer is `{"type", "value", "confidence", "legend"}` where `value` is the float or the chosen string.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_jev.py
import pytest
from typesafe_sdk import Choice, Noul, Score
from clipper.jev import answers_to_dict, build_questions, window_state
from clipper.profiles.loader import load_profile

class FakeNoul:
    type = "noul"
    def __init__(self, v): self.noul = v

class FakeChoice:
    type = "choice"
    def __init__(self, c, conf): self.choice, self.confidence = c, conf
    probabilities = {"hot_take": 0.7, "filler": 0.3}

class FakeScore:
    type = "score"
    def __init__(self, s, conf): self.score, self.confidence = s, conf
    legend = {1: "a", 2: "b", 3: "c", 4: "d", 5: "e"}
    probabilities = {1: 0.1, 2: 0.2, 3: 0.4, 4: 0.2, 5: 0.1}

class FakeResponse:
    model = "jev-1.13.0"
    def __init__(self, answers): self.answers = answers

def test_build_questions_maps_each_type_to_its_sdk_class():
    q = build_questions(load_profile("podcast"))
    assert isinstance(q["clipworthy"], Score)
    assert isinstance(q["clip_format"], Choice)
    assert isinstance(q["poster_line"], Noul)

def test_build_questions_covers_every_profile_question():
    profile = load_profile("lecture")
    assert set(build_questions(profile)) == set(profile.questions)

def test_window_state_carries_energy_and_position():
    window = {"id": 3, "start": 872.4, "end": 902.1, "text": "SPEAKER_01: hi",
              "preceding": "earlier", "position": 0.43, "energy_mean": 0.44,
              "energy_peak": 0.86, "energy_peak_offset": 0.71}
    state = window_state(window, "podcast")
    assert state["profile"] == "podcast"
    assert state["window"]["text"] == "SPEAKER_01: hi"
    assert state["energy_peak_offset"] == 0.71
    assert "id" not in state

def test_answers_to_dict_normalizes_score_against_its_legend():
    """Rubric integers are the model's business, not ours; normalize via legend."""
    out = answers_to_dict(FakeResponse({"clipworthy": FakeScore(4.0, 0.9)}))
    assert out["clipworthy"]["value"] == pytest.approx(0.75)
    assert out["clipworthy"]["raw"] == pytest.approx(4.0)
    assert out["clipworthy"]["confidence"] == pytest.approx(0.9)

def test_answers_to_dict_passes_noul_through_unchanged():
    out = answers_to_dict(FakeResponse({"poster_line": FakeNoul(0.82)}))
    assert out["poster_line"]["value"] == pytest.approx(0.82)
    assert out["poster_line"]["confidence"] == pytest.approx(0.82)

def test_answers_to_dict_keeps_choice_label_and_confidence():
    out = answers_to_dict(FakeResponse({"clip_format": FakeChoice("hot_take", 0.66)}))
    assert out["clip_format"]["value"] == "hot_take"
    assert out["clip_format"]["confidence"] == pytest.approx(0.66)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_jev.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clipper.jev'`

- [ ] **Step 3: Implement `clipper/jev.py`**

```python
from __future__ import annotations

import asyncio

from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score, TypeSafeAPIError

from clipper.profiles.loader import Profile


def build_questions(profile: Profile) -> dict[str, object]:
    questions: dict[str, object] = {}
    for key, spec in profile.questions.items():
        instructions = spec.get("instructions", "")
        criteria = spec.get("criteria")
        if spec["type"] == "score":
            questions[key] = Score(instructions=instructions, criteria=list(criteria))
        elif spec["type"] == "choice":
            questions[key] = Choice(instructions=instructions, criteria=dict(criteria))
        else:
            questions[key] = Noul(instructions=instructions,
                                  criteria=dict(criteria) if criteria else None)
    return questions


def window_state(window: dict, profile_name: str) -> dict:
    return {
        "profile": profile_name,
        "window": {
            "start": window["start"],
            "end": window["end"],
            "text": window["text"],
        },
        "preceding": window["preceding"],
        "position": window["position"],
        "energy_mean": window["energy_mean"],
        "energy_peak": window["energy_peak"],
        "energy_peak_offset": window["energy_peak_offset"],
    }


def answers_to_dict(response) -> dict[str, dict]:
    """Flatten SDK answers into plain JSON, normalizing scores to 0..1.

    Score levels are normalized against the response legend rather than assumed
    to be 1..5, so a rubric change cannot silently shift every composite score.
    """
    out: dict[str, dict] = {}
    for key, answer in response.answers.items():
        if answer.type == "noul":
            out[key] = {"type": "noul", "value": float(answer.noul),
                        "confidence": float(answer.noul)}
        elif answer.type == "choice":
            out[key] = {"type": "choice", "value": answer.choice,
                        "confidence": float(answer.confidence),
                        "probabilities": dict(answer.probabilities)}
        else:
            levels = [int(k) for k in answer.legend]
            lo, hi = min(levels), max(levels)
            span = (hi - lo) or 1
            out[key] = {"type": "score",
                        "value": (float(answer.score) - lo) / span,
                        "raw": float(answer.score),
                        "confidence": float(answer.confidence)}
    return out


async def score_windows(windows: list[dict], profile: Profile,
                        concurrency: int = 16, client=None) -> tuple[list[dict], str]:
    questions = build_questions(profile)
    semaphore = asyncio.Semaphore(concurrency)
    model = "unknown"
    owned = client is None
    client = client or AsyncTypeSafeClient()

    async def one(window: dict) -> dict:
        nonlocal model
        async with semaphore:
            try:
                response = await client.system_one(
                    window_state(window, profile.name), questions
                )
            except TypeSafeAPIError as error:
                return {"id": window["id"], "failed": True,
                        "error": f"{error.status} {error.request_id}"}
            model = response.model
            return {"id": window["id"], "failed": False,
                    "answers": answers_to_dict(response)}

    try:
        results = await asyncio.gather(*(one(w) for w in windows))
    finally:
        if owned:
            await client.close()
    return list(results), model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_jev.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add clipper/jev.py tests/test_jev.py
git commit -m "feat: Jev question building and async window scoring"
```

---

### Task 9: Composite scoring and gates

**Files:**
- Create: `clipper/rank.py`
- Test: `tests/test_rank.py`

**Interfaces:**
- Consumes: score records (Task 8), `Profile` (Task 7).
- Produces: `composite(answers: dict, profile: Profile) -> float`, `passes_gate(answers: dict, profile: Profile) -> bool`, `is_uncertain(answers: dict, profile: Profile) -> bool`, `rank(scores: list[dict], profile: Profile) -> list[dict]` adding `composite`, `gated`, `uncertain` to each record.

Spec §7: `buried_lede` annotates but never penalizes — it describes a fixable in-point, not bad material.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rank.py
import pytest
from clipper.profiles.loader import load_profile
from clipper.rank import composite, is_uncertain, passes_gate, rank

PROFILE = load_profile("podcast")

def _answers(**overrides) -> dict:
    base = {k: {"type": "noul", "value": 0.0, "confidence": 0.9}
            for k in PROFILE.questions}
    for key in ("clipworthy", "hook_strength", "emotional_intensity"):
        base[key] = {"type": "score", "value": 0.0, "raw": 1.0, "confidence": 0.9}
    base["self_contained"] = {"type": "noul", "value": 0.9, "confidence": 0.9}
    for key, value in overrides.items():
        base[key] = {**base[key], "value": value}
    return base

def test_composite_is_zero_for_a_dead_window():
    assert composite(_answers(self_contained=0.0), PROFILE) == pytest.approx(0.0)

def test_clipworthy_dominates_the_score():
    assert composite(_answers(clipworthy=1.0), PROFILE) == pytest.approx(0.45)

def test_bonuses_lift_the_score():
    got = composite(_answers(clipworthy=1.0, poster_line=1.0), PROFILE)
    assert got == pytest.approx(0.53)

def test_needs_context_penalizes():
    got = composite(_answers(clipworthy=1.0, needs_context=1.0), PROFILE)
    assert got == pytest.approx(0.25)

def test_windup_penalizes():
    got = composite(_answers(clipworthy=1.0, opens_with_windup=1.0), PROFILE)
    assert got == pytest.approx(0.30)

def test_buried_lede_does_not_penalize():
    """A buried lede is a fixable in-point, not bad material. Claude trims it."""
    plain = composite(_answers(clipworthy=1.0), PROFILE)
    buried = composite(_answers(clipworthy=1.0, buried_lede=1.0), PROFILE)
    assert buried == pytest.approx(plain)

def test_composite_is_clamped_to_unit_range():
    hot = _answers(clipworthy=1.0, hook_strength=1.0, poster_line=1.0,
                   open_loop=1.0, payoff=1.0, concrete_specifics=1.0,
                   audible_reaction=1.0, disagreement=1.0, personal_story=1.0)
    assert 0.0 <= composite(hot, PROFILE) <= 1.0

def test_gate_rejects_incomprehensible_windows():
    assert passes_gate(_answers(self_contained=0.10), PROFILE) is False
    assert passes_gate(_answers(self_contained=0.90), PROFILE) is True

def test_low_confidence_windows_are_flagged_not_dropped():
    answers = _answers(clipworthy=0.9)
    answers["clipworthy"]["confidence"] = 0.30
    assert is_uncertain(answers, PROFILE) is True

def test_rank_annotates_every_record_and_skips_failures():
    scores = [{"id": 0, "failed": False, "answers": _answers(clipworthy=1.0)},
              {"id": 1, "failed": True, "error": "429 abc"}]
    out = rank(scores, PROFILE)
    assert out[0]["composite"] == pytest.approx(0.45)
    assert out[0]["gated"] is True
    assert "composite" not in out[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rank.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clipper.rank'`

- [ ] **Step 3: Implement `clipper/rank.py`**

```python
from __future__ import annotations

from clipper.profiles.loader import Profile


def _numeric(answers: dict, key: str) -> float:
    answer = answers.get(key)
    if not answer or not isinstance(answer.get("value"), (int, float)):
        return 0.0
    return float(answer["value"])


def composite(answers: dict, profile: Profile) -> float:
    total = sum(weight * _numeric(answers, key)
                for key, weight in profile.weights.items())
    total -= sum(weight * _numeric(answers, key)
                 for key, weight in profile.penalties.items())
    return min(1.0, max(0.0, total))


def passes_gate(answers: dict, profile: Profile) -> bool:
    return all(_numeric(answers, key) >= floor
               for key, floor in profile.gates.items())


def is_uncertain(answers: dict, profile: Profile) -> bool:
    floor = profile.thresholds.get("uncertain_confidence", 0.55)
    driver = answers.get("clipworthy")
    return bool(driver) and float(driver.get("confidence", 1.0)) < floor


def rank(scores: list[dict], profile: Profile) -> list[dict]:
    out: list[dict] = []
    for record in scores:
        if record.get("failed"):
            out.append(record)
            continue
        answers = record["answers"]
        out.append({**record,
                    "composite": composite(answers, profile),
                    "gated": passes_gate(answers, profile),
                    "uncertain": is_uncertain(answers, profile)})
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rank.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add clipper/rank.py tests/test_rank.py
git commit -m "feat: composite scoring, gates and uncertainty flagging"
```

---

### Task 10: Candidate merging and backward extension

**Files:**
- Create: `clipper/merge.py`
- Test: `tests/test_merge.py`

**Interfaces:**
- Consumes: ranked scores (Task 9), windows (Task 6), `Profile` (Task 7).
- Produces: `merge_candidates(windows: list[dict], ranked: list[dict], profile: Profile) -> dict` returning `{"candidates": [...], "uncertain": [...]}`.

Candidate dict: `{"id", "start", "end", "peak_composite", "mean_composite", "window_ids", "curve", "clip_format", "signals", "backward_extended"}`.

Spec §7: candidates cap at 60s; a candidate is extended backwards when its peak window has a late `energy_peak_offset`; energy alone never promotes a candidate.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_merge.py
import pytest
from clipper.merge import merge_candidates
from clipper.profiles.loader import load_profile

PROFILE = load_profile("podcast")

def _window(i: int, start: float, offset: float = 0.2, peak: float = 0.3) -> dict:
    return {"id": i, "start": start, "end": start + 30.0, "text": "t", "preceding": "",
            "position": 0.1, "energy_mean": 0.3, "energy_peak": peak,
            "energy_peak_offset": offset}

def _ranked(i: int, composite: float, gated: bool = True,
            uncertain: bool = False, fmt: str = "hot_take") -> dict:
    return {"id": i, "failed": False, "composite": composite, "gated": gated,
            "uncertain": uncertain,
            "answers": {"clip_format": {"type": "choice", "value": fmt, "confidence": 0.8},
                        "buried_lede": {"type": "noul", "value": 0.1, "confidence": 0.9},
                        "opens_with_windup": {"type": "noul", "value": 0.1, "confidence": 0.9}}}

def test_adjacent_hot_windows_merge_into_one_candidate():
    windows = [_window(0, 0.0), _window(1, 10.0), _window(2, 20.0)]
    ranked = [_ranked(0, 0.8), _ranked(1, 0.8), _ranked(2, 0.8)]
    out = merge_candidates(windows, ranked, PROFILE)
    assert len(out["candidates"]) == 1
    assert out["candidates"][0]["window_ids"] == [0, 1, 2]

def test_cold_windows_split_candidates():
    windows = [_window(i, i * 10.0) for i in range(5)]
    ranked = [_ranked(0, 0.8), _ranked(1, 0.1), _ranked(2, 0.1),
              _ranked(3, 0.1), _ranked(4, 0.8)]
    assert len(merge_candidates(windows, ranked, PROFILE)["candidates"]) == 2

def test_below_threshold_windows_are_excluded():
    windows = [_window(0, 0.0)]
    assert merge_candidates(windows, [_ranked(0, 0.2)], PROFILE)["candidates"] == []

def test_gated_windows_are_excluded_regardless_of_score():
    windows = [_window(0, 0.0)]
    ranked = [_ranked(0, 0.95, gated=False)]
    assert merge_candidates(windows, ranked, PROFILE)["candidates"] == []

def test_candidate_is_capped_at_max_seconds():
    windows = [_window(i, i * 10.0) for i in range(12)]
    ranked = [_ranked(i, 0.8) for i in range(12)]
    out = merge_candidates(windows, ranked, PROFILE)
    assert all(c["end"] - c["start"] <= 60.0 + 1e-6 for c in out["candidates"])

def test_late_energy_peak_extends_candidate_backwards():
    """The laugh is not the clip; the line that caused it is. Extend backwards."""
    windows = [_window(0, 100.0, offset=0.95, peak=0.9)]
    out = merge_candidates(windows, [_ranked(0, 0.8)], PROFILE)
    candidate = out["candidates"][0]
    assert candidate["start"] < 100.0
    assert candidate["backward_extended"] is True

def test_early_energy_peak_does_not_extend():
    windows = [_window(0, 100.0, offset=0.1, peak=0.9)]
    out = merge_candidates(windows, [_ranked(0, 0.8)], PROFILE)
    assert out["candidates"][0]["backward_extended"] is False

def test_backward_extension_never_goes_below_zero():
    windows = [_window(0, 2.0, offset=0.95, peak=0.9)]
    out = merge_candidates(windows, [_ranked(0, 0.8)], PROFILE)
    assert out["candidates"][0]["start"] >= 0.0

def test_energy_alone_never_promotes_a_candidate():
    """Loud but empty. Without a transcript signal there is no clip."""
    windows = [_window(0, 0.0, offset=0.9, peak=1.0)]
    out = merge_candidates(windows, [_ranked(0, 0.15)], PROFILE)
    assert out["candidates"] == []

def test_uncertain_windows_go_to_their_own_bucket():
    windows = [_window(0, 0.0)]
    ranked = [_ranked(0, 0.8, uncertain=True)]
    out = merge_candidates(windows, ranked, PROFILE)
    assert out["candidates"] == []
    assert out["uncertain"][0]["window_ids"] == [0]

def test_candidate_reports_peak_and_curve():
    windows = [_window(0, 0.0), _window(1, 10.0)]
    out = merge_candidates(windows, [_ranked(0, 0.6), _ranked(1, 0.9)], PROFILE)
    candidate = out["candidates"][0]
    assert candidate["peak_composite"] == pytest.approx(0.9)
    assert candidate["curve"] == [0.6, 0.9]

def test_candidates_are_sorted_by_peak_descending():
    windows = [_window(i, i * 40.0) for i in range(3)]
    ranked = [_ranked(0, 0.6), _ranked(1, 0.9), _ranked(2, 0.75)]
    out = merge_candidates(windows, ranked, PROFILE)
    assert [c["peak_composite"] for c in out["candidates"]] == [0.9, 0.75, 0.6]

def test_failed_windows_are_ignored():
    windows = [_window(0, 0.0), _window(1, 10.0)]
    ranked = [{"id": 0, "failed": True, "error": "429"}, _ranked(1, 0.8)]
    out = merge_candidates(windows, ranked, PROFILE)
    assert out["candidates"][0]["window_ids"] == [1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_merge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clipper.merge'`

- [ ] **Step 3: Implement `clipper/merge.py`**

```python
from __future__ import annotations

from collections import Counter

from clipper.profiles.loader import Profile

SIGNAL_KEYS = ("buried_lede", "opens_with_windup", "ends_cleanly",
               "needs_speaker_id", "requires_visual", "poster_line",
               "audible_reaction", "self_contained")


def _group(ids: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for wid in ids:
        if runs and wid == runs[-1][-1] + 1:
            runs[-1].append(wid)
        else:
            runs.append([wid])
    return runs


def _build(run: list[int], by_id: dict[int, dict], ranked_by_id: dict[int, dict],
           profile: Profile, index: int) -> dict:
    max_seconds = profile.merge.get("max_seconds", 60.0)
    extend_by = profile.merge.get("backward_extend_seconds", 8.0)
    extend_above = profile.merge.get("backward_extend_offset", 0.6)

    curve = [ranked_by_id[i]["composite"] for i in run]
    peak_id = run[curve.index(max(curve))]
    start = by_id[run[0]]["start"]
    end = by_id[run[-1]]["end"]

    # Reactions lag their cause: if the peak window's energy spikes late, the
    # material that caused it sits before the window, not inside it.
    peak_window = by_id[peak_id]
    extended = peak_window["energy_peak_offset"] >= extend_above
    if extended:
        start = max(0.0, start - extend_by)

    if end - start > max_seconds:
        centre = (by_id[peak_id]["start"] + by_id[peak_id]["end"]) / 2
        start = max(start, centre - max_seconds / 2)
        end = min(end, start + max_seconds)

    formats = Counter(
        ranked_by_id[i]["answers"].get("clip_format", {}).get("value", "filler")
        for i in run
    )
    peak_answers = ranked_by_id[peak_id]["answers"]
    return {
        "id": index,
        "start": start,
        "end": end,
        "window_ids": run,
        "curve": curve,
        "peak_composite": max(curve),
        "mean_composite": sum(curve) / len(curve),
        "clip_format": formats.most_common(1)[0][0],
        "signals": {k: peak_answers[k]["value"]
                    for k in SIGNAL_KEYS if k in peak_answers},
        "backward_extended": extended,
    }


def merge_candidates(windows: list[dict], ranked: list[dict], profile: Profile) -> dict:
    threshold = profile.thresholds.get("candidate", 0.55)
    by_id = {w["id"]: w for w in windows}
    ranked_by_id = {r["id"]: r for r in ranked if not r.get("failed")}

    hot, unsure = [], []
    for wid, record in sorted(ranked_by_id.items()):
        if not record.get("gated") or record["composite"] < threshold:
            continue
        (unsure if record.get("uncertain") else hot).append(wid)

    candidates = [_build(run, by_id, ranked_by_id, profile, i)
                  for i, run in enumerate(_group(hot))]
    candidates.sort(key=lambda c: c["peak_composite"], reverse=True)
    for i, candidate in enumerate(candidates):
        candidate["id"] = i

    uncertain = [_build(run, by_id, ranked_by_id, profile, i)
                 for i, run in enumerate(_group(unsure))]
    return {"candidates": candidates, "uncertain": uncertain}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_merge.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add clipper/merge.py tests/test_merge.py
git commit -m "feat: candidate merging with reaction-lag backward extension"
```

---

### Task 11: Score stage wiring

**Files:**
- Create: `clipper/score.py`
- Test: `tests/test_score_stage.py`

**Interfaces:**
- Consumes: `Run` (Task 2), `build_windows` (Task 6), `load_profile` (Task 7), `score_windows` (Task 8), `rank` (Task 9), `merge_candidates` (Task 10).
- Produces: `async run_score(run: Run, profile_name: str, client=None) -> dict`.

Writes `scores.json` (`{"profile","jev_model","windows":[...]}`) and `candidates.json` (`{"profile","jev_model","candidates":[...],"uncertain":[...]}`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_score_stage.py
import pytest
from clipper.run import Run
from clipper.score import run_score

class FakeClient:
    """Returns a fixed high score for every window."""
    def __init__(self): self.calls = 0
    async def system_one(self, state, questions):
        self.calls += 1
        return FakeResponse(questions)
    async def close(self): pass

class FakeResponse:
    model = "jev-1.13.0"
    def __init__(self, questions):
        self.answers = {k: _answer(k, q) for k, q in questions.items()}

def _answer(key, question):
    kind = type(question).__name__.lower()
    if kind == "score":
        return type("S", (), {"type": "score", "score": 4.5, "confidence": 0.9,
                              "legend": {i: "x" for i in range(1, 6)},
                              "probabilities": {}})()
    if kind == "choice":
        return type("C", (), {"type": "choice", "choice": "hot_take",
                              "confidence": 0.8, "probabilities": {"hot_take": 0.8}})()
    value = 0.9 if key == "self_contained" else 0.2
    return type("N", (), {"type": "noul", "noul": value})()

def _seed(run: Run):
    words = [{"word": f"w{i}", "start": i / 3, "end": (i + 0.9) / 3,
              "score": 0.9, "speaker": "SPEAKER_00"} for i in range(300)]
    run.write_json("source.json", {"duration": 100.0, "energy": [0.4] * 100,
                                   "video": {"width": 1920, "height": 1080}})
    run.write_json("transcript.json", {
        "language": "en", "model": "large-v3", "diarized": True,
        "segments": [{"start": 0.0, "end": 100.0, "speaker": "SPEAKER_00",
                      "text": "t", "words": words}]})

@pytest.mark.asyncio
async def test_score_writes_both_artifacts(tmp_path):
    run = Run.create(tmp_path, "ep")
    _seed(run)
    await run_score(run, "podcast", client=FakeClient())
    assert run.exists("scores.json")
    assert run.exists("candidates.json")

@pytest.mark.asyncio
async def test_score_records_the_resolved_model_version(tmp_path):
    """Thresholds tuned on one version must not silently drift onto another."""
    run = Run.create(tmp_path, "ep")
    _seed(run)
    await run_score(run, "podcast", client=FakeClient())
    assert run.read_json("scores.json")["jev_model"] == "jev-1.13.0"
    assert run.read_json("candidates.json")["jev_model"] == "jev-1.13.0"

@pytest.mark.asyncio
async def test_score_calls_jev_once_per_window(tmp_path):
    run = Run.create(tmp_path, "ep")
    _seed(run)
    client = FakeClient()
    await run_score(run, "podcast", client=client)
    assert client.calls == len(run.read_json("scores.json")["windows"])

@pytest.mark.asyncio
async def test_score_also_writes_windows_artifact(tmp_path):
    run = Run.create(tmp_path, "ep")
    _seed(run)
    await run_score(run, "podcast", client=FakeClient())
    assert len(run.read_json("windows.json")["windows"]) > 0

@pytest.mark.asyncio
async def test_rescoring_a_different_profile_reuses_the_transcript(tmp_path):
    run = Run.create(tmp_path, "ep")
    _seed(run)
    await run_score(run, "podcast", client=FakeClient())
    await run_score(run, "lecture", client=FakeClient())
    assert run.read_json("candidates.json")["profile"] == "lecture"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_score_stage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clipper.score'`

- [ ] **Step 3: Implement `clipper/score.py`**

```python
from __future__ import annotations

from clipper.jev import score_windows
from clipper.merge import merge_candidates
from clipper.profiles.loader import load_profile
from clipper.rank import rank
from clipper.run import Run
from clipper.window import build_windows


async def run_score(run: Run, profile_name: str, client=None) -> dict:
    profile = load_profile(profile_name)
    source = run.read_json("source.json")
    transcript = run.read_json("transcript.json")

    windows = build_windows(
        transcript,
        source["energy"],
        duration=source["duration"],
        window_seconds=profile.window.get("seconds", 30.0),
        step_seconds=profile.window.get("step", 10.0),
        context_seconds=profile.window.get("context", 20.0),
    )
    run.write_json("windows.json", {"profile": profile.name, "windows": windows})

    scores, model = await score_windows(windows, profile, client=client)
    ranked = rank(scores, profile)
    run.write_json("scores.json", {"profile": profile.name,
                                   "jev_model": model, "windows": ranked})

    merged = merge_candidates(windows, ranked, profile)
    failures = sum(1 for r in ranked if r.get("failed"))
    candidates = {"profile": profile.name, "jev_model": model,
                  "failed_windows": failures, **merged}
    run.write_json("candidates.json", candidates)

    if not merged["candidates"]:
        top = sorted((r.get("composite", 0.0) for r in ranked), reverse=True)[:5]
        print("No candidate cleared the threshold "
              f"({profile.thresholds.get('candidate', 0.55)}). "
              f"Top composites were: {[round(t, 3) for t in top]}. "
              "Lower `thresholds.candidate` in the profile, or try another profile.")
    return candidates
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_score_stage.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add clipper/score.py tests/test_score_stage.py
git commit -m "feat: score stage wiring producing scores and candidates"
```

---

### Task 12: Caption cues, rebasing and SRT

**Files:**
- Create: `clipper/captions.py`
- Test: `tests/test_captions.py`

**Interfaces:**
- Consumes: `transcript.json` (Task 5).
- Produces: `Cue` frozen dataclass (`start: float`, `end: float`, `words: list[dict]`, `speaker: str`, with a `text` property), `words_in_range(transcript, start, end) -> list[dict]`, `build_cues(words, max_chars=42, max_seconds=3.0) -> list[Cue]`, `rebase(cues, clip_start) -> list[Cue]`, `render_srt(cues) -> str`, `format_srt_time(seconds) -> str`.

Rebasing per the reference project: `adjusted = original - clip_start`, clamped at 0.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_captions.py
import pytest
from clipper.captions import (
    Cue, build_cues, format_srt_time, rebase, render_srt, words_in_range,
)

def _words(n=12, start=100.0, dur=0.4, speaker="SPEAKER_00"):
    return [{"word": f"word{i}", "start": start + i * dur,
             "end": start + i * dur + dur * 0.9, "score": 0.9, "speaker": speaker}
            for i in range(n)]

def _transcript(words):
    return {"language": "en", "model": "m", "diarized": True,
            "segments": [{"start": words[0]["start"], "end": words[-1]["end"],
                          "speaker": words[0]["speaker"], "text": "x", "words": words}]}

def test_words_in_range_selects_overlapping_words_only():
    t = _transcript(_words(20))
    got = words_in_range(t, 102.0, 104.0)
    assert all(w["end"] > 102.0 and w["start"] < 104.0 for w in got)
    assert got

def test_build_cues_breaks_on_character_limit():
    cues = build_cues(_words(30), max_chars=20)
    assert all(len(c.text) <= 20 for c in cues)

def test_build_cues_breaks_on_duration_limit():
    cues = build_cues(_words(30, dur=0.5), max_chars=500, max_seconds=2.0)
    assert all(c.end - c.start <= 2.0 + 1e-6 for c in cues)

def test_build_cues_breaks_on_speaker_change():
    words = _words(6, speaker="SPEAKER_00") + _words(6, start=103.0, speaker="SPEAKER_01")
    cues = build_cues(words, max_chars=500, max_seconds=60.0)
    assert len(cues) == 2
    assert cues[0].speaker == "SPEAKER_00"
    assert cues[1].speaker == "SPEAKER_01"

def test_rebase_subtracts_clip_start():
    cues = build_cues(_words(6))
    out = rebase(cues, clip_start=100.0)
    assert out[0].start == pytest.approx(0.0, abs=0.01)

def test_rebase_clamps_negative_times_to_zero():
    """A cue that starts before the cut must not produce a negative timestamp."""
    cue = Cue(start=99.0, end=101.0, words=_words(2, start=99.0), speaker="SPEAKER_00")
    assert rebase([cue], clip_start=100.0)[0].start == 0.0

def test_rebase_drops_cues_entirely_before_the_cut():
    cue = Cue(start=90.0, end=95.0, words=_words(2, start=90.0), speaker="SPEAKER_00")
    assert rebase([cue], clip_start=100.0) == []

def test_format_srt_time_uses_comma_separator():
    assert format_srt_time(3661.25) == "01:01:01,250"

def test_format_srt_time_handles_zero():
    assert format_srt_time(0.0) == "00:00:00,000"

def test_render_srt_numbers_cues_from_one():
    out = render_srt(build_cues(_words(12), max_chars=20))
    assert out.startswith("1\n")
    assert "\n2\n" in out

def test_render_srt_ends_with_a_blank_line():
    assert render_srt(build_cues(_words(4))).endswith("\n\n")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_captions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clipper.captions'`

- [ ] **Step 3: Implement the SRT half of `clipper/captions.py`**

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    words: list[dict]
    speaker: str

    @property
    def text(self) -> str:
        return " ".join(w["word"].strip() for w in self.words)


def words_in_range(transcript: dict, start: float, end: float) -> list[dict]:
    out: list[dict] = []
    for segment in transcript["segments"]:
        for word in segment["words"]:
            if word["end"] > start and word["start"] < end:
                out.append(word)
    return out


def build_cues(words: list[dict], max_chars: int = 42,
               max_seconds: float = 3.0) -> list[Cue]:
    cues: list[Cue] = []
    buffer: list[dict] = []

    def flush() -> None:
        if buffer:
            cues.append(Cue(start=buffer[0]["start"], end=buffer[-1]["end"],
                            words=list(buffer), speaker=buffer[0]["speaker"]))
            buffer.clear()

    for word in words:
        if buffer:
            too_long = len(" ".join(w["word"].strip() for w in buffer + [word])) > max_chars
            too_slow = word["end"] - buffer[0]["start"] > max_seconds
            changed = word["speaker"] != buffer[0]["speaker"]
            if too_long or too_slow or changed:
                flush()
        buffer.append(word)
    flush()
    return cues


def rebase(cues: list[Cue], clip_start: float) -> list[Cue]:
    """Shift cues into clip-local time. Reference project: adjusted = original
    - clip_start, clamped at 0."""
    out: list[Cue] = []
    for cue in cues:
        if cue.end <= clip_start:
            continue
        out.append(Cue(
            start=max(0.0, cue.start - clip_start),
            end=max(0.0, cue.end - clip_start),
            words=[{**w,
                    "start": max(0.0, w["start"] - clip_start),
                    "end": max(0.0, w["end"] - clip_start)} for w in cue.words],
            speaker=cue.speaker,
        ))
    return out


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def render_srt(cues: list[Cue]) -> str:
    blocks = [
        f"{i}\n{format_srt_time(c.start)} --> {format_srt_time(c.end)}\n{c.text}\n"
        for i, c in enumerate(cues, start=1)
    ]
    return "\n".join(blocks) + "\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_captions.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add clipper/captions.py tests/test_captions.py
git commit -m "feat: caption cues, clip-local rebasing and SRT rendering"
```

---

### Task 13: ASS rendering with word-level highlighting

**Files:**
- Modify: `clipper/captions.py`
- Test: `tests/test_ass.py`

**Interfaces:**
- Consumes: `Cue`, `rebase` (Task 12).
- Produces: `font_size_for_height(height: int) -> int`, `format_ass_time(seconds: float) -> str`, `render_ass(cues: list[Cue], height: int, speakers: dict[str, str] | None = None) -> str`.

The reference project only ever produced SRT plus `force_style`; word-level highlighting has no prior art there and is written from scratch. Highlighting uses `\k` karaoke timing, which libass renders natively.

Font sizes follow the reference guide: 720p → 20, 1080p → 24, 4K → 48.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ass.py
import pytest
from clipper.captions import (
    Cue, font_size_for_height, format_ass_time, render_ass,
)

def _cue(start=0.0, n=3, dur=0.4, speaker="SPEAKER_00"):
    words = [{"word": f"w{i}", "start": start + i * dur,
              "end": start + (i + 1) * dur, "score": 0.9, "speaker": speaker}
             for i in range(n)]
    return Cue(start=words[0]["start"], end=words[-1]["end"],
               words=words, speaker=speaker)

def test_font_size_matches_reference_table():
    assert font_size_for_height(720) == 20
    assert font_size_for_height(1080) == 24
    assert font_size_for_height(2160) == 48

def test_vertical_output_gets_large_captions():
    """1080x1920 shorts want big captions, not 1080p-horizontal sizing."""
    assert font_size_for_height(1920) >= 40

def test_format_ass_time_uses_centiseconds_and_one_hour_digit():
    assert format_ass_time(3661.25) == "1:01:01.25"
    assert format_ass_time(0.0) == "0:00:00.00"

def test_ass_has_required_sections():
    out = render_ass([_cue()], height=1080)
    assert "[Script Info]" in out
    assert "[V4+ Styles]" in out
    assert "[Events]" in out

def test_ass_declares_play_resolution():
    out = render_ass([_cue()], height=1920)
    assert "PlayResY: 1920" in out

def test_each_word_gets_its_own_karaoke_tag():
    out = render_ass([_cue(n=3)], height=1080)
    dialogue = [l for l in out.splitlines() if l.startswith("Dialogue:")][0]
    assert dialogue.count("\\k") == 3

def test_karaoke_durations_are_in_centiseconds():
    out = render_ass([_cue(n=2, dur=0.5)], height=1080)
    assert "\\k50}" in out

def test_speaker_label_is_emitted_when_names_are_supplied():
    out = render_ass([_cue(speaker="SPEAKER_01")], height=1080,
                     speakers={"SPEAKER_01": "Marco"})
    assert "Marco" in out
    assert "Style: Speaker" in out

def test_no_speaker_label_without_names():
    out = render_ass([_cue(speaker="SPEAKER_01")], height=1080)
    assert "Marco" not in out
    assert "Dialogue:" in out

def test_unmapped_speaker_produces_no_label():
    out = render_ass([_cue(speaker="SPEAKER_09")], height=1080,
                     speakers={"SPEAKER_01": "Marco"})
    assert "SPEAKER_09" not in out

def test_braces_in_text_are_escaped():
    """An unescaped brace would be parsed as an ASS override block."""
    cue = Cue(start=0.0, end=1.0, speaker="SPEAKER_00",
              words=[{"word": "{hi}", "start": 0.0, "end": 1.0,
                      "score": 0.9, "speaker": "SPEAKER_00"}])
    out = render_ass([cue], height=1080)
    assert "{hi}" not in out
    assert "(hi)" in out

def test_empty_cue_list_still_produces_a_valid_file():
    out = render_ass([], height=1080)
    assert "[Events]" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ass.py -v`
Expected: FAIL with `ImportError: cannot import name 'render_ass'`

- [ ] **Step 3: Append to `clipper/captions.py`**

```python
ASS_HEADER = """[Script Info]
ScriptType: v4.00+
WrapStyle: 2
ScaledBorderAndShadow: yes
PlayResX: {play_x}
PlayResY: {play_y}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Arial,{size},&H00FFFFFF,&H0000D7FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},1,2,60,60,{margin},1
Style: Speaker,Arial,{speaker_size},&H00D0D0D0,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,1,40,40,{speaker_margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def font_size_for_height(height: int) -> int:
    """Reference guide: 720p -> 20, 1080p -> 24, 4K -> 48."""
    if height <= 720:
        return 20
    if height <= 1080:
        return 24
    if height <= 1440:
        return 32
    return 48


def format_ass_time(seconds: float) -> str:
    centis = int(round(seconds * 100))
    hours, centis = divmod(centis, 360_000)
    minutes, centis = divmod(centis, 6_000)
    secs, centis = divmod(centis, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _escape(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")")


def _karaoke(cue: Cue) -> str:
    parts: list[str] = []
    for word in cue.words:
        centis = max(1, int(round((word["end"] - word["start"]) * 100)))
        parts.append(f"{{\\k{centis}}}{_escape(word['word'].strip())}")
    return " ".join(parts)


def render_ass(cues: list[Cue], height: int,
               speakers: dict[str, str] | None = None) -> str:
    size = font_size_for_height(height)
    header = ASS_HEADER.format(
        play_x=int(height * 9 / 16) if height >= 1920 else int(height * 16 / 9),
        play_y=height,
        size=size,
        outline=max(1, size // 10),
        margin=int(height * 0.08),
        speaker_size=max(14, int(size * 0.7)),
        speaker_margin=int(height * 0.04),
    )
    lines: list[str] = []
    for cue in cues:
        start, end = format_ass_time(cue.start), format_ass_time(cue.end)
        lines.append(f"Dialogue: 0,{start},{end},Caption,,0,0,0,,{_karaoke(cue)}")
        name = (speakers or {}).get(cue.speaker)
        if name:
            lines.append(
                f"Dialogue: 0,{start},{end},Speaker,,0,0,0,,{_escape(name)}"
            )
    return header + "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ass.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add clipper/captions.py tests/test_ass.py
git commit -m "feat: ASS captions with word-level karaoke highlighting"
```

---

### Task 14: Crop arithmetic and filter chain

**Files:**
- Create: `clipper/filters.py`
- Test: `tests/test_filters.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `crop_expression(width: int, height: int, crop_x: str | int) -> str`, `subtitles_expression(path: Path) -> str`, `build_filter_chain(width, height, vertical, subtitle_path, crop_x="center") -> str | None`.

Spec §9: order is crop, then scale, then subtitles, so caption sizing is computed in output space.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_filters.py
from pathlib import Path
import pytest
from clipper.filters import build_filter_chain, crop_expression, subtitles_expression

def test_crop_expression_is_nine_by_sixteen_of_the_height():
    assert crop_expression(1920, 1080, "center") == "crop=607:1080:656:0"

def test_crop_centers_by_default():
    expr = crop_expression(1920, 1080, "center")
    x = int(expr.split(":")[2])
    assert x == (1920 - 607) // 2

def test_explicit_crop_x_is_honoured():
    assert crop_expression(1920, 1080, 100).endswith(":100:0")

def test_crop_x_is_clamped_into_frame():
    assert crop_expression(1920, 1080, 5000).split(":")[2] == str(1920 - 607)
    assert crop_expression(1920, 1080, -50).split(":")[2] == "0"

def test_source_narrower_than_nine_by_sixteen_is_not_cropped():
    """Already-vertical footage must not be cropped to nothing."""
    assert crop_expression(1080, 1920, "center") == ""

def test_filter_order_is_crop_then_scale_then_subtitles(tmp_path):
    sub = tmp_path / "c.ass"
    sub.write_text("")
    chain = build_filter_chain(1920, 1080, vertical=True, subtitle_path=sub)
    assert chain.index("crop=") < chain.index("scale=") < chain.index("subtitles=")

def test_vertical_scales_to_1080x1920(tmp_path):
    chain = build_filter_chain(1920, 1080, vertical=True, subtitle_path=None)
    assert "scale=1080:1920" in chain

def test_horizontal_has_no_crop_or_scale(tmp_path):
    sub = tmp_path / "c.ass"
    sub.write_text("")
    chain = build_filter_chain(1920, 1080, vertical=False, subtitle_path=sub)
    assert "crop=" not in chain
    assert "scale=" not in chain
    assert "subtitles=" in chain

def test_no_filters_at_all_returns_none():
    assert build_filter_chain(1920, 1080, vertical=False, subtitle_path=None) is None

def test_subtitles_expression_escapes_windows_drive_colon(tmp_path):
    expr = subtitles_expression(Path("C:/tmp/clipper_x/c.ass"))
    assert "C\\:" in expr or "C\\\\:" in expr

def test_subtitles_expression_rejects_a_path_with_spaces():
    """ffmpeg truncates such paths at the first space. Stage to a temp dir."""
    with pytest.raises(ValueError, match="space"):
        subtitles_expression(Path("C:/Users/Linhao Jin/c.ass"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_filters.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clipper.filters'`

- [ ] **Step 3: Implement `clipper/filters.py`**

```python
from __future__ import annotations

from pathlib import Path

VERTICAL_W, VERTICAL_H = 1080, 1920


def crop_expression(width: int, height: int, crop_x: str | int = "center") -> str:
    target = int(height * 9 / 16)
    if target >= width:
        return ""  # already at or narrower than 9:16
    if crop_x == "center":
        x = (width - target) // 2
    else:
        x = min(max(0, int(crop_x)), width - target)
    return f"crop={target}:{height}:{x}:0"


def subtitles_expression(path: Path) -> str:
    """Build a `subtitles=` filter argument.

    The path must already be space-free. ffmpeg truncates subtitle paths at the
    first space regardless of quoting or backslash escaping, so callers stage
    the file into a short temp directory first (see render.staged_subtitles).
    """
    text = path.as_posix()
    if " " in text:
        raise ValueError(
            f"Subtitle path contains a space: {text}. ffmpeg's subtitles filter "
            "truncates at the first space. Stage the file into a space-free "
            "temp directory before building the filter."
        )
    return "subtitles=" + text.replace(":", r"\:")


def build_filter_chain(width: int, height: int, vertical: bool,
                       subtitle_path: Path | None,
                       crop_x: str | int = "center") -> str | None:
    parts: list[str] = []
    if vertical:
        crop = crop_expression(width, height, crop_x)
        if crop:
            parts.append(crop)
        parts.append(f"scale={VERTICAL_W}:{VERTICAL_H}")
    if subtitle_path is not None:
        parts.append(subtitles_expression(subtitle_path))
    return ",".join(parts) if parts else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_filters.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add clipper/filters.py tests/test_filters.py
git commit -m "feat: crop arithmetic and ffmpeg filter chain construction"
```

---

### Task 15: Render stage

Read `scripts/clip_video.py`, `scripts/burn_subtitles.py` and `TECHNICAL_NOTES.md` from the reference project before starting this task.

**Files:**
- Create: `clipper/render.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `preflight` (Task 1), `Run` (Task 2), captions (Tasks 12–13), filters (Task 14).
- Produces: `slugify(text: str) -> str`, `staged_subtitles(content: str, suffix: str)` (context manager yielding a space-free `Path`), `validate_clip(clip: dict, duration: float) -> dict`, `build_command(ffmpeg, source, clip, output, filter_chain) -> list[str]`, `render_clip(...) -> Path`, `run_render(run: Run, vertical=False, captions="burn") -> list[dict]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_render.py
import shutil
import subprocess
from pathlib import Path
import pytest
from clipper.render import (
    build_command, slugify, staged_subtitles, validate_clip,
)

def test_slugify_makes_a_filesystem_safe_name():
    assert slugify("The thing nobody says about funding!") == "the-thing-nobody-says-about-funding"

def test_slugify_truncates_long_titles():
    assert len(slugify("word " * 60)) <= 60

def test_slugify_never_returns_empty():
    assert slugify("!!!") == "clip"

def test_staged_subtitles_path_has_no_spaces():
    """The whole point: ffmpeg truncates subtitle paths at the first space."""
    with staged_subtitles("content", ".ass") as path:
        assert " " not in str(path)
        assert path.read_text() == "content"

def test_staged_subtitles_cleans_up():
    with staged_subtitles("x", ".ass") as path:
        parent = path.parent
    assert not parent.exists()

def test_validate_clip_clamps_out_to_duration():
    out = validate_clip({"in": 90.0, "out": 200.0}, duration=100.0)
    assert out["out"] == 100.0

def test_validate_clip_rejects_clips_under_ten_seconds():
    with pytest.raises(ValueError, match="10"):
        validate_clip({"in": 10.0, "out": 18.0}, duration=100.0)

def test_validate_clip_rejects_inverted_boundaries():
    with pytest.raises(ValueError, match="after"):
        validate_clip({"in": 50.0, "out": 40.0}, duration=100.0)

def test_validate_clip_rejects_in_point_past_the_end():
    with pytest.raises(ValueError, match="beyond"):
        validate_clip({"in": 500.0, "out": 520.0}, duration=100.0)

def test_build_command_re_encodes_rather_than_stream_copying():
    """Claude trimmed to the word; stream copy would snap back to keyframes."""
    cmd = build_command(Path("ffmpeg"), Path("src.mp4"),
                        {"in": 10.0, "out": 40.0}, Path("out.mp4"), None)
    assert "-c" not in cmd and "copy" not in cmd
    assert "libx264" in cmd

def test_build_command_seeks_accurately():
    cmd = build_command(Path("ffmpeg"), Path("src.mp4"),
                        {"in": 10.0, "out": 40.0}, Path("out.mp4"), None)
    assert "-ss" in cmd and "-to" in cmd
    assert "-accurate_seek" in cmd

def test_build_command_includes_the_filter_chain():
    cmd = build_command(Path("ffmpeg"), Path("src.mp4"), {"in": 0.0, "out": 20.0},
                        Path("out.mp4"), "scale=1080:1920")
    assert "-vf" in cmd
    assert cmd[cmd.index("-vf") + 1] == "scale=1080:1920"

@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_render_produces_a_playable_file(tmp_path):
    """Generate a source at test time; no media is committed to the repo."""
    source = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=20:size=640x480:rate=25",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
         "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source)],
        capture_output=True, check=True,
    )
    from clipper.render import render_clip
    out = render_clip(Path(shutil.which("ffmpeg")), source,
                      {"in": 2.0, "out": 14.0}, tmp_path / "clip.mp4", None)
    assert out.exists() and out.stat().st_size > 0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    )
    assert float(probe.stdout.strip()) == pytest.approx(12.0, abs=0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clipper.render'`

- [ ] **Step 3: Implement `clipper/render.py`**

```python
from __future__ import annotations

import contextlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from clipper.captions import (
    build_cues, rebase, render_ass, render_srt, words_in_range,
)
from clipper.filters import build_filter_chain
from clipper.preflight import preflight
from clipper.run import Run

MIN_CLIP_SECONDS = 10.0


def slugify(text: str, limit: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:limit].rstrip("-") or "clip")


@contextlib.contextmanager
def staged_subtitles(content: str, suffix: str):
    """Write subtitles into a space-free temp directory.

    ffmpeg's subtitles filter truncates paths at the first space. Quoting,
    single quotes and backslash escaping have all been tried and none work.
    """
    directory = Path(tempfile.mkdtemp(prefix="clipper_"))
    try:
        path = directory / f"sub{suffix}"
        path.write_text(content, encoding="utf-8")
        yield path
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def validate_clip(clip: dict, duration: float) -> dict:
    start, end = float(clip["in"]), float(clip["out"])
    if start >= duration:
        raise ValueError(f"Clip in-point {start:.2f}s is beyond the source end ({duration:.2f}s).")
    if end <= start:
        raise ValueError(f"Clip out-point {end:.2f}s is not after its in-point {start:.2f}s.")
    end = min(end, duration)
    if end - start < MIN_CLIP_SECONDS:
        raise ValueError(
            f"Clip is {end - start:.2f}s, under the {MIN_CLIP_SECONDS:.0f}s minimum. "
            "Shorter clips read as incomplete."
        )
    return {**clip, "in": start, "out": end}


def build_command(ffmpeg: Path, source: Path, clip: dict,
                  output: Path, filter_chain: str | None) -> list[str]:
    cmd = [str(ffmpeg), "-y", "-accurate_seek",
           "-ss", f"{clip['in']:.3f}", "-to", f"{clip['out']:.3f}",
           "-i", str(source)]
    if filter_chain:
        cmd += ["-vf", filter_chain]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart", str(output)]
    return cmd


def render_clip(ffmpeg: Path, source: Path, clip: dict,
                output: Path, filter_chain: str | None) -> Path:
    result = subprocess.run(build_command(ffmpeg, source, clip, output, filter_chain),
                            capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Render failed for {output.name}:\n{result.stderr[-2000:]}")
    return output


def run_render(run: Run, vertical: bool = False, captions: str = "burn") -> list[dict]:
    plan = run.read_json("plan.json")
    source_meta = run.read_json("source.json")
    transcript = run.read_json("transcript.json")
    tools = preflight(require_subtitles=captions == "burn")

    source = Path(source_meta["path"])
    width, height = source_meta["video"]["width"], source_meta["video"]["height"]
    speakers = plan.get("speakers") or {}
    out_dir = run.clips_dir()
    written: list[dict] = []

    for index, raw in enumerate(plan["clips"], start=1):
        clip = validate_clip(raw, source_meta["duration"])
        mode = clip.get("captions", captions)
        want_vertical = clip.get("vertical", vertical)
        stem = f"{index:02d}-{slugify(clip.get('title', ''))}"

        cues = rebase(build_cues(words_in_range(transcript, clip["in"], clip["out"])),
                      clip["in"])
        (out_dir / f"{stem}.srt").write_text(render_srt(cues), encoding="utf-8")

        out_height = 1920 if want_vertical else height
        labelled = speakers if clip.get("speaker_label") else None
        ass = render_ass(cues, out_height, labelled) if mode == "burn" else None

        target = out_dir / f"{stem}.mp4"
        if ass is not None:
            with staged_subtitles(ass, ".ass") as staged:
                chain = build_filter_chain(width, height, want_vertical, staged,
                                           clip.get("crop_x", "center"))
                render_clip(tools.ffmpeg, source, clip, target, chain)
        else:
            chain = build_filter_chain(width, height, want_vertical, None,
                                       clip.get("crop_x", "center"))
            render_clip(tools.ffmpeg, source, clip, target, chain)

        meta = {k: v for k, v in clip.items() if k != "jev"}
        meta["duration"] = round(clip["out"] - clip["in"], 2)
        meta["jev"] = clip.get("jev", {})
        (out_dir / f"{stem}.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append({"file": target.name, **meta})
    return written
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_render.py -v`
Expected: 13 passed (the last skips if ffmpeg is absent)

- [ ] **Step 5: Commit**

```bash
git add clipper/render.py tests/test_render.py
git commit -m "feat: render stage with temp-staged subtitles and accurate seeking"
```

---

### Task 16: LLM writer adapter

**Files:**
- Create: `clipper/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `WriterConfig` frozen dataclass (`base_url: str`, `api_key: str`, `model: str`), `writer_from_env() -> WriterConfig | None`, `PRESETS: dict[str, str]`, `build_payload(config, system, user) -> dict`, `write_metadata(config, transcript_text, clip_format) -> dict` returning `{"title","hook","description"}`.

One OpenAI-compatible client covers DeepSeek and Kimi; only the base URL differs.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm.py
import json
import pytest
from clipper.llm import (
    PRESETS, WriterConfig, build_payload, parse_metadata, writer_from_env,
)

def test_presets_cover_deepseek_and_kimi():
    assert "deepseek" in PRESETS
    assert "kimi" in PRESETS
    assert all(url.startswith("https://") for url in PRESETS.values())

def test_writer_from_env_returns_none_when_unconfigured(monkeypatch):
    for key in ("WRITER_BASE_URL", "WRITER_API_KEY", "WRITER_MODEL"):
        monkeypatch.delenv(key, raising=False)
    assert writer_from_env() is None

def test_writer_from_env_builds_config(monkeypatch):
    monkeypatch.setenv("WRITER_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("WRITER_API_KEY", "sk-x")
    monkeypatch.setenv("WRITER_MODEL", "deepseek-chat")
    config = writer_from_env()
    assert config.model == "deepseek-chat"

def test_writer_from_env_expands_a_preset_name(monkeypatch):
    monkeypatch.setenv("WRITER_BASE_URL", "kimi")
    monkeypatch.setenv("WRITER_API_KEY", "sk-x")
    monkeypatch.setenv("WRITER_MODEL", "moonshot-v1-8k")
    assert writer_from_env().base_url == PRESETS["kimi"]

def test_writer_from_env_requires_a_key(monkeypatch):
    monkeypatch.setenv("WRITER_BASE_URL", "kimi")
    monkeypatch.delenv("WRITER_API_KEY", raising=False)
    monkeypatch.setenv("WRITER_MODEL", "m")
    with pytest.raises(ValueError, match="WRITER_API_KEY"):
        writer_from_env()

def test_build_payload_requests_json_output():
    config = WriterConfig("https://x/v1", "sk", "m")
    payload = build_payload(config, "sys", "user")
    assert payload["model"] == "m"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"][0]["role"] == "system"

def test_parse_metadata_reads_the_three_fields():
    raw = json.dumps({"title": "T", "hook": "H", "description": "D"})
    assert parse_metadata(raw) == {"title": "T", "hook": "H", "description": "D"}

def test_parse_metadata_strips_markdown_fencing():
    """Several providers wrap JSON in a fenced block despite json_object mode."""
    raw = '```json\n{"title":"T","hook":"H","description":"D"}\n```'
    assert parse_metadata(raw)["title"] == "T"

def test_parse_metadata_rejects_missing_fields():
    with pytest.raises(ValueError, match="hook"):
        parse_metadata('{"title": "T", "description": "D"}')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clipper.llm'`

- [ ] **Step 3: Implement `clipper/llm.py`**

```python
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import httpx

PRESETS = {
    "deepseek": "https://api.deepseek.com/v1",
    "kimi": "https://api.moonshot.cn/v1",
}

SYSTEM = (
    "You write metadata for short video clips. Reply with JSON only, with keys "
    "title, hook and description. The title is under 60 characters and is not "
    "clickbait. The hook is one sentence that could be the first line on screen. "
    "The description is two sentences."
)


@dataclass(frozen=True)
class WriterConfig:
    base_url: str
    api_key: str
    model: str


def writer_from_env() -> WriterConfig | None:
    base = os.environ.get("WRITER_BASE_URL")
    model = os.environ.get("WRITER_MODEL")
    if not base or not model:
        return None
    key = os.environ.get("WRITER_API_KEY")
    if not key:
        raise ValueError("WRITER_BASE_URL is set but WRITER_API_KEY is missing.")
    return WriterConfig(base_url=PRESETS.get(base, base), api_key=key, model=model)


def build_payload(config: WriterConfig, system: str, user: str) -> dict:
    return {
        "model": config.model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "response_format": {"type": "json_object"},
        "temperature": 0.7,
    }


def parse_metadata(raw: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    data = json.loads(text)
    for field in ("title", "hook", "description"):
        if field not in data:
            raise ValueError(f"Writer response is missing '{field}': {text[:200]}")
    return {k: data[k] for k in ("title", "hook", "description")}


def write_metadata(config: WriterConfig, transcript_text: str, clip_format: str) -> dict:
    user = f"Clip format: {clip_format}\n\nTranscript:\n{transcript_text}"
    response = httpx.post(
        f"{config.base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {config.api_key}"},
        json=build_payload(config, SYSTEM, user),
        timeout=60.0,
    )
    response.raise_for_status()
    return parse_metadata(response.json()["choices"][0]["message"]["content"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add clipper/llm.py tests/test_llm.py
git commit -m "feat: OpenAI-compatible writer adapter for DeepSeek and Kimi"
```

---

### Task 17: Plan validation and the Claude skill

**Files:**
- Create: `clipper/plan.py`, `.claude/skills/clipper/SKILL.md`
- Test: `tests/test_plan.py`

**Interfaces:**
- Consumes: `Run` (Task 2).
- Produces: `LENGTH_TARGETS: dict[str, tuple[float, float]]`, `validate_plan(plan: dict, duration: float) -> list[str]` returning human-readable warnings, `check_plan(run: Run) -> list[str]`.

`clipper plan` does not generate the plan — Claude does. This command validates what Claude wrote, so a bad plan fails before rendering rather than during it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plan.py
import pytest
from clipper.plan import LENGTH_TARGETS, validate_plan

def _plan(**clip):
    base = {"in": 100.0, "out": 130.0, "title": "T", "hook": "H",
            "description": "D", "clip_format": "story_arc"}
    return {"profile": "podcast", "jev_model": "jev-1.13.0",
            "clips": [{**base, **clip}]}

def test_length_targets_cover_every_clip_format():
    for fmt in ("cold_open", "debate", "how_to", "story_arc",
                "hot_take", "confession", "list"):
        assert fmt in LENGTH_TARGETS

def test_valid_plan_produces_no_warnings():
    assert validate_plan(_plan(), duration=5400.0) == []

def test_clip_under_ten_seconds_is_flagged():
    warnings = validate_plan(_plan(out=108.0), duration=5400.0)
    assert any("10" in w for w in warnings)

def test_clip_over_sixty_seconds_is_flagged_not_rejected():
    """Spec: flag for a second look; an exceptional segment can carry it."""
    warnings = validate_plan(_plan(out=200.0), duration=5400.0)
    assert any("60" in w for w in warnings)

def test_clip_outside_its_format_target_is_flagged():
    warnings = validate_plan(_plan(clip_format="hot_take", out=145.0), duration=5400.0)
    assert any("hot_take" in w for w in warnings)

def test_missing_title_is_flagged():
    warnings = validate_plan(_plan(title=""), duration=5400.0)
    assert any("title" in w.lower() for w in warnings)

def test_out_point_past_source_end_is_flagged():
    warnings = validate_plan(_plan(out=6000.0), duration=5400.0)
    assert any("source" in w.lower() for w in warnings)

def test_overlapping_clips_are_flagged():
    plan = _plan()
    plan["clips"].append({**plan["clips"][0], "in": 120.0, "out": 150.0})
    assert any("overlap" in w.lower() for w in validate_plan(plan, 5400.0))

def test_speaker_label_without_a_speakers_map_is_flagged():
    plan = _plan(speaker_label=True)
    assert any("speakers" in w.lower() for w in validate_plan(plan, 5400.0))

def test_missing_jev_model_is_flagged():
    plan = _plan()
    del plan["jev_model"]
    assert any("jev_model" in w for w in validate_plan(plan, 5400.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_plan.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clipper.plan'`

- [ ] **Step 3: Implement `clipper/plan.py`**

```python
from __future__ import annotations

from clipper.run import Run

# Spec section 8, from retention data by format.
LENGTH_TARGETS: dict[str, tuple[float, float]] = {
    "hot_take": (15.0, 25.0),
    "cold_open": (15.0, 25.0),
    "confession": (20.0, 35.0),
    "debate": (20.0, 35.0),
    "how_to": (25.0, 40.0),
    "list": (25.0, 40.0),
    "story_arc": (30.0, 45.0),
}
HARD_MIN = 10.0
SOFT_MAX = 60.0


def validate_plan(plan: dict, duration: float) -> list[str]:
    warnings: list[str] = []
    if not plan.get("jev_model"):
        warnings.append("plan.json has no jev_model; thresholds cannot be traced to a version.")

    spans: list[tuple[float, float, int]] = []
    for index, clip in enumerate(plan.get("clips", []), start=1):
        start, end = float(clip["in"]), float(clip["out"])
        length = end - start
        label = f"clip {index}"

        if end > duration:
            warnings.append(f"{label}: out-point {end:.1f}s is past the source end ({duration:.1f}s).")
        if length < HARD_MIN:
            warnings.append(f"{label}: {length:.1f}s is under the {HARD_MIN:.0f}s minimum and will be rejected.")
        elif length > SOFT_MAX:
            warnings.append(f"{label}: {length:.1f}s exceeds {SOFT_MAX:.0f}s; worth a second look.")
        else:
            fmt = clip.get("clip_format")
            target = LENGTH_TARGETS.get(fmt)
            if target and not target[0] <= length <= target[1]:
                warnings.append(
                    f"{label}: {length:.1f}s is outside the {fmt} target "
                    f"of {target[0]:.0f}-{target[1]:.0f}s."
                )
        if not clip.get("title"):
            warnings.append(f"{label}: missing a title.")
        if clip.get("speaker_label") and not plan.get("speakers"):
            warnings.append(f"{label}: speaker_label is set but plan.json has no speakers map.")
        spans.append((start, end, index))

    for a_start, a_end, a in sorted(spans):
        for b_start, b_end, b in sorted(spans):
            if a < b and a_start < b_end and b_start < a_end:
                warnings.append(f"clips {a} and {b} overlap.")
    return warnings


def check_plan(run: Run) -> list[str]:
    return validate_plan(run.read_json("plan.json"), run.read_json("source.json")["duration"])
```

- [ ] **Step 4: Write `.claude/skills/clipper/SKILL.md`**

````markdown
---
name: clipper
description: Use when selecting and trimming clips from a clipper run — turns candidates.json into plan.json
---

# Clipper: the plan stage

Deterministic scripts do everything except this. You read `candidates.json` and
the transcript, and you write `plan.json`.

## Inputs

- `runs/<name>/candidates.json` — ranked candidates plus an `uncertain` bucket.
- `runs/<name>/transcript.json` — word-level timings. Trim against these.
- `runs/<name>/source.json` — duration, resolution.

Read candidates first. Only pull transcript slices for candidates you are
seriously considering; do not read the whole transcript.

## What each signal means

- `peak_composite` / `curve` — where the strongest material sits in the candidate.
- `signals.buried_lede` high — the best line is late. **Move the in-point later.**
- `signals.opens_with_windup` high — the opening is setup. **Cut it.**
- `signals.ends_cleanly` low — the thought does not complete. Extend, or drop it.
- `signals.needs_speaker_id` high — set `speaker_label: true` on the clip.
- `backward_extended: true` — the candidate was extended back from an audible
  reaction. The line that caused the reaction is near the start. **Do not trim
  the front of these without reading it.**

## Trimming rules

1. Start on the strongest line. Mid-sentence is fine and often correct.
2. Cut every wind-up. No "so, um", no greetings, no preamble.
3. End on the payoff. Do not let it drag past the peak.
4. Never open on the reaction. Keep the cause inside the clip.

## Length targets

| `clip_format` | Target |
|---|---|
| `hot_take`, `cold_open` | 15-25s |
| `confession`, `debate` | 20-35s |
| `how_to`, `list` | 25-40s |
| `story_arc` | 30-45s |

Under 10s is rejected outright. Over 60s needs a reason.

## The uncertain bucket

Jev flagged these as low-confidence. Read their transcript slices and decide
yourself. Do not skip the bucket; that is where the model deferred to you.

## Output

Write `runs/<name>/plan.json`:

```json
{
  "source": "runs/<name>",
  "profile": "podcast",
  "jev_model": "<copy from candidates.json>",
  "speakers": { "SPEAKER_00": "Ana", "SPEAKER_01": "Marco" },
  "clips": [
    { "in": 874.10, "out": 897.40, "title": "...", "hook": "...",
      "description": "...", "clip_format": "hot_take", "vertical": true,
      "crop_x": "center", "captions": "burn", "speaker_label": true,
      "jev": { "clipworthy": 4.2, "hook_strength": 3.8, "confidence": 0.91 } }
  ]
}
```

Carry `jev_model` across verbatim. Then run `clipper plan <run> --check` and fix
what it reports before rendering.

If the user has their own editing skills loaded, theirs override these defaults.
````

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_plan.py -v`
Expected: 10 passed

- [ ] **Step 6: Commit**

```bash
git add clipper/plan.py .claude/skills/clipper/SKILL.md tests/test_plan.py
git commit -m "feat: plan validation and the Claude plan-stage skill"
```

---

### Task 18: CLI wiring and the `all` pipeline

**Files:**
- Create: `clipper/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: every stage.
- Produces: `main(argv: list[str] | None = None) -> int`, `default_run_name(video: Path) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
from pathlib import Path
import pytest
from clipper.cli import default_run_name, main

def test_default_run_name_includes_date_and_stem():
    name = default_run_name(Path("/videos/Episode 47.mp4"))
    assert name.endswith("-episode-47")
    assert name[:4].isdigit()

def test_no_arguments_prints_usage_and_fails(capsys):
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()

def test_unknown_profile_exits_with_a_helpful_message(capsys, tmp_path):
    code = main(["score", str(tmp_path), "--profile", "vlog"])
    assert code == 1
    assert "podcast" in capsys.readouterr().err

def test_missing_artifact_names_the_producing_stage(capsys, tmp_path):
    (tmp_path / "empty").mkdir()
    code = main(["window", str(tmp_path / "empty")])
    assert code == 1
    assert "transcribe" in capsys.readouterr().err

def test_plan_check_reports_warnings(capsys, tmp_path):
    from clipper.run import Run
    run = Run.create(tmp_path, "ep")
    run.write_json("source.json", {"duration": 5400.0,
                                   "video": {"width": 1920, "height": 1080}})
    run.write_json("plan.json", {"jev_model": "jev-1.13.0", "clips": [
        {"in": 10.0, "out": 15.0, "title": "T", "clip_format": "hot_take"}]})
    code = main(["plan", str(run.root), "--check"])
    assert code == 1
    assert "10" in capsys.readouterr().out

def test_plan_check_passes_a_clean_plan(capsys, tmp_path):
    from clipper.run import Run
    run = Run.create(tmp_path, "ep")
    run.write_json("source.json", {"duration": 5400.0,
                                   "video": {"width": 1920, "height": 1080}})
    run.write_json("plan.json", {"jev_model": "jev-1.13.0", "clips": [
        {"in": 100.0, "out": 120.0, "title": "T", "clip_format": "hot_take"}]})
    assert main(["plan", str(run.root), "--check"]) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clipper.cli'`

- [ ] **Step 3: Implement `clipper/cli.py`**

```python
from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from datetime import date
from pathlib import Path

from clipper.ingest import ingest
from clipper.plan import check_plan
from clipper.preflight import PreflightError, preflight
from clipper.profiles.loader import ProfileError, available_profiles
from clipper.render import run_render
from clipper.run import MissingArtifact, Run
from clipper.score import run_score
from clipper.transcribe import transcribe
from clipper.window import build_windows

RUNS_DIR = Path("runs")


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
            os.environ.setdefault(key.strip(), value.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clipper")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("ingest"); p.add_argument("video"); p.add_argument("--run")
    p = sub.add_parser("transcribe"); p.add_argument("run")
    p.add_argument("--model", default="large-v3"); p.add_argument("--no-diarize", action="store_true")
    p = sub.add_parser("window"); p.add_argument("run")
    p = sub.add_parser("score"); p.add_argument("run")
    p.add_argument("--profile", required=True, choices=available_profiles())
    p = sub.add_parser("plan"); p.add_argument("run"); p.add_argument("--check", action="store_true")
    p = sub.add_parser("render"); p.add_argument("run")
    p.add_argument("--vertical", action="store_true")
    p.add_argument("--captions", default="burn", choices=["burn", "sidecar", "none"])
    p = sub.add_parser("all"); p.add_argument("video")
    p.add_argument("--profile", required=True, choices=available_profiles())
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
            source = run.read_json("source.json")
            windows = build_windows(run.read_json("transcript.json"),
                                    source["energy"], duration=source["duration"])
            run.write_json("windows.json", {"windows": windows})
            print(f"{len(windows)} windows")

        elif args.command == "score":
            run = Run.open(Path(args.run))
            result = asyncio.run(run_score(run, args.profile))
            print(f"{len(result['candidates'])} candidates, "
                  f"{len(result['uncertain'])} uncertain, "
                  f"{result['failed_windows']} failed windows")

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
            result = asyncio.run(run_score(run, args.profile))
            print(f"{len(result['candidates'])} candidates in {run.root}.")
            print("Next: ask Claude to run the plan stage "
                  "(.claude/skills/clipper/SKILL.md), then `clipper render`.")

    except (ProfileError, MissingArtifact, PreflightError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the whole suite**

Run: `pytest -v`
Expected: all tests pass; the live-marked test is deselected

- [ ] **Step 5: Write `README.md`**

```markdown
# JEV Claude Clipper

Turns a long video into short, upload-ready clips. WhisperX transcribes it,
TypeSafe Jev scores every 30-second window for clip-worthiness, Claude selects
and trims, ffmpeg renders with captions.

Design: `docs/superpowers/specs/2026-09-21-jev-claude-clipper-design.md`

## Setup

    pip install -e ".[dev]"
    cp .env.example .env    # add TYPESAFE_API_KEY, optionally HF_TOKEN

ffmpeg must be on PATH and must include libass. Check:

    ffmpeg -filters | grep subtitles

## Use

    clipper all "episode47.mp4" --profile podcast --vertical

Then ask Claude to run the plan stage, and:

    clipper plan runs/2026-09-21-episode47 --check
    clipper render runs/2026-09-21-episode47 --vertical

Profiles: `podcast`, `talking_head`, `lecture`, `stream`.
Re-scoring with a different profile reuses the cached transcript and costs
about two cents.
```

- [ ] **Step 6: Commit**

```bash
git add clipper/cli.py tests/test_cli.py README.md
git commit -m "feat: CLI wiring, all pipeline, and README"
```

---

## Self-Review

**Spec coverage.** §3 architecture → Tasks 2, 18. §4 ingest and rolling-baseline
energy → Tasks 3, 4. §5 transcribe → Task 5. §6 windowing, reaction lag → Task 6.
§7 state, all core and profile questions, ranking, gates, confidence routing,
merging, cost and model versioning → Tasks 7–11. §8 plan, trimming rules, length
targets, writer delegation, `plan.json` → Tasks 16, 17. §9 render, filter order,
captions, speaker label, Windows path handling → Tasks 12–15. §10 CLI and config
→ Task 18. §11 failure modes and tests → distributed, with each failure row
covered by a named test. §12 deferred items are correctly absent.

**Placeholders.** None. Every code step contains runnable code; every test step
contains real assertions.

**Type consistency.** `Run.read_json`/`write_json` used identically everywhere.
Window dict keys match between Task 6 and Tasks 8, 10. Score record shape
(`id`, `failed`, `answers`) is consistent across Tasks 8, 9, 10, 11. `Cue` is
constructed in Task 12 and consumed unchanged in Tasks 13, 15. `Profile` field
names match between Tasks 7, 8, 9, 10. `build_filter_chain` signature matches
between Tasks 14 and 15. `clip_format` values are identical in `core.yaml`
(Task 7), `LENGTH_TARGETS` (Task 17) and the SKILL.md table.

**Known gap, deliberate:** `validate_plan` warns rather than rejects, while
`validate_clip` raises. Clips under 10s are caught in both places — the first as
a warning during `plan --check`, the second as a hard error during render. This
is intended: the plan stage should surface everything at once, the render stage
should refuse to produce a bad file.
