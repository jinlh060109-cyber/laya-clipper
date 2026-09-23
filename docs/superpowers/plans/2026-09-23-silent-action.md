# Silent Action Moments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gameplay with no speech — whole videos or quiet stretches in a stream — becomes clip candidates that Claude judges from contact-sheet frames.

**Architecture:** A new `action` stage after `windows`: `silence.py` finds stretches of ≥ 8 s without speech, `motion.py` measures per-second picture change and scene cuts with one ffmpeg `scdet` pass, `action.py` scores and chooses windows from loudness + motion + cuts, and `frames.py` tiles 6 frames per candidate into one JPG. Results go to `action.json`; the skill tells Claude to look at the sheets. The speech path is taught to survive a transcript with no words.

**Tech Stack:** Python 3.12 stdlib, ffmpeg (`scdet`, `tile`, concat demuxer), pytest. No new dependencies (numpy and Pillow are not installed).

**Spec:** `docs/superpowers/specs/2026-09-23-silent-action-design.md`

## Global Constraints

- No new Python dependencies; ffmpeg does all pixel work.
- Silent gap: `min_gap = 8.0` s. Island: ≤ 5 words and ≤ 3.0 s.
- Motion pass: `fps=2,scale=64:-2,scdet=threshold=10,metadata=print`; cut = scene score ≥ 10.
- Action windows 20 s, step 10 s. `score = 0.45·loudness + 0.35·motion + 0.20·cut_rate`; `cut_rate = min(1, cuts/4)`; loudness = mean of top-3 energy seconds; static = mean raw mafd < 0.5.
- Choose: threshold 0.6, groups split at 60 s, top 15, fill to 5.
- Contact sheet: 6 frames at `start + (k+0.5)·len/6`, 320 px wide, tiled 3×2, `runs/<name>/frames/action-NN.jpg`.
- Stage order: `upload, ingest, transcribe, windows, action, profile, score`.
- Tests run with `.venv/Scripts/python.exe -m pytest -q`. Real-ffmpeg tests skip when `shutil.which("ffmpeg")` is None, as `tests/test_ingest.py` does.

## Review Focus

1. A run directory whose path contains an apostrophe or non-ASCII characters — the concat list must not break (list uses bare file names; cwd is the temp dir). Test in Task 5.
2. Re-running `action` after a previous run chose more candidates — stale `frames/action-*.jpg` must be removed. Test in Task 6.
3. A video shorter than 8 s, or a transcript whose words run past `duration` — no crash, no negative spans. Tests in Tasks 2 and 4.
4. Auto profile on a silent video — must not load Laya just to vote on zero windows. Test in Task 7.
5. A mostly-talking video — the motion pass must not run when there are no silent spans. Test in Task 6.

---

### Task 1: The speech path survives a video with no speech

**Files:**
- Modify: `clipper/transcribe.py` (normalize_transcript, transcribe)
- Modify: `clipper/stages/score.py` (run_score)
- Modify: `clipper/plan.py` (validate_plan laya_model warning)
- Modify: `clipper/render.py` (run_render: skip burn when no cues)
- Test: `tests/test_transcribe.py`, `tests/test_stage_score.py`, `tests/test_plan.py`, `tests/test_render.py`

**Interfaces:**
- Produces: `normalize_transcript(...)` returns `{"language", "model", "diarized", "segments": []}` for no speech. `run_score` on zero windows returns `{"scored": 0, "failed": 0, "candidates": 0, "uncertain": 0, "device": "none"}` and writes `candidates.json` with `"laya_model": None`.

- [ ] **Step 1: Failing tests**

`tests/test_transcribe.py` — replace `test_empty_segments_raise` with:

```python
def test_no_speech_gives_an_empty_transcript_not_an_error():
    out = normalize_transcript({"segments": []}, "small", False, "en")
    assert out == {"language": "en", "model": "small", "diarized": False, "segments": []}
```

`tests/test_stage_score.py`:

```python
def test_zero_windows_skip_laya_and_write_empty_candidates(tmp_path):
    run = Run.create(tmp_path, "silent")
    run.write_json("windows.json", {"windows": []})
    run.write_json("transcript.json", {"language": "en", "segments": []})

    def boom(*args, **kwargs):
        raise AssertionError("Laya must not load for zero windows")

    import clipper.stages.score as stage
    stage.load_agent, saved = boom, stage.load_agent
    try:
        result = run_score(run, "stream")
    finally:
        stage.load_agent = saved
    assert result == {"scored": 0, "failed": 0, "candidates": 0,
                      "uncertain": 0, "device": "none"}
    cands = run.read_json("candidates.json")
    assert cands["laya_model"] is None and cands["candidates"] == [] and cands["uncertain"] == []
```

`tests/test_plan.py`:

```python
def test_an_explicit_null_laya_model_is_not_warned_about():
    plan = {"laya_model": None, "clips": [{"in": 0, "out": 20, "title": "t"}]}
    assert not any("laya_model" in p for p in validate_plan(plan, 100.0))


def test_a_missing_laya_model_is_still_warned_about():
    plan = {"clips": [{"in": 0, "out": 20, "title": "t"}]}
    assert any("laya_model" in p for p in validate_plan(plan, 100.0))
```

`tests/test_render.py` (follow the file's existing fakes for `preflight`/`render_clip`; the assertion is what matters):

```python
def test_a_clip_with_no_speech_is_rendered_without_burned_subtitles(tmp_path, monkeypatch):
    run = <run with source.json, transcript {"segments": []}, plan with one 0-20 s clip>
    chains = []
    monkeypatch.setattr(render, "render_clip",
                        lambda ffmpeg, source, clip, target, chain, cwd=None: chains.append(chain))
    run_render(run, captions="burn")
    assert "subtitles" not in chains[0]
```

- [ ] **Step 2: Run, expect FAIL** — `.venv/Scripts/python.exe -m pytest -q tests/test_transcribe.py tests/test_stage_score.py tests/test_plan.py tests/test_render.py`

- [ ] **Step 3: Implement**

`clipper/transcribe.py`, in `normalize_transcript`, replace the raise with:

```python
    segments = result.get("segments") or []
    if not segments:
        # Gameplay without commentary is a real input, not an error: the
        # action stage finds its moments from picture and sound instead.
        return {"language": language, "model": model,
                "diarized": diarized, "segments": []}
```

In `transcribe`, wrap alignment and diarization in `if result["segments"]:` (alignment of nothing fails in whisperx), keeping `diarized = False` otherwise.

`clipper/stages/score.py`, after reading `windows`:

```python
    if not windows:
        # Nothing was said, so there is nothing for Laya to read.
        empty = {"profile": profile.name, "laya_model": None}
        run.write_json("scores.json", {**empty, "windows": []})
        run.write_json("candidates.json", {**empty, "candidates": [], "uncertain": []})
        return {"scored": 0, "failed": 0, "candidates": 0, "uncertain": 0, "device": "none"}
```

`clipper/plan.py`:

```python
    if "laya_model" not in plan:
        warnings.append(...same text...)
    elif plan["laya_model"] and not plan["laya_model"].get("confidence_calibrated", True):
```

`clipper/render.py`: `ass = render_ass(cues, out_height, labelled) if mode == "burn" and cues else None`.

- [ ] **Step 4: Run, expect PASS**, then full suite.
- [ ] **Step 5: Commit** `fix: a video with no speech is a valid input, not an error`

---

### Task 2: Silent spans — `clipper/silence.py`

**Files:** Create `clipper/silence.py`; Test `tests/test_silence.py`

**Interfaces:**
- Produces: `silent_spans(transcript: dict, duration: float, min_gap: float = 8.0) -> list[list[float]]` — sorted, non-overlapping `[start, end]` pairs, each ≥ `min_gap`.

- [ ] **Step 1: Failing tests**

```python
from clipper.silence import silent_spans


def words(*spans):
    return {"segments": [{"words": [{"word": "w", "start": s, "end": e} for s, e in spans]}]}


def talk(start, end, step=0.5):
    """Continuous speech: a word every `step` seconds."""
    out, t = [], start
    while t < end:
        out.append((t, min(end, t + step)))
        t += step
    return out


def test_no_words_means_the_whole_video_is_silent():
    assert silent_spans({"segments": []}, 120.0) == [[0.0, 120.0]]


def test_gaps_of_at_least_min_gap_are_silent_and_shorter_ones_are_not():
    t = words(*talk(0, 10), *talk(15, 30), *talk(45, 60))
    assert silent_spans(t, 60.0) == [[30.0, 45.0]]


def test_lead_in_and_tail_count():
    t = words(*talk(20, 40))
    assert silent_spans(t, 100.0) == [[0.0, 20.0], [40.0, 100.0]]


def test_a_short_isolated_utterance_does_not_split_the_silence():
    t = words(*talk(0, 10), (50.0, 50.4), (50.5, 51.0), *talk(90, 100))
    assert silent_spans(t, 100.0) == [[10.0, 90.0]]


def test_a_long_isolated_utterance_does_split_it():
    t = words(*talk(0, 10), *talk(50, 55), *talk(90, 100))
    assert silent_spans(t, 100.0) == [[10.0, 50.0], [55.0, 90.0]]


def test_a_video_shorter_than_min_gap_has_no_spans():
    assert silent_spans({"segments": []}, 5.0) == []


def test_words_past_the_end_are_clamped():
    t = words(*talk(0, 10), *talk(98, 104))
    assert silent_spans(t, 100.0) == [[10.0, 98.0]]
```

- [ ] **Step 2: Run, expect FAIL** (module missing).

- [ ] **Step 3: Implement**

```python
from __future__ import annotations

ISLAND_WORDS = 5
ISLAND_SECONDS = 3.0


def _utterances(words: list[dict], min_gap: float) -> list[tuple[float, float, int]]:
    """(start, end, word count) runs of words with no gap of `min_gap` or more."""
    out: list[list[float]] = []
    for word in sorted(words, key=lambda w: w["start"]):
        if out and word["start"] - out[-1][1] < min_gap:
            out[-1][1] = max(out[-1][1], word["end"])
            out[-1][2] += 1
        else:
            out.append([word["start"], word["end"], 1])
    return [(s, e, int(n)) for s, e, n in out]


def silent_spans(transcript: dict, duration: float, min_gap: float = 8.0) -> list[list[float]]:
    """Stretches of at least `min_gap` seconds in which nobody really speaks.

    A short isolated utterance does not count as speech: over game music it is
    usually Whisper inventing "Thanks for watching!", and a lone "let's go!"
    mid-fight is part of the action, not a reason to split it.
    """
    words = [w for seg in transcript.get("segments") or [] for w in seg.get("words") or []]
    speech = [(min(s, duration), min(e, duration))
              for s, e, n in _utterances(words, min_gap)
              if not (n <= ISLAND_WORDS and e - s <= ISLAND_SECONDS)]
    spans: list[list[float]] = []
    cursor = 0.0
    for start, end in speech:
        if start - cursor >= min_gap:
            spans.append([round(cursor, 3), round(start, 3)])
        cursor = max(cursor, end)
    if duration - cursor >= min_gap:
        spans.append([round(cursor, 3), round(duration, 3)])
    return spans
```

- [ ] **Step 4: PASS**. **Step 5: Commit** `feat: find stretches without speech`

---

### Task 3: Motion — `clipper/motion.py`

**Files:** Create `clipper/motion.py`; Test `tests/test_motion.py`

**Interfaces:**
- Consumes: `clipper.energy.rolling_baseline`, `clipper.ingest.ffmpeg_tail`.
- Produces: `parse_scdet(lines, seconds, progress=None) -> tuple[list[float], list[int]]` (raw mafd per second, cuts per second) and `measure_motion(ffmpeg: Path, video: Path, seconds: int, progress=None) -> dict` with keys `motion_raw`, `motion`, `cuts`, each of length `seconds`.

- [ ] **Step 1: Failing tests**

```python
import shutil
import subprocess

import pytest

from clipper.motion import measure_motion, parse_scdet


def frame(t, mafd, score):
    return [f"[Parsed_metadata_3 @ 0] frame:0 pts:0 pts_time:{t}",
            f"[Parsed_metadata_3 @ 0] lavfi.scd.mafd={mafd}",
            f"[Parsed_metadata_3 @ 0] lavfi.scd.score={score}"]


def test_frames_are_averaged_per_whole_second_and_cuts_counted():
    lines = frame(0, 1.0, 0) + frame(0.5, 3.0, 0) + frame(1.0, 20.0, 19.0) + frame(1.5, 4.0, 2.0)
    raw, cuts = parse_scdet(lines, 3)
    assert raw == [2.0, 12.0, 12.0]  # second 2 has no frames: carried forward
    assert cuts == [0, 1, 0]


def test_scdets_own_summary_line_is_not_a_second_cut():
    lines = frame(1.0, 20.0, 19.0) + ["[Parsed_scdet_2 @ 0] lavfi.scd.score: 19.051, lavfi.scd.time: 1"]
    assert parse_scdet(lines, 2)[1] == [0, 1]


def test_frames_past_the_last_second_are_ignored_and_progress_is_reported():
    seen = []
    parse_scdet(frame(0, 1, 0) + frame(1, 1, 0) + frame(9, 5, 50),
                2, progress=lambda d, t: seen.append((d, t)))
    assert seen[-1] == (2, 2)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_real_video_moves_more_after_a_static_start_and_the_join_is_a_cut(tmp_path):
    video = tmp_path / "v.mp4"
    subprocess.run([shutil.which("ffmpeg"), "-v", "error",
                    "-f", "lavfi", "-i", "color=c=gray:s=320x180:d=4:r=10",
                    "-f", "lavfi", "-i", "testsrc2=s=320x180:d=4:r=10",
                    "-filter_complex", "[0][1]concat=n=2:v=1[v]", "-map", "[v]",
                    str(video)], check=True)
    out = measure_motion(shutil.which("ffmpeg"), video, 8)
    assert len(out["motion_raw"]) == len(out["motion"]) == len(out["cuts"]) == 8
    assert max(out["motion_raw"][:3]) < 0.5 < min(out["motion_raw"][5:])
    assert sum(out["cuts"][3:6]) >= 1


def test_ffmpeg_failure_is_a_runtime_error_quoting_ffmpeg(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("needs ffmpeg")
    with pytest.raises(RuntimeError, match="Motion"):
        measure_motion(ffmpeg, tmp_path / "missing.mp4", 5)
```

- [ ] **Step 2: FAIL**.

- [ ] **Step 3: Implement**

```python
from __future__ import annotations

import re
import subprocess
from collections import deque
from pathlib import Path

from clipper.energy import rolling_baseline
from clipper.ingest import ffmpeg_tail

CUT_SCORE = 10.0
FILTER = "fps=2,scale=64:-2,scdet=threshold=10,metadata=print"
_PTS = re.compile(r"pts_time:(\S+)")


def _value(line: str, key: str) -> float | None:
    if key not in line:
        return None
    try:
        return float(line.rsplit("=", 1)[1])
    except ValueError:
        return None


def parse_scdet(lines, seconds: int, progress=None) -> tuple[list[float], list[int]]:
    """Mean frame difference and scene-cut count per whole second."""
    sums, counts, cuts = [0.0] * seconds, [0] * seconds, [0] * seconds
    second: int | None = None
    for line in lines:
        match = _PTS.search(line)
        if match:
            try:
                t = float(match.group(1))
            except ValueError:
                second = None
                continue
            now = int(t) if 0 <= t < seconds else None
            if progress and now is not None and now != second:
                progress(now + 1, seconds)
            second = now
            continue
        if second is None:
            continue
        mafd = _value(line, "lavfi.scd.mafd=")
        if mafd is not None:
            sums[second] += mafd
            counts[second] += 1
            continue
        score = _value(line, "lavfi.scd.score=")
        if score is not None and score >= CUT_SCORE:
            cuts[second] += 1
    raw: list[float] = []
    for total, n in zip(sums, counts):
        # A second with no sampled frame (variable frame rate, or the tail)
        # repeats the last known value rather than reading as a freeze.
        raw.append(round(total / n, 4) if n else (raw[-1] if raw else 0.0))
    return raw, cuts


def measure_motion(ffmpeg: Path, video: Path, seconds: int, progress=None) -> dict:
    """One decode of the video: per-second picture change, normalized, and cuts."""
    proc = subprocess.Popen(
        [str(ffmpeg), "-v", "info", "-i", str(video), "-vf", FILTER, "-an", "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace")
    tail: deque[str] = deque(maxlen=40)

    def lines():
        for line in proc.stderr:
            tail.append(line)
            yield line

    raw, cuts = parse_scdet(lines(), seconds, progress)
    if proc.wait() != 0:
        raise RuntimeError(f"Motion measurement failed on {video}:\n{ffmpeg_tail(''.join(tail))}")
    if progress:
        progress(seconds, seconds)
    return {"motion_raw": raw, "motion": rolling_baseline(raw), "cuts": cuts}
```

- [ ] **Step 4: PASS**. **Step 5: Commit** `feat: measure per-second motion and scene cuts with ffmpeg`

---

### Task 4: Score and choose — `clipper/action.py`

**Files:** Create `clipper/action.py`; Test `tests/test_action.py`

**Interfaces:**
- Consumes: spans from Task 2, arrays from Task 3, `source["energy"]`.
- Produces: `action_windows(spans, energy, motion, motion_raw, cuts) -> list[dict]` (keys `start, end, score, peak_time, static, signals{loudness, motion, cut_rate}`), `choose_action(windows) -> list[dict]` (keys `id, start, end, peak_score, peak_time, signals`), constants `WEIGHTS`, `THRESHOLD`.

- [ ] **Step 1: Failing tests**

```python
from clipper.action import THRESHOLD, action_windows, choose_action


def flat(n, value):
    return [value] * n


def arrays(n, energy=0.5, motion=0.5, raw=3.0, cuts=0):
    return flat(n, energy), flat(n, motion), flat(n, raw), flat(n, cuts)


def test_a_long_span_is_cut_into_20s_windows_every_10s_ending_flush():
    wins = action_windows([[0.0, 45.0]], *arrays(60))
    assert [(w["start"], w["end"]) for w in wins] == [(0, 20), (10, 30), (20, 40), (25, 45)]


def test_a_short_span_is_one_window_and_under_8s_none():
    assert [(w["start"], w["end"]) for w in action_windows([[5.0, 17.0]], *arrays(30))] == [(5, 17)]
    assert action_windows([[0.0, 7.0]], *arrays(30)) == []


def test_the_score_weighs_peak_loudness_motion_and_cuts():
    energy, motion, raw, cuts = arrays(20, energy=0.0, motion=1.0)
    energy[4] = energy[9] = energy[14] = 0.9
    cuts[3] = 8
    (w,) = action_windows([[0.0, 20.0]], energy, motion, raw, cuts)
    assert w["signals"] == {"loudness": 0.9, "motion": 1.0, "cut_rate": 1.0}
    assert w["score"] == round(0.45 * 0.9 + 0.35 + 0.20, 4)
    assert w["peak_time"] == 4.5 and w["static"] is False


def test_a_frozen_picture_is_static():
    (w,) = action_windows([[0.0, 20.0]], *arrays(20, raw=0.2))
    assert w["static"] is True


def w(start, end, score, static=False):
    return {"start": start, "end": end, "score": score, "peak_time": start + 1,
            "static": static, "signals": {}}


def test_hot_windows_that_touch_merge_and_split_at_60s():
    wins = [w(t, t + 20, 0.8) for t in range(0, 100, 10)]
    chosen = sorted(choose_action(wins), key=lambda c: c["start"])
    assert [(c["start"], c["end"]) for c in chosen] == [(0, 60), (50, 110)]


def test_static_windows_are_never_chosen_even_to_fill():
    assert choose_action([w(0, 20, 0.95, static=True)]) == []


def test_top_15_by_peak_score():
    wins = [w(t * 100, t * 100 + 20, 0.6 + t / 100) for t in range(20)]
    chosen = choose_action(wins)
    assert len(chosen) == 15
    assert chosen[0]["peak_score"] == 0.79 and [c["id"] for c in chosen] == list(range(15))


def test_fewer_than_5_hot_are_filled_with_the_best_non_overlapping_rest():
    wins = [w(0, 20, 0.9), w(10, 30, 0.5), w(100, 120, 0.4), w(200, 220, 0.3),
            w(300, 320, 0.2), w(400, 420, 0.1)]
    chosen = choose_action(wins)
    assert [(c["start"], c["peak_score"]) for c in chosen] == [
        (0, 0.9), (100, 0.4), (200, 0.3), (300, 0.2), (400, 0.1)]


def test_threshold_is_the_spec_value():
    assert THRESHOLD == 0.6
```

- [ ] **Step 2: FAIL**.

- [ ] **Step 3: Implement**

```python
from __future__ import annotations

import math

WINDOW_SECONDS = 20.0
STEP_SECONDS = 10.0
MIN_SPAN = 8.0
WEIGHTS = {"loudness": 0.45, "motion": 0.35, "cut_rate": 0.20}
THRESHOLD = 0.6
MAX_SECONDS = 60.0
TOP_N = 15
MIN_CANDIDATES = 5
STATIC_BELOW = 0.5
CUTS_FOR_FULL = 4


def _starts(start: float, end: float) -> list[float]:
    if end - start <= WINDOW_SECONDS:
        return [start]
    n = math.ceil((end - start - WINDOW_SECONDS) / STEP_SECONDS)
    starts = [start + k * STEP_SECONDS for k in range(n)]
    return starts + [end - WINDOW_SECONDS]


def _seconds(start: float, end: float, n: int) -> list[int]:
    lo, hi = max(0, math.ceil(start)), min(n, math.floor(end))
    if lo < hi:
        return list(range(lo, hi))
    return [min(n - 1, max(0, int((start + end) / 2)))] if n else []


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def action_windows(spans, energy, motion, motion_raw, cuts) -> list[dict]:
    """Score every window inside the silent spans from sound and picture alone."""
    windows: list[dict] = []
    for span_start, span_end in spans:
        if span_end - span_start < MIN_SPAN:
            continue
        for start in _starts(span_start, span_end):
            end = min(span_end, start + WINDOW_SECONDS)
            secs = _seconds(start, end, len(energy))
            if not secs:
                continue
            signals = {
                "loudness": round(_mean(sorted((energy[i] for i in secs), reverse=True)[:3]), 3),
                "motion": round(_mean(motion[i] for i in secs), 3),
                "cut_rate": round(min(1.0, sum(cuts[i] for i in secs) / CUTS_FOR_FULL), 3),
            }
            windows.append({
                "start": round(start, 3), "end": round(end, 3),
                "score": round(sum(WEIGHTS[k] * v for k, v in signals.items()), 4),
                "peak_time": max(secs, key=lambda i: energy[i]) + 0.5,
                "static": _mean(motion_raw[i] for i in secs) < STATIC_BELOW,
                "signals": signals,
            })
    return windows


def _candidate(group: list[dict]) -> dict:
    best = max(group, key=lambda w: w["score"])
    return {"start": group[0]["start"], "end": max(w["end"] for w in group),
            "peak_score": best["score"], "peak_time": best["peak_time"],
            "signals": best["signals"]}


def _overlaps(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def choose_action(windows: list[dict]) -> list[dict]:
    """Group hot windows into candidates; top up to a handful when few are hot."""
    live = [w for w in windows if not w["static"]]
    groups: list[list[dict]] = []
    for win in sorted((w for w in live if w["score"] >= THRESHOLD), key=lambda w: w["start"]):
        group = groups[-1] if groups else None
        if group and win["start"] <= group[-1]["end"] and win["end"] - group[0]["start"] <= MAX_SECONDS:
            group.append(win)
        else:
            groups.append([win])
    chosen = sorted((_candidate(g) for g in groups),
                    key=lambda c: c["peak_score"], reverse=True)[:TOP_N]
    # A silent video with nothing "hot" still gives Claude something to look at.
    for win in sorted(live, key=lambda w: w["score"], reverse=True):
        if len(chosen) >= MIN_CANDIDATES:
            break
        if not any(_overlaps(win, c) for c in chosen):
            chosen.append(_candidate([win]))
    chosen.sort(key=lambda c: c["peak_score"], reverse=True)
    return [{"id": i, **c} for i, c in enumerate(chosen)]
```

- [ ] **Step 4: PASS**. **Step 5: Commit** `feat: score and choose action windows from loudness, motion and cuts`

---

### Task 5: Contact sheets — `clipper/frames.py`

**Files:** Create `clipper/frames.py`; Test `tests/test_frames.py`

**Interfaces:**
- Produces: `sheet_times(start, end, n=6) -> list[float]`, `contact_sheet(ffmpeg: Path, video: Path, start: float, end: float, out: Path) -> list[float]`.

- [ ] **Step 1: Failing tests**

```python
import shutil
import subprocess

import pytest

from clipper.frames import contact_sheet, sheet_times

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def test_times_are_centred_in_six_equal_slices():
    assert sheet_times(10.0, 22.0) == [11.0, 13.0, 15.0, 17.0, 19.0, 21.0]


def make_video(path):
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi",
                    "-i", "testsrc2=s=640x360:d=6:r=10", str(path)], check=True)


@needs_ffmpeg
def test_a_sheet_is_three_by_two_frames_of_320px(tmp_path):
    video = tmp_path / "v.mp4"
    make_video(video)
    out = tmp_path / "frames" / "action-00.jpg"
    times = contact_sheet(shutil.which("ffmpeg"), video, 0.0, 6.0, out)
    assert times == sheet_times(0.0, 6.0)
    probe = subprocess.run([shutil.which("ffprobe"), "-v", "error", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", str(out)],
                           capture_output=True, text=True, check=True)
    assert probe.stdout.strip() == "960,360"
    assert [p.name for p in out.parent.iterdir()] == ["action-00.jpg"]


@needs_ffmpeg
def test_paths_with_apostrophes_and_non_ascii_work(tmp_path):
    base = tmp_path / "Jev's 游戏 run"
    base.mkdir()
    video = base / "v.mp4"
    make_video(video)
    out = base / "frames" / "action-00.jpg"
    contact_sheet(shutil.which("ffmpeg"), video, 1.0, 5.0, out)
    assert out.stat().st_size > 0


@needs_ffmpeg
def test_a_missing_video_is_a_runtime_error(tmp_path):
    with pytest.raises(RuntimeError, match="frame"):
        contact_sheet(shutil.which("ffmpeg"), tmp_path / "nope.mp4", 0, 6, tmp_path / "s.jpg")
```

- [ ] **Step 2: FAIL**.

- [ ] **Step 3: Implement**

```python
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from clipper.ingest import ffmpeg_tail

COLS, ROWS = 3, 2
FRAME_WIDTH = 320


def sheet_times(start: float, end: float, n: int = COLS * ROWS) -> list[float]:
    step = (end - start) / n
    return [round(start + (k + 0.5) * step, 2) for k in range(n)]


def _ffmpeg(args: list[str], cwd: Path, what: str) -> None:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Could not {what}:\n{ffmpeg_tail(result.stderr)}")


def contact_sheet(ffmpeg: Path, video: Path, start: float, end: float, out: Path) -> list[float]:
    """Six frames across [start, end], tiled 3x2 into one JPG. Returns their times."""
    times = sheet_times(start, end)
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    video = Path(video).resolve()
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        work = Path(tmp)
        names = []
        for k, t in enumerate(times):
            name = f"f{k}.jpg"
            _ffmpeg([str(ffmpeg), "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
                     "-frames:v", "1", "-vf", f"scale={FRAME_WIDTH}:-2", "-q:v", "3",
                     "-y", name], work, f"pull the frame at {t:.1f}s from {video.name}")
            if not (work / name).exists():
                raise RuntimeError(f"Could not pull the frame at {t:.1f}s from {video.name}.")
            names.append(name)
        # Bare names in the list, run from the temp dir: a run path containing
        # an apostrophe would otherwise break the concat demuxer's quoting.
        (work / "list.txt").write_text("".join(f"file '{n}'\n" for n in names), encoding="utf-8")
        _ffmpeg([str(ffmpeg), "-v", "error", "-f", "concat", "-safe", "0", "-i", "list.txt",
                 "-vf", f"tile={COLS}x{ROWS}", "-frames:v", "1", "-q:v", "3", "-y", str(out)],
                work, f"tile the frames for {out.name}")
    return times
```

- [ ] **Step 4: PASS**. **Step 5: Commit** `feat: contact sheets of six frames per action candidate`

---

### Task 6: The action stage and CLI

**Files:**
- Create: `clipper/stages/action.py`
- Modify: `clipper/cli.py` (new `action` command; `all` runs it; progress printer factory)
- Test: `tests/test_stage_action.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: Tasks 2–5; `clipper.preflight.preflight(require_subtitles=False).ffmpeg`.
- Produces: `run_action(run: Run, progress=None) -> dict` with keys `silent_seconds: float`, `spans: int`, `candidates: int`; writes `motion.json` (only when spans exist) and `action.json` (spec §9 shape).

- [ ] **Step 1: Failing tests** (`tests/test_stage_action.py`)

```python
import shutil
import subprocess

import pytest

import clipper.stages.action as stage
from clipper.run import Run

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def silent_run(tmp_path, seconds=30, transcript=None):
    run = Run.create(tmp_path, "game")
    video = run.path("video.mp4")
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi",
                    "-i", f"testsrc2=s=320x180:d={seconds}:r=10", str(video)], check=True)
    run.write_json("source.json", {"path": str(video), "duration": float(seconds),
                                   "energy": [0.5] * seconds})
    run.write_json("transcript.json", transcript or {"language": "en", "segments": []})
    return run


@needs_ffmpeg
def test_a_silent_video_gets_candidates_with_sheets(tmp_path):
    run = silent_run(tmp_path)
    result = stage.run_action(run)
    action = run.read_json("action.json")
    assert result == {"silent_seconds": 30.0, "spans": 1, "candidates": len(action["candidates"])}
    assert action["spans"] == [[0.0, 30.0]] and action["threshold"] == 0.6
    assert 1 <= len(action["candidates"]) <= 5
    first = action["candidates"][0]
    assert first["sheet"] == "frames/action-00.jpg" and len(first["sheet_times"]) == 6
    assert run.path(first["sheet"]).exists()
    assert run.exists("motion.json")


@needs_ffmpeg
def test_stale_sheets_are_removed_and_motion_is_reused(tmp_path, monkeypatch):
    run = silent_run(tmp_path)
    stage.run_action(run)
    run.path("frames/action-14.jpg").write_bytes(b"old")
    monkeypatch.setattr(stage, "measure_motion",
                        lambda *a, **k: pytest.fail("motion.json should be reused"))
    stage.run_action(run)
    assert not run.path("frames/action-14.jpg").exists()


def test_a_video_that_talks_throughout_never_decodes_motion(tmp_path, monkeypatch):
    run = Run.create(tmp_path, "talk")
    words = [{"word": "w", "start": t / 2, "end": t / 2 + 0.4} for t in range(60)]
    run.write_json("source.json", {"path": "x.mp4", "duration": 30.0, "energy": [0.5] * 30})
    run.write_json("transcript.json", {"segments": [{"words": words}]})
    monkeypatch.setattr(stage, "preflight", lambda **k: pytest.fail("no tools needed"))
    assert stage.run_action(run) == {"silent_seconds": 0.0, "spans": 0, "candidates": 0}
    assert run.read_json("action.json")["candidates"] == []
    assert not run.exists("motion.json")
```

`tests/test_cli.py` (match its existing fakes for `all`, which monkeypatch names in `clipper.cli`):

```python
def test_action_prints_what_it_found(tmp_path, monkeypatch, capsys):
    run = Run.create(tmp_path, "g")
    monkeypatch.setattr(cli, "run_action",
                        lambda r, progress=None: {"silent_seconds": 600.0, "spans": 3, "candidates": 7})
    assert main(["action", str(run.root)]) == 0
    assert "7 action moments from 10.0 min without speech" in capsys.readouterr().out


def test_action_says_so_when_there_is_no_silence(tmp_path, monkeypatch, capsys):
    run = Run.create(tmp_path, "g")
    monkeypatch.setattr(cli, "run_action",
                        lambda r, progress=None: {"silent_seconds": 0.0, "spans": 0, "candidates": 0})
    assert main(["action", str(run.root)]) == 0
    assert "No stretch of 8 s or more without speech" in capsys.readouterr().out
```

and extend the existing `all` test so its fake `run_action` records a call and the test asserts it ran between windows and score.

- [ ] **Step 2: FAIL**.

- [ ] **Step 3: Implement** `clipper/stages/action.py`:

```python
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
    for old in frames.glob("action-*.jpg") if frames.exists() else []:
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
```

`clipper/cli.py`:

```python
def _printer(label: str):
    """One stderr line, rewritten in place, at most every 1% of the work."""
    def show(done: int, total: int) -> None:
        if done == total or done % max(1, total // 100) == 0:
            end = "\n" if done == total else ""
            print(f"\r{label}: {done}/{total}", end=end, file=sys.stderr, flush=True)
    return show


_print_progress = _printer("Scoring windows")


def _report_action(result: dict) -> None:
    if not result["spans"]:
        print("No stretch of 8 s or more without speech; no action moments.")
    else:
        print(f"{result['candidates']} action moments from "
              f"{result['silent_seconds'] / 60:.1f} min without speech")
```

Parser: `p = sub.add_parser("action"); p.add_argument("run")`. Command:

```python
        elif args.command == "action":
            run = Run.open(Path(args.run))
            _report_action(run_action(run, progress=_printer("Measuring motion (s)")))
```

In `all`, after `write_windows(run)`: `_report_action(run_action(run, progress=_printer("Measuring motion (s)")))`. Import `from clipper.stages.action import run_action`.

- [ ] **Step 4: PASS** + full suite. **Step 5: Commit** `feat: action stage and clipper action command`

---

### Task 7: Web — stage row, progress, Auto on silent video

**Files:**
- Modify: `clipper/web/jobs.py`, `clipper/web/index.html`
- Test: `tests/test_jobs.py`, `tests/test_web.py`

**Interfaces:**
- Consumes: `run_action(run, progress)` from Task 6; `write_windows` returns the window list.
- Produces: `STAGES = ("upload", "ingest", "transcribe", "windows", "action", "profile", "score")`; `Stages.action: Callable[[Run, Callable[[int, int], None]], dict]` (field after `windows`); `job.result` gains `action_candidates`, `silent_seconds`.

- [ ] **Step 1: Failing tests** (`tests/test_jobs.py`; add `action=step("action", {"silent_seconds": 90.0, "spans": 2, "candidates": 3})` to `recorder_stages`, update the order test to `["ingest", "transcribe", "windows", "action", "score"]`):

```python
def test_action_runs_after_windows_and_its_result_is_reported(run):
    log = []
    status = _go(JobRunner(recorder_stages(log)), run)
    assert [n for n, _ in log].index("action") == 3
    assert status["result"]["action_candidates"] == 3
    assert status["result"]["silent_seconds"] == 90.0


def test_action_progress_is_visible_while_it_runs(run):
    gate, seen = threading.Event(), {}
    stages = recorder_stages([])

    def action(r, progress):
        progress(40, 100)
        seen["status"] = runner.status()
        return {"silent_seconds": 1.0, "spans": 1, "candidates": 1}

    stages.action = action
    runner = JobRunner(stages)
    _go(runner, run)
    assert seen["status"]["progress"] == {"stage": "action", "done": 40, "total": 100}


def test_auto_on_a_silent_video_picks_stream_without_loading_laya(run):
    log = []
    stages = recorder_stages(log)
    stages.windows = lambda r: []
    status = _go(JobRunner(stages), run, profile="auto")
    assert "load_agent" not in [n for n, _ in log]
    assert status["profile"] == "stream" and status["fallback"] is True
```

(`tests/test_web.py`: add `action=ok({"silent_seconds": 0.0, "spans": 0, "candidates": 0})` to `gated_stages`; the existing "page names every stage" test then covers the page.)

- [ ] **Step 2: FAIL**.

- [ ] **Step 3: Implement**

`jobs.py`: add `"action"` to `STAGES` after `"windows"`; add the `action` field to `Stages` after `windows`; in `default_stages` import `run_action` and build `Stages(...)` with keywords, `action=run_action`. In `_execute`:

```python
            windows = self._step(job, "windows", lambda: stages.windows(run))
            found = self._step(job, "action", lambda: stages.action(run, reporter("action")))

            def pick() -> dict:
                nonlocal agent
                if settings["profile"] != "auto":
                    return {"profile": settings["profile"], "votes": None, "fallback": False}
                if windows is not None and len(windows) == 0:
                    # Nothing was said, so Laya has nothing to vote on; gameplay
                    # is the only kind of video that is silent throughout.
                    return {"profile": "stream", "votes": None, "fallback": True}
                agent = stages.load_agent(run, settings)
                return stages.choose_profile(run, agent[0])
```

with `reporter(stage)` replacing the inner `report`:

```python
            def reporter(stage: str):
                def report(done: int, total: int) -> None:
                    # In memory only: status() is polled every second.
                    with self._lock:
                        job.progress = {"stage": stage, "done": done, "total": total}
                return report
```

(define it before the windows step), score uses `reporter("score")`, and when finishing:

```python
            if isinstance(found, dict):
                result = {**result, "action_candidates": found.get("candidates", 0),
                          "silent_seconds": found.get("silent_seconds", 0.0)}
```

`index.html`: add `["action", "Find action"]` after windows in `STAGES`; make the progress display use `status.progress.stage` rather than assuming score (read the existing code); in `summarize`, when `r.action_candidates` is defined add a line: `r.action_candidates + " action moments from " + (r.silent_seconds / 60).toFixed(1) + " min without speech."`.

- [ ] **Step 4: PASS** + full suite. **Step 5: Commit** `feat: web runs the action stage and handles silent videos`

---

### Task 8: Skill, plan targets, README

**Files:** Modify `.claude/skills/clipper/SKILL.md`, `clipper/plan.py` (`LENGTH_TARGETS`), `README.md`; Test `tests/test_plan.py`

- [ ] **Step 1: Failing test**

```python
def test_action_clips_have_a_15_to_30s_target():
    plan = {"laya_model": None, "clips": [{"in": 0, "out": 40, "title": "t", "clip_format": "action"}]}
    assert any("action target" in p for p in validate_plan(plan, 100.0))
```

- [ ] **Step 2: FAIL**. **Step 3:** add `"action": (15.0, 30.0)` to `LENGTH_TARGETS`. Add to SKILL.md after "The uncertain bucket":

```markdown
## Action moments (no speech)

If `runs/<name>/action.json` has candidates, these are stretches where nobody
speaks, found from loudness, on-screen motion and scene cuts. Laya never saw
them. For each one, Read its `sheet` image: six frames, left to right, top row
first, taken at `sheet_times`. Judge them like an editor: keep clutch plays,
fights, deaths, big reveals and anything a viewer would rewind; drop menus,
loading screens, cutscenes and walking around. Their `signals` only say it was
loud and busy, not that it was good.

Clips taken from them use `clip_format: "action"` (15-30s), no `laya` object,
and `captions: "none"` unless someone speaks in the range. Write the title from
what is on screen. For a video with no speech at all, set `"laya_model": null`
in `plan.json`.
```

README: under Use, one paragraph: gameplay without commentary works; the `action` stage finds quiet moments and Claude looks at frames; `clipper action <run>` re-runs it.

- [ ] **Step 4: PASS** + full suite. **Step 5: Commit** `docs: skill reads action moments; action clip length target`

---

### Task 9: Real run

- [ ] Build a ~3 min synthetic "gameplay" video with ffmpeg: static gray 40 s, `testsrc2` + loud `sine` bursts 40 s, `mandelbrot` 40 s, static 40 s, no speech. Run `clipper all <video> --profile stream --model small`; confirm transcribe survives, `action.json` has candidates in the busy parts, sheets exist, Read one sheet. Record timings in the ledger.
