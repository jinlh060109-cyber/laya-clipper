# Silent action moments — Design

Date: 2026-09-23
Status: Approved direction (approach A); remaining calls made as rulings below

## 1. Problem

Laya scores transcript windows, and windows are anchored on transcribed words
(`clipper/window.py`). So:

- A gameplay video with no commentary fails outright: `normalize_transcript`
  raises "Transcription produced no speech".
- In a stream with commentary, a stretch where the streamer goes quiet (a
  clutch, a boss fight) is never windowed, so it can never become a clip.

Laya cannot judge these stretches, because it reads text only. This design adds
a second, speech-free path that finds the stretches with cheap signals and
hands the best ones to Claude, who looks at frames and decides at the plan
stage.

## 2. Scope

Both kinds of gaming video: no speech at all, and commentary with silent gaps.
One mechanism covers both — a silent stretch is found the same way either way.

Out of scope: kill-feed or on-screen text detection, game recognition, local
vision models, changing Laya or the speech path.

## 3. Pipeline

```
ingest → transcribe → windows → action (new) → profile → score → [Claude plan] → render
```

`action` runs after `windows` and before `profile`, on the CLI (`clipper action
<run>`, and inside `clipper all`) and on the web page (a new stage row). It
writes `motion.json` (cached; the video is decoded once) and `action.json`.

## 4. Silent video must survive the speech stages

- `normalize_transcript` returns `{"segments": []}` (with the detected
  language) instead of raising. `transcribe` skips alignment when Whisper found
  no segments.
- `build_windows` already returns `[]` for no words.
- `run_score` with zero windows does not load Laya. It writes empty
  `scores.json` / `candidates.json` with `"laya_model": null` and returns
  `{"scored": 0, "failed": 0, "candidates": 0, "uncertain": 0, "device": "none"}`.
- Web Auto profile with zero windows skips Laya and picks `stream` with
  `fallback: true` (the only profile whose description covers gaming).
- `plan --check` warns about a missing `laya_model` only when the key is
  absent; an explicit `null` (silent video) is not warned about.
- Render: a clip with no words in range skips the burned subtitles (an empty
  `.ass` buys nothing and needs libass).

## 5. Finding silent stretches — `clipper/silence.py`

`silent_spans(transcript, duration, min_gap=8.0) -> list[[start, end]]`

1. Group words into utterances: a new utterance starts after any inter-word
   gap ≥ `min_gap`.
2. Drop *islands*: an utterance of ≤ 5 words lasting ≤ 3 s. These are
   Whisper's inventions over game music ("Thanks for watching!") or a single
   "let's go!" shouted mid-fight — neither should split the action.
   (Ruling: word alignment `score` is not used, because interpolated real words
   also carry 0.0.)
3. Silent spans are the gaps ≥ `min_gap` between the remaining utterances,
   plus the lead-in and tail. No remaining words → `[[0, duration]]`.

## 6. Measuring motion — `clipper/motion.py`

One ffmpeg pass: `fps=2,scale=64:-2,scdet=threshold=10,metadata=print`, stderr
streamed and parsed line by line (`pts_time`, `lavfi.scd.mafd`,
`lavfi.scd.score`). No numpy or Pillow — ffmpeg does the pixel work.

Per whole second `i` (frames with `i <= pts_time < i+1`):

- `motion_raw[i]`: mean mafd (absolute picture change).
- `motion[i]`: `rolling_baseline(motion_raw)` — 0.5 means typical for this
  part of the video, the same normalization as `energy`.
- `cuts[i]`: number of frames with scene score ≥ 10.

Lengths padded/truncated to `int(duration)` like energy. Progress is reported
as `(seconds_done, total_seconds)` from `pts_time`. Skipped entirely when there
are no silent spans (no decode for a video that talks throughout).

## 7. Scoring and choosing — `clipper/action.py`

Windows of 20 s, step 10 s, inside each silent span (a span of 8–20 s is one
window covering it; spans under 8 s never exist). Per window, over its whole
seconds:

- `loudness` = mean of the 3 highest `energy` seconds (peaks matter, not the
  average).
- `motion` = mean `motion`.
- `cut_rate` = `min(1, cuts / 4)`.
- `score = 0.45·loudness + 0.35·motion + 0.20·cut_rate`.
- `static` = mean `motion_raw` < 0.5: menus, pause and loading screens. Static
  windows are never chosen.

Choosing (ruling: thresholds are starting points, visible in `action.json`):

1. Windows with `score ≥ 0.6`, grouped when they overlap or touch, and a group
   is split at 60 s (same cap as `merge.max_seconds`).
2. Each group becomes a candidate spanning its windows; `peak_score` is its
   best window's score, `peak_time` the loudest second in that window.
3. Sorted by `peak_score`, the top 15 are kept.
4. If fewer than 5 qualify, the best remaining non-static windows that do not
   overlap a chosen candidate fill up to 5 — a silent video always gives Claude
   something to look at.

## 8. Frames — contact sheets

For each chosen candidate: 6 frames at `start + (k + 0.5)·len/6`, each pulled
with input seeking (`-ss t -i video -frames:v 1`, 320 px wide), then tiled 3×2
with ffmpeg's `tile` filter via the concat demuxer into
`runs/<name>/frames/action-NN.jpg` (960×360 for 16:9). Temporary single frames
are deleted. One image per candidate keeps Claude's reads to ~15.

## 9. `action.json`

```json
{
  "min_gap": 8.0,
  "silent_seconds": 812.0,
  "spans": [[0.0, 95.5], [140.2, 410.0]],
  "weights": {"loudness": 0.45, "motion": 0.35, "cut_rate": 0.2},
  "threshold": 0.6,
  "candidates": [
    {"id": 0, "start": 150.0, "end": 190.0, "peak_score": 0.81, "peak_time": 171.5,
     "signals": {"loudness": 0.93, "motion": 0.74, "cut_rate": 0.5},
     "sheet": "frames/action-00.jpg",
     "sheet_times": [153.3, 160.0, 166.7, 173.3, 180.0, 186.7]}
  ]
}
```

## 10. Plan stage (skill)

A new SKILL section, "Action moments": if `action.json` has candidates, Read
each `sheet` image (the times are left-to-right, top row first) and decide like
an editor — keep clutch plays, fights, deaths, big reveals; drop menus,
loading screens and cutscenes. Clips taken from it use `clip_format: "action"`
(target 15–30 s, added to `LENGTH_TARGETS`), no `laya` object, `captions:
"none"` when nobody speaks in the range, and the title comes from what is on
screen. Speech candidates and action candidates may cover neighbouring time;
the existing overlap warning applies.

## 11. CLI and web

- `clipper action <run>` prints progress ("Measuring motion: 120/5400s") and
  "N action moments from M min without speech". `clipper all` runs it after
  windows.
- Web: `STAGES` gains `"action"` after `"windows"`; `Stages` gains an
  `action(run, progress)` field; progress uses the existing `job.progress`
  with `stage: "action"`. The summary adds the action line; `job.result`
  carries `action_candidates` and `silent_seconds`.

## 12. Errors

- ffmpeg failure in the motion pass or a frame pull raises `RuntimeError` with
  `ffmpeg_tail` of stderr; the stage fails like any other (web shows it on the
  action row). A partial `motion.json` is never written (write after parse).
- A missing contact sheet is fatal for that run, not skipped — a candidate
  Claude cannot see is useless.

## 13. Testing

Offline, with real ffmpeg on synthetic lavfi video (as `test_ingest.py`
already does): static `color` vs moving `testsrc2`, concatenated, to assert
motion is higher and a cut is found at the join; a contact sheet has the
expected size. Pure-function tests for `silent_spans` (islands, edges, no
words) and action scoring/choosing (threshold, grouping, 60 s split, top-N,
fill-to-5, static exclusion). Stage tests for empty transcript, zero-window
score, Auto with zero windows, web stage order/progress, CLI output, plan
`laya_model: null`, render skipping empty subtitles.
