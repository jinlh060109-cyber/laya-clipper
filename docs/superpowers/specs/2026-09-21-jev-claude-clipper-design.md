# JEV Claude Clipper — Design

Date: 2026-09-21
Status: Approved design, ready for implementation planning

## 1. Overview

A local tool that turns a long video file into short, upload-ready clips.

The pipeline transcribes the video, uses TypeSafe's **Jev** model to score every
part of the transcript for clip-worthiness, and uses **Claude** (orchestrating in
Claude Code, with user-supplied video editing skills) to select clips, trim their
boundaries, and write their titles and descriptions. FFmpeg renders the results
with captions and optional vertical reframing.

The division of labour is the central idea. Jev is a non-autoregressive "System
One" model: it cannot write text, but it answers typed questions about state in
one parallel pass, at roughly 100ms and $0.042 per million input tokens with free
output. That makes it ideal for fanning out hundreds of calibrated judgments over
a transcript. Claude does what Jev structurally cannot: read, reason about
boundaries, and write.

### Goals

- Input: a local video file. Output: rendered clips with captions and metadata.
- Four selectable content profiles: podcast, talking-head, lecture, stream.
- Runs as a Claude Code skill over plain Python scripts.
- Resumable: re-scoring with a different profile must not re-transcribe.
- Inspectable: every stage leaves a JSON artifact on disk.
- Test suite runs offline with no API key and no media files committed.

### Non-goals

- Face-tracked vertical reframing (static crop only; see §12).
- Downloading from URLs (see §12).
- Uploading or publishing anywhere.
- A GUI or web interface.

## 2. Terminology

| Term | Meaning |
|---|---|
| **Window** | A fixed-length overlapping slice of transcript, the unit Jev scores. |
| **Candidate** | Adjacent high-scoring windows merged into one region. |
| **Clip** | A candidate that Claude selected and trimmed into final in/out points. |
| **Profile** | A YAML file defining the Jev question bundle for one content type. |

## 3. Architecture

Every stage is a CLI subcommand that reads JSON artifacts from a run directory
and writes JSON artifacts back. Nothing passes between stages in memory, and no
large artifact passes through Claude's context.

```
video.mp4
  -> ingest     -> source.json, audio.wav
  -> transcribe -> transcript.json
  -> window     -> windows.json
  -> score      -> scores.json, candidates.json      (Jev)
  -> plan       -> plan.json                         (Claude + editing skills)
  -> render     -> clips/*.mp4, *.srt, *.json        (ffmpeg)
```

### Run directory

```
runs/2026-09-21-ep47/
  source.json
  audio.wav
  transcript.json
  windows.json
  scores.json
  candidates.json
  plan.json
  clips/
    01-the-thing-nobody-says.mp4
    01-the-thing-nobody-says.srt
    01-the-thing-nobody-says.json
```

`runs/` is gitignored.

### Project layout

```
clipper/
  cli.py          ingest | transcribe | window | score | plan | render | all
  ingest.py       ffprobe metadata, audio extraction, per-second RMS energy
  transcribe.py   WhisperX ASR, forced alignment, pyannote diarization
  window.py       transcript -> overlapping windows
  score.py        Jev client, concurrency, retries, merge into candidates
  profiles/
    podcast.yaml
    talking_head.yaml
    lecture.yaml
    stream.yaml
  captions.py     word timings -> ASS (burn-in) and SRT (sidecar)
  render.py       ffmpeg cut, optional 9:16 crop, subtitle burn
  llm.py          OpenAI-compatible adapter: claude | deepseek | kimi
.claude/skills/
  clipper/SKILL.md    how Claude drives the pipeline
  <user editing skills>
docs/superpowers/specs/
runs/                 gitignored
tests/
pyproject.toml
```

Python throughout: WhisperX and `typesafe-sdk` are Python, and ffmpeg is driven
by subprocess regardless.

## 4. Stage: ingest

Probes the source with `ffprobe` and extracts audio.

- `source.json`: duration, container, video stream (codec, width, height, fps,
  rotation), audio stream (codec, sample rate, channels), and the absolute source
  path.
- `audio.wav`: 16 kHz mono PCM, which is what WhisperX consumes.
- Per-second RMS audio energy, normalized to 0..1 over the whole file, stored in
  `source.json` as an array. The stream profile needs this: reaction spikes live
  in the audio, not in the words.

Rotation metadata is read here and respected at render time, so portrait phone
footage is not cropped sideways.

## 5. Stage: transcribe

WhisperX, in three passes: ASR, forced alignment for word-level timestamps, and
pyannote diarization for speaker labels.

`transcript.json`:

```json
{
  "language": "en",
  "model": "large-v3",
  "diarized": true,
  "segments": [
    { "start": 872.41, "end": 878.90, "speaker": "SPEAKER_01",
      "text": "...",
      "words": [ { "word": "So", "start": 872.41, "end": 872.58,
                   "score": 0.91, "speaker": "SPEAKER_01" } ] }
  ]
}
```

Word-level timings are load-bearing twice over: Claude trims clip boundaries
against them in §8, and captions are generated from them in §9.

If no HuggingFace token is configured, diarization is skipped with a warning,
`diarized` is set to `false`, and every word is labelled `SPEAKER_00`. All four
profiles still run; the podcast profile simply loses its speaker-dependent
signals.

This is the only slow stage. It is cached by source file hash plus model name,
so re-runs of everything downstream are effectively free.

## 6. Stage: window

Slices the transcript into overlapping windows.

- Target 45 seconds of content per window, stepping every 15 seconds.
- Edges snap outward to word boundaries; no window begins mid-word.
- Each window carries the preceding 20 seconds of transcript as context, so
  "needs prior context" is answerable.
- Each window carries its mean normalized audio energy and its fractional
  position in the source.

A 90-minute source produces roughly 360 windows.

## 7. Stage: score (Jev)

### State

One Jev call per window. State is text-only, as Jev requires:

```json
{
  "profile": "podcast",
  "window": {
    "start": 872.4,
    "end": 918.1,
    "text": "SPEAKER_01: ...\nSPEAKER_00: ..."
  },
  "preceding": "...last 20 seconds of transcript...",
  "position": 0.43,
  "audio_energy": 0.71
}
```

### Core question bundle

Asked by every profile. Jev evaluates each question in parallel and in isolation
against the same state, so a dozen questions cost about the same as one and
create no context rot. We therefore decompose aggressively rather than asking a
single fuzzy "is this good?".

| Key | Type | Definition |
|---|---|---|
| `clipworthy` | Score, 5 levels | Headline ranking signal: would a stranger stop scrolling. |
| `hook_strength` | Score, 5 levels | How well the window's first sentence works as an opener. |
| `self_contained` | Noul | Understandable with nothing said before it. |
| `needs_context` | Noul | References something not present in the window. |
| `starts_midthought` | Noul | Boundary repair hint for the plan stage. |
| `ends_cleanly` | Noul | The thought completes before the window ends. |
| `category` | Choice | story / opinion / explanation / joke / reaction / filler |
| `emotional_intensity` | Score, 5 levels | |

### Profile additions

| Profile | Added questions |
|---|---|
| `podcast` | `disagreement` (Noul), `personal_story` (Noul) |
| `talking_head` | `hot_take` (Noul), `direct_address` (Noul) |
| `lecture` | `complete_explanation` (Noul), `actionable` (Noul), `requires_visual` (Noul) |
| `stream` | `reaction_spike` (Noul), `narratable` (Noul) |

`requires_visual` matters for lectures: it flags clips that die without the
slides on screen, which are exactly the clips that look good on paper and fail
when posted.

Profiles are YAML. Tuning a rubric means editing a file and re-running `score` —
a few cents and a few seconds, with no retranscription.

Score primitives use 5 ordered levels, within Jev's 2..10 range. Choice options
stay well under the 255 cardinality limit. Combined state plus questions is on
the order of 850 tokens per call, far under the 64k combined and 32k
single-question budgets.

### Ranking and merging

Composite score per window, from `clipworthy` and `hook_strength`, penalized by
`needs_context` and `starts_midthought`. Windows above threshold that overlap or
abut are merged into candidates, capped at 90 seconds. Each candidate retains its
per-window score curve so the plan stage can see where the peak sits.

### Confidence is a routing signal

Jev returns calibrated confidence on Choice and Score answers. Low-confidence
windows are **not** silently dropped. They are marked `uncertain` and written to
a separate bucket in `candidates.json` for Claude to adjudicate by reading the
transcript. This is the intended complementarity: Jev reports honestly when it is
unsure, and the model that can read decides.

### Throughput, cost, versioning

- Async with a semaphore of 16 and a token-bucket limiter beneath the 1,200
  requests/minute ceiling; `retry-after` honored on HTTP 429.
- Roughly 306k input tokens for a 90-minute video: about **1.3 cents**.
- Scoring wall time: roughly 20-30 seconds.
- `jev-latest` by default. The resolved model version is recorded in
  `scores.json` and carried into `plan.json`, because thresholds tuned against one
  version must not silently drift onto another.

A window that fails after retries is marked `failed` and the run continues.
Losing 3 of 360 windows is not a reason to discard an eight-minute
transcription.

## 8. Stage: plan (Claude)

The one stage Claude performs itself rather than a script running unattended.

Claude reads `candidates.json` plus the transcript slices for each candidate, and
writes `plan.json`. Four jobs:

1. Select which candidates survive.
2. Trim in/out points to natural sentence edges using word timings, guided by
   `starts_midthought` and `ends_cleanly`.
3. Write a title, hook, and description per clip.
4. Adjudicate the `uncertain` bucket.

User-supplied editing skills in `.claude/skills/` are invoked here. This is the
point where "how I like clips cut" lives as instructions Claude follows, rather
than as constants in a Python file.

### Writer delegation

`llm.py` is a single OpenAI-compatible client with a configurable base URL,
covering DeepSeek and Kimi alike. `--writer deepseek` routes the per-clip
title/hook/description generation there; omitting it means Claude writes them
inline. Selection and trimming always stay with Claude, because those require the
editing skills in context.

### plan.json

```json
{
  "source": "runs/2026-09-21-ep47",
  "profile": "podcast",
  "jev_model": "jev-1.13.0",
  "clips": [
    {
      "in": 874.10,
      "out": 919.65,
      "title": "The thing nobody says about funding",
      "hook": "Everyone gets this backwards.",
      "description": "...",
      "vertical": true,
      "crop_x": "center",
      "captions": "burn",
      "jev": { "clipworthy": 4.2, "hook_strength": 3.8, "confidence": 0.91 }
    }
  ]
}
```

`plan.json` is editable by hand. Re-running `render` against a hand-edited plan
is a supported workflow.

## 9. Stage: render

### Cutting

Always re-encode; never stream-copy. Stream copy snaps to keyframes, and after
Claude has trimmed to the word, that precision is not worth discarding. Captions
require a re-encode regardless.

### Filter chain

Order is crop, then scale, then subtitles, so caption sizing is computed in
output space:

```
crop=ih*9/16:ih:x:0, scale=1080:1920, subtitles=clip.ass
```

Horizontal output skips the crop and scale. Vertical is per-clip in `plan.json`,
defaulting to the `--vertical` CLI flag. `crop_x` accepts `center` or an explicit
pixel offset. Face tracking is out of scope.

### Captions

Generated from the word timings already in `transcript.json`:

- `clip.ass` with word-level highlighting, for burn-in.
- `clip.srt` sidecar, written unconditionally — it costs nothing and is wanted
  eventually.

`captions` per clip is `burn`, `sidecar`, or `none`.

### Windows path handling

Two known ffmpeg traps, both of which this project hits directly because the
working directory contains spaces:

1. The `subtitles=` filter escapes Windows drive-letter colons and spaces
   poorly. Subtitle files are therefore staged into a short temporary directory
   with an ASCII, space-free path before the filter references them.
2. A plain ffmpeg build may lack libass, and thus the `subtitles` filter,
   entirely. Preflight checks for the filter itself via `ffmpeg -filters`, not
   merely for the presence of the binary.

## 10. CLI and configuration

```
clipper ingest     <video> [--run NAME]
clipper transcribe <run> [--model large-v3] [--no-diarize]
clipper window     <run>
clipper score      <run> --profile podcast
clipper plan       <run>              # normally invoked by Claude
clipper render     <run> [--vertical] [--captions burn|sidecar|none]
clipper all        <video> --profile podcast [--vertical]
```

Configuration via environment, read from `.env`:

- `TYPESAFE_API_KEY` — required for `score`.
- `HF_TOKEN` — optional; absent disables diarization.
- `WRITER_BASE_URL`, `WRITER_API_KEY`, `WRITER_MODEL` — optional, for
  DeepSeek/Kimi delegation.
- `FFMPEG_PATH`, `FFPROBE_PATH` — optional overrides.

`.env` is gitignored. No key is ever written into a run artifact.

## 11. Error handling and tests

### Failure modes

| Condition | Behaviour |
|---|---|
| `ffmpeg`/`ffprobe` missing | Fail immediately with an install hint. |
| `subtitles` filter unavailable | Fail at preflight, naming libass. |
| No `HF_TOKEN` | Warn, skip diarization, continue. |
| Silent or speechless audio | Clear message, no empty artifacts written. |
| Jev 429 | Backoff honoring `retry-after`. |
| Jev window fails after retries | Mark `failed`, continue the run. |
| Nothing clears threshold | Report top scores rather than writing empty `clips/`. |
| Clip boundary past source end | Clamp to duration. |

### Testing

The artifact contract makes nearly everything a pure function over small JSON.

- Unit tests with JSON fixtures, no video and no network: windowing math, merge
  and composite scoring, sentence-boundary snapping, ASS and SRT generation, crop
  arithmetic, boundary clamping, profile YAML loading and validation.
- Jev client tested against recorded response fixtures. One live smoke test
  behind an opt-in pytest marker.
- Render tests generate a 10-second source at test time with ffmpeg's `lavfi`
  sources, covering the real ffmpeg invocation without committing media.

`pytest` passes offline with no API key.

## 12. Deferred extensions

Recorded deliberately, not scoped into this build.

- **URL input.** A `yt-dlp` front-door so a YouTube URL can substitute for a
  local file. Small and self-contained; omitted because the stated input is a
  local file.
- **Face-tracked vertical crop.** Active-speaker detection with smoothing, so a
  two-person podcast cuts between faces. Substantially harder than static crop.
- **Bilingual captions.** If added, translation must batch approximately 20
  subtitle lines per request rather than one at a time.
- **Semantic chaptering.** Topic-boundary detection as a navigation artifact.
  Considered and rejected as the primary segmentation strategy, because missed
  boundaries silently swallow good material on unstructured footage.

## 13. References

- TypeSafe docs: https://docs.typesafe.ai/introduction
- Introducing System One Models and Jev: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- Practical Jev guide: https://dev.to/valyuai/how-to-use-jev-a-practical-guide-to-typesafes-system-one-model-g5e
- Prior art, ffmpeg and libass lessons: https://github.com/op7418/Youtube-clipper-skill
