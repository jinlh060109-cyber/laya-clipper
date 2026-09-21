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

The selection criteria in §7 are derived from research into clipping practice,
summarized with citations in §14. Where a design decision follows from that
research, it is marked **[R]** and the reasoning is given.

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
- Trend or topicality scoring. Opus Clip includes "Trend" as one of four virality
  axes, but we have no trend data source, and a model asked to guess at trends
  returns confident noise. Excluded deliberately. **[R]**

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
  ingest.py       ffprobe metadata, audio extraction, rolling-baseline energy
  transcribe.py   WhisperX ASR, forced alignment, pyannote diarization
  window.py       transcript -> overlapping windows
  score.py        Jev client, concurrency, retries, merge into candidates
  profiles/
    podcast.yaml
    talking_head.yaml
    lecture.yaml
    stream.yaml
  captions.py     word timings -> ASS (burn-in) and SRT (sidecar)
  render.py       ffmpeg cut, optional 9:16 crop, subtitle burn, speaker label
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
- Per-second RMS audio energy, stored in `source.json` as an array.

### Energy normalization **[R]**

Energy is normalized against a **rolling 5-minute baseline**, not against the
whole file. Twitch highlight-detection practice is explicit about this: absolute
loudness is a poor signal because a consistently loud speaker saturates the
scale everywhere, while a genuine spike inside a quiet stretch fails to register.
The stored value is a local z-score, clamped to 0..1.

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

- **30 seconds of content per window, stepping every 10 seconds.** **[R]**
- Edges snap outward to word boundaries; no window begins mid-word.
- Each window carries the preceding 20 seconds of transcript as context, so
  "needs prior context" is answerable.
- Each window carries its mean normalized audio energy, its peak energy, and
  `energy_peak_offset` — where within the window the peak falls.
- Each window carries its fractional position in the source.

A 90-minute source produces roughly 540 windows.

### Why 30/10 **[R]**

The window is the scoring unit, so it should approximate the thing being judged.
The retention data converges on 15–30 seconds as the range with the highest
completion rates, with sharp drop-off past 45 seconds; human podcast clippers
independently arrive at "inside about 45 seconds." A 30-second window is
therefore roughly clip-shaped, and a 10-second step gives three-fold overlap so
that boundary placement has 10-second resolution before Claude refines it to the
word.

### Reaction lag **[R]**

Reactions follow their cause. Twitch systems shift detection windows backwards
because chat lags the moment by 3–10 seconds; laughter lags a joke by one to
three. `energy_peak_offset` exists so that merging can extend a candidate
*backwards* from a spike rather than starting at it. A clip that opens on the
laugh has already missed the line that caused it.

## 7. Stage: score (Jev)

### State

One Jev call per window. State is text-only, as Jev requires:

```json
{
  "profile": "podcast",
  "window": {
    "start": 872.4,
    "end": 902.1,
    "text": "SPEAKER_01: ...\nSPEAKER_00: ..."
  },
  "preceding": "...last 20 seconds of transcript...",
  "position": 0.43,
  "energy_mean": 0.44,
  "energy_peak": 0.86,
  "energy_peak_offset": 0.71
}
```

### Core question bundle

Asked by every profile. Jev evaluates each question in parallel and in isolation
against the same state, so fifteen questions cost about the same as one and
create no context rot. We therefore decompose aggressively rather than asking a
single fuzzy "is this good?".

The bundle is organized around the three virality axes that survive scrutiny —
**Hook**, **Flow**, **Value** — plus the failure modes that clipping practice
identifies as the usual causes of death. **[R]**

**Hook — does it stop the scroll**

| Key | Type | Definition |
|---|---|---|
| `hook_strength` | Score, 5 levels | How well the window's opening line works as a first line. |
| `hook_type` | Choice | pattern_interrupt / direct_promise / question / contradiction / story_open / none |
| `open_loop` | Noul | The opening raises a question the viewer needs resolved. |

**Flow — does it hold together**

| Key | Type | Definition |
|---|---|---|
| `self_contained` | Noul | Understandable with nothing said before it. |
| `needs_context` | Noul | References something not present in the window. |
| `ends_cleanly` | Noul | The thought completes before the window ends. |
| `payoff` | Noul | Delivers on what the opening implies. |

**Value — is it worth watching**

| Key | Type | Definition |
|---|---|---|
| `clipworthy` | Score, 5 levels | Headline ranking signal: would a stranger stop scrolling. |
| `poster_line` | Noul | Contains a sentence you could put on a poster. |
| `concrete_specifics` | Noul | Contains a specific number, name, or worked example. |
| `audible_reaction` | Noul | Someone laughs, reacts, or the room responds. |
| `emotional_intensity` | Score, 5 levels | |
| `clip_format` | Choice | cold_open / debate / how_to / story_arc / hot_take / confession / list / filler |

**Failure modes — explicit negative signals**

| Key | Type | Definition |
|---|---|---|
| `opens_with_windup` | Noul | The first sentence is setup, backstory, or throat-clearing rather than the point. |
| `buried_lede` | Noul | The most interesting moment arrives well after the window starts. |

### Two of these deserve explanation

**`opens_with_windup` replaces the obvious `starts_midthought`.** **[R]** An
earlier draft of this design penalized windows that began mid-sentence. That was
wrong, and it would have actively suppressed good clips: professional clippers
deliberately start on the strongest line, *mid-sentence if necessary*, skipping
the setup entirely. Beginning mid-thought is a technique, not a defect. The real
killer is the wind-up — the slow build, the generic greeting, the buried lede —
and since 50–60% of viewer drop-off happens within the first three seconds, that
is the thing worth measuring. `buried_lede` is its companion: it tells Claude the
material is good but the in-point should move later, which is precisely the
trimming action to take.

**`audible_reaction` and `poster_line` encode the most-repeated human
heuristic.** **[R]** Clipping practice is consistent that you should listen for
the reaction, not the topic — cut where something *happened*, not where something
correct was said. The practical markers clippers report using are a sentence you
could put on a poster, a moment the room went quiet or cracked up, a real
disagreement, and a specific number or example. `poster_line`,
`audible_reaction`, `disagreement` and `concrete_specifics` are those four
markers made measurable.

### Profile additions

| Profile | Added questions |
|---|---|
| `podcast` | `disagreement` (Noul), `personal_story` (Noul), `needs_speaker_id` (Noul) |
| `talking_head` | `hot_take` (Noul), `direct_address` (Noul) |
| `lecture` | `complete_explanation` (Noul), `one_concept` (Noul), `actionable` (Noul), `requires_visual` (Noul) |
| `stream` | `reaction_spike` (Noul), `narratable` (Noul), `needs_gameplay_context` (Noul) |

- `needs_speaker_id` **[R]** — insufficient on-screen speaker context is a
  top-cited reason clips fail. When true, render draws a speaker label (§9).
- `requires_visual` — flags lecture clips that die without the slides on screen,
  which are exactly the clips that read well as transcript and fail when posted.
- `one_concept` **[R]** — educational segmentation research is clear that a clip
  should carry exactly one idea; two half-explained ideas beat neither.
- `needs_gameplay_context` — the stream analogue: incomprehensible without
  knowing the game state.

Profiles are YAML. Tuning a rubric means editing a file and re-running `score` —
a few cents and a few seconds, with no retranscription.

Score primitives use 5 ordered levels, within Jev's 2..10 range. Choice options
stay well under the 255 cardinality limit. Combined state plus questions is on
the order of 750 tokens per call, far under the 64k combined and 32k
single-question budgets.

### Ranking and merging

Composite score per window:

- **Positive:** `clipworthy` and `hook_strength` dominate, lifted by
  `poster_line`, `open_loop`, `payoff`, `concrete_specifics` and
  `audible_reaction`.
- **Negative:** `needs_context` and `opens_with_windup` penalize.
  `buried_lede` does *not* penalize — it annotates, because it describes a
  fixable in-point rather than bad material.
- **Gate:** `self_contained` below a floor disqualifies outright. A clip nobody
  can follow is not a clip.

Windows above threshold that overlap or abut are merged into candidates, **capped
at 60 seconds** **[R]**, extended backwards where `energy_peak_offset` indicates
a reaction lag. Each candidate retains its per-window score curve so the plan
stage can see where the peak sits.

**Energy alone never promotes a candidate.** **[R]** A spike must be corroborated
by a transcript-derived signal. Twitch systems require the same corroboration,
and without it the stream profile produces clips that are loud and empty.

### Confidence is a routing signal

Jev returns calibrated confidence on Choice and Score answers. Low-confidence
windows are **not** silently dropped. They are marked `uncertain` and written to
a separate bucket in `candidates.json` for Claude to adjudicate by reading the
transcript. This is the intended complementarity: Jev reports honestly when it is
unsure, and the model that can read decides.

### Throughput, cost, versioning

- Async with a semaphore of 16 and a token-bucket limiter beneath the 1,200
  requests/minute ceiling; `retry-after` honored on HTTP 429.
- Roughly 405k input tokens for a 90-minute video: about **1.7 cents**.
- Scoring wall time: roughly 30-45 seconds, rate-limit bound rather than
  latency bound.
- `jev-latest` by default. The resolved model version is recorded in
  `scores.json` and carried into `plan.json`, because thresholds tuned against one
  version must not silently drift onto another.

A window that fails after retries is marked `failed` and the run continues.
Losing 3 of 540 windows is not a reason to discard an eight-minute
transcription.

## 8. Stage: plan (Claude)

The one stage Claude performs itself rather than a script running unattended.

Claude reads `candidates.json` plus the transcript slices for each candidate, and
writes `plan.json`. Five jobs:

1. Select which candidates survive.
2. Trim in/out points to final positions using word timings.
3. Write a title, hook, and description per clip.
4. Adjudicate the `uncertain` bucket.
5. Map speaker IDs to display names where `needs_speaker_id` is set.

User-supplied editing skills in `.claude/skills/` are invoked here. This is the
point where "how I like clips cut" lives as instructions Claude follows, rather
than as constants in a Python file.

### Trimming rules **[R]**

These are defaults; user editing skills override them.

- **Move the in-point to the strongest line.** Where `buried_lede` is high, the
  clip starts later than the candidate does. Starting mid-sentence is permitted
  and often correct.
- **Cut the wind-up.** Where `opens_with_windup` is high, drop the setup.
- **End at the peak.** Stop on the payoff; do not let the clip drag past it.
- **Never open on the reaction.** Where the candidate was extended backwards
  from an energy spike, keep the cause inside the clip.

### Length targets **[R]**

Target **20–45 seconds**, by format, from retention data:

| `clip_format` | Target |
|---|---|
| `hot_take`, `cold_open` | 15–25s |
| `confession`, `debate` | 20–35s |
| `how_to`, `list` | 25–40s |
| `story_arc` | 30–45s |

Below 10 seconds a clip reads as incomplete and is rejected. Above 60 seconds it
is flagged for a second look rather than rejected outright, since an exceptional
segment can carry the length.

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
  "speakers": { "SPEAKER_00": "Ana", "SPEAKER_01": "Marco" },
  "clips": [
    {
      "in": 874.10,
      "out": 897.40,
      "title": "The thing nobody says about funding",
      "hook": "Everyone gets this backwards.",
      "description": "...",
      "clip_format": "hot_take",
      "vertical": true,
      "crop_x": "center",
      "captions": "burn",
      "speaker_label": true,
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
- `clip.srt` sidecar, written unconditionally.

**Burn-in is the default,** not an option to opt into. **[R]** Burned-in captions
measure 15–25% higher retention, and captioned video is markedly more likely to
be watched to completion. `captions` per clip may still be set to `sidecar` or
`none`.

### Speaker label **[R]**

When a clip has `speaker_label: true`, render draws a small persistent name label
using the `speakers` map from `plan.json`, switching on diarized speaker change.
One ASS style, bottom-left, no animation. This exists because missing on-screen
speaker context is one of the most-cited reasons a podcast clip fails, and the
fix is a single line of subtitle rendering.

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
| Clip shorter than 10s after trimming | Reject with reason recorded. |

### Testing

The artifact contract makes nearly everything a pure function over small JSON.

- Unit tests with JSON fixtures, no video and no network: windowing math,
  rolling-baseline energy normalization, backward extension from an energy peak,
  merge and composite scoring, the `self_contained` gate, length-target
  enforcement, ASS and SRT generation, speaker-label switching, crop arithmetic,
  boundary clamping, profile YAML loading and validation.
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
- **Outcome feedback.** Recording per-clip performance after posting and using it
  to re-weight the composite score. The highest-value extension by some distance,
  since every threshold in §7 is currently a prior rather than a measurement.

## 13. Calibration

Every weight and threshold in §7 is an informed starting point, not a measured
one. Two things make them cheap to correct, and both are deliberate design
choices rather than happy accidents:

1. Re-scoring costs under two cents and under a minute, because `transcript.json`
   is cached.
2. `scores.json` retains every raw Jev answer, not just the composite, so a
   weighting change is a recomputation rather than a re-run.

The intended workflow is to run a few known episodes, compare the ranking against
clips you would have picked by hand, and adjust the profile YAML. The questions
are the stable part of the design; the weights are not.

## 14. Research basis

Findings behind the **[R]** decisions above.

**Clip length.** Retention data converges on 15–30 seconds for the highest
completion rates, often exceeding 80%, with dramatic drop-off past 45 seconds;
per-format targets of 15–20s for tips, 25–40s for tutorials, 18–28s for comedy
and 30–45s for storytelling. Human podcast clippers independently recommend
landing "inside about 45 seconds." Drove the 30s window, the 60s candidate cap,
and the length targets in §8.

**The first three seconds.** 50–60% of viewers who leave do so within three
seconds; strong three-second retention correlates with several times more
impressions. Named hook failure modes: slow build, generic greeting, buried lede,
clickbait mismatch. Drove `hook_strength`, `hook_type`, `open_loop`,
`opens_with_windup`, `buried_lede`, and `payoff`.

**Start on the strongest line, mid-sentence if needed.** Professional clipping
guidance is explicit that clips should open on the best line and skip setup
entirely. Directly overturned an earlier `starts_midthought` penalty in this
design, which would have suppressed correctly-cut clips.

**Listen for the reaction, not the topic.** The most consistent human heuristic:
cut where something happened, not where something correct was said. Reported
markers are a poster-worthy sentence, a moment the room reacted, a genuine
disagreement, and a specific number or example. Drove `poster_line`,
`audible_reaction`, `disagreement`, `concrete_specifics`.

**Clip archetypes.** Seven repeatable formats — cold-open hook, two-person
debate, "here's how" answer, story arc, hot-seat line, relatable confession,
list-in-a-clip. Drove the `clip_format` Choice, which in turn drives per-format
length targets.

**Why clips fail.** Requiring prior context, opening on a wind-up, running long,
missing on-screen speaker context, and choosing personally interesting over
inherently watchable. Drove the `self_contained` gate, `needs_speaker_id`, and
the speaker label in §9.

**Stream highlight detection.** Audio and chat spike detection normalizes against
a rolling channel baseline rather than absolute values, shifts detection windows
backwards because chat lags the moment by 3–10 seconds, and requires
corroboration from a second signal before promoting a spike. Drove rolling-baseline
normalization in §4, `energy_peak_offset` and backward extension in §6–7, and the
corroboration rule.

**Captions.** Burned-in captions measure 15–25% higher retention, with captioned
video substantially more likely to be watched to completion. Made burn-in the
default rather than an option.

**Educational segmentation.** Short single-concept videos outperform long ones on
both engagement and outcomes; the operative principle is one idea per segment.
Drove `one_concept` in the lecture profile.

**Commercial scoring precedent.** Opus Clip's virality score spans Hook, Flow,
Value and Trend. The first three organize the core bundle in §7. Trend is
excluded: no data source, and guessing produces confident noise.

### Sources

- TypeSafe docs: https://docs.typesafe.ai/introduction
- Introducing System One Models and Jev: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- Practical Jev guide: https://dev.to/valyuai/how-to-use-jev-a-practical-guide-to-typesafes-system-one-model-g5e
- Opus Clip virality score: https://help.opus.pro/docs/article/virality-score
- Shorts hook formulas: https://www.opus.pro/blog/youtube-shorts-hook-formulas
- Ideal Shorts length and retention: https://www.opus.pro/blog/ideal-youtube-shorts-length-format-retention
- What to actually clip from every episode: https://luminaclippers.com/blog/podcast-clip-ideas
- Podcast clipping guide: https://clipping.net/blog/podcast-clipping-guide
- Chat velocity as a highlight signal: https://clipme.com/blog/chat-velocity-viral-moments
- PogChampNet, Twitch chat as highlight labels: https://medium.com/@farzatv/pogchampnet-how-we-used-twitch-chat-deep-learning-to-create-automatic-game-highlights-with-only-61ed7f7b22d4
- Micro-lecture video guidance: https://teaching.uic.edu/cate-teaching-guides/digital-learning/micro-lecture-videos/
- Prior art, ffmpeg and libass lessons: https://github.com/op7418/Youtube-clipper-skill
