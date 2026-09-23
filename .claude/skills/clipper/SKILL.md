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

## Run settings

If `runs/<name>/settings.json` exists, the run was started from the web page.
Read it before choosing clips:

- `prompt`: what the user is looking for. Let it steer which candidates you
  pick and how you trim. It outranks the default ranking order, never the
  trimming rules or the 10s minimum.
- `vertical`: set `"vertical"` on every clip to this value.
- `profile_chosen`: the profile the run was scored with. Copy it into
  `plan.json` as `"profile"`. If `profile_fallback` is true, Laya could not
  tell the content type; tell the user which profile was used.

## What each signal means

Every signal and composite is on a 0..1 scale.

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

Laya flagged these as low-confidence. Read their transcript slices and decide
yourself. Do not skip the bucket; that is where the model deferred to you.
Their composites may sit below the candidate threshold: when Laya is unsure,
its low score is no more trustworthy than a high one would be.

## Output

Write `runs/<name>/plan.json`:

```json
{
  "source": "runs/<name>",
  "profile": "podcast",
  "laya_model": { "...": "copy the whole object from candidates.json" },
  "speakers": { "SPEAKER_00": "Ana", "SPEAKER_01": "Marco" },
  "clips": [
    { "in": 874.10, "out": 897.40, "title": "...", "hook": "...",
      "description": "...", "clip_format": "hot_take", "vertical": true,
      "crop_x": "center", "captions": "burn", "speaker_label": true,
      "laya": { "clipworthy": 0.72, "hook_strength": 0.61, "confidence": 0.18 } }
  ]
}
```

Carry `laya_model` across verbatim. Then run `clipper plan <run> --check`.
Fix every `error:` line; render refuses the plan until they are gone. Read
each `warning:` line and either fix it or keep the clip on purpose.

If the user has their own editing skills loaded, theirs override these defaults.
