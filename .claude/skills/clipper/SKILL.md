---
name: clipper
description: Use when editing a clip that clipper produced — a folder under runs/<name>/clips/ with prompt.md, edit.json, cut.mp4 and final.mp4 — to apply its edit prompt (punch-ins, pacing, style notes) and write edited.mp4
---

# Clipper: the AI editor (step 10)

The app has already chosen the clip, cut it and made a default edit. You
apply what a fixed ffmpeg recipe could not.

## Input: one clip folder

`runs/<name>/clips/NN-title/` holds:

- `prompt.md` — the instruction. It starts with the creator's style notes
  (verbatim; they apply to every clip), then this clip's title, hook, quote,
  punch-ins and its words with clip-local timestamps.
- `edit.json` — the same, structured: `source_range`, `layout`, `vertical`,
  `captions`, `caption_case`, `encoder`, `punch_ins`, `title`, `hook`.
- `cut.mp4` — the untouched cut. Start from this.
- `final.mp4` — the default edit (layout + captions). Use it as reference,
  or as the base when the only change is small.
- `captions.srt` — captions in clip time.

Read `prompt.md` first. Do not read the whole run's transcript; the clip's
words are in the prompt.

## What to apply

1. **Style notes** win over everything below. If they ask for something you
   cannot do with the tools available, say so instead of guessing.
2. **Punch-ins**: at each `at` second, a quick zoom (about 1.15x for 1-2 s),
   e.g. with ffmpeg `zoompan` or `crop` + `scale` enabled only in that range.
3. **Hook**: if there is one and the style allows on-screen text, show it for
   the first 2 s (ffmpeg `drawtext`, or an ASS line burned with the captions).
4. **Layout and captions** from `edit.json`: `fit` keeps the whole frame over
   a blurred copy (never crop on-screen text away); `crop` fills 9:16.
   Captions from `captions.srt`, uppercase when `caption_case` is `upper`.
5. **Pacing** only when the notes ask for it (e.g. "cut every pause"): find
   gaps between words in the prompt's transcript and remove them with
   `select`/`aselect` or by concatenating segments.

Write the result to `edited.mp4` in the same folder. Never overwrite
`cut.mp4` or `final.mp4`. Re-encode with the encoder named in `edit.json`
(`auto` means: whatever `clipper hardware` shows as automatic).

## Checking your work

Grab two or three frames (`ffmpeg -ss T -i edited.mp4 -frames:v 1 f.png`)
around each punch-in and the first second, look at them, and confirm the
duration with ffprobe. Report what you applied and anything you skipped.

## Motion graphics

For animated graphics (counters, kinetic text, lower thirds), use the app's
built-in AI director instead (`clipper make ... --director`); it renders them
with Remotion and keeps the plain edit as `final_plain.mp4`.
