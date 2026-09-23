# AI-first clipper pipeline — design

Replaces the Laya-scans-every-window pipeline with the user's 10-step flow
(diagram, 2026-09-23). The LLM reads the whole transcript once and proposes
clips plus the questions Laya should answer; Laya scores only those
candidates; a human previews the selection; style and optional per-clip AI
details turn each clip into an edit prompt; the default editor (and any AI
editing skill) edits the clip from that prompt.

## The ten steps and their artifacts (per run folder)

| # | Step | Code | Artifact |
|---|------|------|----------|
| 1 | Video input | `ingest.py` (unchanged) | `video.*`, `source.json` (keeps the original path) |
| 2 | Whisper transcription | `transcribe.py` (unchanged format) | `transcript.json` (word timestamps) |
| 3-4 | Transcript to AI; AI sets content type, clip boundaries, Laya questions | `segment.py` + `ai.py` | `segments.json` |
| 5 | Laya scores candidate clips | `rate.py` | `scored.json` |
| — | Moments without speech (kept from before) | `stages/action.py` | `action.json` |
| 6 | Selected clips preview | `select.py`, web preview table | `selection.json` |
| 7 | User style prompt (written once, reused) | `style.py` | `style.json` (project level), snapshot in the run |
| 8 | AI fill-in (optional toggle, per clip) | `fillin.py` | inside each clip's `edit.json` |
| 9 | Final edit prompt per clip | `prompts.py` | `clips/NN-slug/prompt.md` + `edit.json` |
| 10 | Clips edited | `edit.py` | `clips/NN-slug/cut.mp4` (raw cut), `final.mp4` (default editor) |

Phase A (steps 1-6) is "Analyze". Phase B (7-10) is "Make clips" and runs
only on the clips the user keeps in the preview.

## Step 3-4: one LLM call per video

- Input: every transcript segment as `[start-end] text`, the user's
  "what are you looking for" prompt, and the rules below.
- Output (strict JSON, schema-constrained on Claude): `content_type`
  (podcast, interview, talking_head, tutorial, lecture, comedy, stream,
  gaming, vlog, news, other), `summary`, `questions` (up to 6, typed:
  `score` with 3-5 levels, `choice` with 2-6 options, `noul` yes/no),
  `weights` (per score/noul question) and `candidates` (start, end,
  category, reason, hook_line).
- Rules given to the model: clips 15-60 s, at most ~150 words so the clip
  plus a question fits Laya's 512-token window (a question and its options
  use up to 192 tokens, leaving ~320 for the clip); boundaries on sentence
  ends; no overlaps; 5-25 candidates depending on length.
- After the call: validate questions (same rules the profile loader used),
  snap every boundary to real word times, drop clips under 10 s, clamp to
  the source, and estimate tokens (words x 1.35); a clip over the budget is
  kept but marked `truncated: true`.
- Without a configured AI, candidates are sentence-aligned 20-45 s chunks
  and only the built-in questions are used (`ai: null`); the UI says so.

## Step 5: Laya

Built-in questions always asked: `clipworthy`, `hook_strength`,
`self_contained`, `ends_cleanly` (wording from the old core profile), plus
the AI's questions. Score = weighted mean of normalized answers
(built-ins 0.40 / 0.25 / 0.10 / 0.05, AI questions share 0.20 by their
weights; weights renormalize when AI questions are absent). Confidence =
mean confidence of the weighted answers. `uncertain` uses the per-type
floors from before (score 0.10, choice 0.15, noul 0.55). The state Laya
reads: content type, then the clip text, then its duration.

## Step 6: selection

Keep the top N (default 5) with score >= min score (default 0.30); action
moments are appended (best 2). Every row: id, start, end, duration,
category, score, confidence, uncertain, title hint, source (`ai`/`chunks`/
`action`), `include`, `fill_in`. The preview lets the user untick rows,
toggle fill-in per row, and play the range from the original video.

## Step 7: style

`style.json` beside `runs/` (override with `CLIPPER_STYLE`): structured
fields the default editor applies (`layout` auto|crop|fit, `vertical`,
`captions` burn|sidecar|none, `caption_case` sentence|upper,
`encoder` auto|libx264|nvenc|amf|qsv|videotoolbox) plus free-text `notes`
(tone, pacing, brand rules) that only the AI editor reads. Never rewritten
by AI. Copied into the run when Make clips starts.

## Step 8: AI fill-in

Per clip, only that clip's transcript slice (clip-local timestamps) plus
the style notes. Returns `title`, `hook`, `description`, `caption_quote`,
`punch_ins` (clip-local seconds + reason). Off by default; per-row toggle.
When off, titles come from the AI's `hook_line`/category.

## Step 9-10: prompt and edit

`edit.json` = clip range, category, scores, style fields, fill-in fields.
`prompt.md` = the style notes verbatim + the clip's metadata + fill-in
details + the transcript slice, in one instruction an editing skill can act
on. `edit.py` cuts `cut.mp4` from the original and renders `final.mp4`
with layout + captions + encoder. The Claude Code skill
`.claude/skills/clipper/SKILL.md` becomes the AI editor: it reads each
clip folder's `prompt.md` and applies what the default editor could not
(punch-ins, pacing); an MCP video server can be added later.

## Layout (fix for wide motion graphics)

- `crop`: centre 9:16 crop (old behaviour).
- `fit`: whole 16:9 frame scaled to 1080 wide in the middle of a blurred,
  zoomed copy of itself; captions below the picture.
- `auto` (default): sample 6 frames of the clip; if on-screen graphics or
  text reach outside the 9:16 crop area, use `fit`, else `crop`.

## Captions

Cues break at sentence ends first, then clause punctuation, then the
42-character/3-second limits; a number is never separated from the word
after it. Optional UPPERCASE style.

## Hardware

- Devices: `auto`, `cuda` (NVIDIA; AMD on Linux via ROCm, where torch also
  reports `cuda`), `xpu` (Intel Arc), `mps` (Apple), `cpu`. The UI lists
  each with what was detected (GPU name) and disables missing ones with an
  install hint.
- Whisper: faster-whisper on NVIDIA CUDA (float16) and CPU (int8);
  transformers Whisper on ROCm, XPU and MPS.
- Laya: same device, passed explicitly.
- Video encoding: probe ffmpeg's `h264_nvenc`, `h264_amf`, `h264_qsv`,
  `h264_videotoolbox` with a one-frame test encode; `auto` picks the first
  that works, else `libx264`.
- AMD on Windows: no ROCm torch, so models run on CPU; encoding still uses
  AMF. The README has a per-vendor torch install table.

## Removed

Sliding windows, profile YAMLs and the Laya profile vote, window merging
and ranking, and the plan/render pair (replaced by selection + edit). The
Laya loader, action stage, transcription, ingest and caption code stay.
