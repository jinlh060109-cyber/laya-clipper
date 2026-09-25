# Changelog

## 2026-09-24

### Added
- **AI director with Remotion motion graphics** (optional step 11,
  `clipper/director.py`, `remotion/`). After the edit, the chosen AI works
  on each clip as a tool-calling agent: it reads only that clip's script,
  looks at frames with its vision model (`look`), plans graphics
  (`add_graphics`, `remove_graphic`), checks them drawn on real frames
  (`preview`) and renders (`render`). The graphics are a fixed kit of
  Remotion components (kinetic text, counter, lower third, pointer,
  sticker, progress bar). The AI fills in props that are validated before
  anything is drawn; it never writes code. Remotion renders a transparent
  ProRes 4444 layer and ffmpeg lays it over `final.mp4`, keeping
  `final_plain.mp4`. Every tool call is logged to `director.json`. Works
  with every provider through `ToolChat`. Placement is computed per layout,
  so graphics use the empty bands of `fit`/`black`/`square` and never cover
  captions or the hook title. Turn it on with *AI director* in the style
  panel or `clipper make --director`. The checkbox is greyed out with the
  reason when Node or `npm install` is missing.
- **Boundary check** (`clipper/boundaries.py`, skill
  `clipper/skills/clip_boundaries.md`). A second, small AI call sees the
  transcript as numbered sentences, judges each clip's start and end, and
  moves it to sentences that give it both. The new range must stay 10-90 s
  and not overlap another clip; a story the proposal split into two clips
  is merged back into one. The preview tags moved clips *range fixed
  by AI*. On 18 hand-labelled clips it judged starts right 15/18 and ends
  12-14/18.
- **Laya fixed layer and variable layer.** The fixed layer (`FIXED` in
  `clipper/questions.py`) is asked of every clip: `clipworthy`,
  `hook_strength`, `has_start`, `has_end` (80% of the score). The AI's own
  questions are the variable layer (20%). `has_start`, `has_end` and
  `hook_strength` read only the clip's opening and closing sentence (a
  second Laya pass). On the whole text Laya scored at chance on them (AUC
  0.29-0.48); on the edges it reaches 0.76 (start) and 0.71 (end). Clips
  that fail are tagged *no start* / *no end* in the preview. The *How Laya
  decided* table shows each question's layer and what it read.
- `ToolChat` tool results can carry images, for Claude and for
  OpenAI-style providers. A model that rejects images carries on without
  them.
- Unit tests for the boundary check, the director (validation, placement,
  a scripted agent loop), edge-state rating and image tool results. The
  E2E AI test now also checks the boundary check, the fixed layer and a
  real director run (`look` before `render`, `final_plain.mp4`, duration
  unchanged).

### Changed
- The built-in questions `self_contained` and `ends_cleanly` are replaced
  by `has_start` and `has_end`. Fixed-layer weights went from 0.40/0.25/0.10/0.05
  to 0.30/0.20/0.15/0.15.
- `clipper/skills/laya_questions.md` tells the AI which checks the fixed
  layer already covers.
- README: the new steps, the fixed and variable layers, the AI director,
  Remotion setup and licence.

### Removed
- The self-written MCP finishing server and host (`clipper/finish/`,
  `clipper/skills/finish_clips.md`, `mcp_servers.example.json`), its UI
  toggle, CLI flag, pipeline step, tests and the `mcp` package. Kinocut was
  already removed earlier.

### Fixed
- The web page no longer throws an unhandled "Failed to fetch" when a
  status poll fails (app restarting, network blip). It says it lost contact
  and keeps retrying.
- Earlier today: the whisperx/pyannote torchcodec warning and transformers
  warnings on transcription. Segment-stage invalid JSON (strict JSON schema
  on the Token Plan, one retry, clearer errors). ffmpeg children no longer
  inherit stdin. Laya yes/no questions are asked as neutral A/B choices,
  because `noul` follows its labels more than the clip.
