# AI-first Clipper Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild clipper around the user's 10-step flow — one LLM call proposes clips and Laya questions, Laya scores them, the user previews, style + optional AI fill-in produce a per-clip edit prompt, and the default editor renders each clip — with a fit layout for wide graphics and hardware choices for NVIDIA, AMD, Intel, Apple and CPU.

**Architecture:** Each step is a small module that reads and writes one JSON artifact in the run folder, so every step is testable alone and re-runnable. The web job runner runs two pipelines: Analyze (steps 1-6) and Make clips (steps 7-10). Heavy libraries (torch, whisperx, laya, anthropic) are imported inside functions so the test venv needs none of them.

**Tech Stack:** Python 3.12, ffmpeg, whisperx / transformers Whisper, Laya (`laya` package), `anthropic` SDK (Claude), httpx (OpenAI-compatible providers), stdlib `http.server` web UI, pytest.

**Spec:** `docs/superpowers/specs/2026-09-23-ai-first-pipeline-design.md`

## Global Constraints

- Default Claude model: `claude-opus-5`; requests opt into `fallbacks: "default"` with beta `server-side-fallback-2026-07-01`, check `stop_reason == "refusal"` before reading content.
- Claude structured output via `output_config={"format": {"type": "json_schema", "schema": ...}}`, streaming with `get_final_message()`, `max_tokens` 64000 for segmentation, 16000 for fill-in.
- Laya context: 512 tokens; question + options take up to 192; clip text budget 300 estimated tokens (words x 1.35).
- Clip length: 10 s hard minimum; AI asked for 15-60 s.
- Tests run in `.venv` (no torch/numpy/anthropic); real runs use `.venv-gpu`.
- Never read the whole transcript for a clip-level call: fill-in gets only that clip's slice.
- User-facing text in plain language; no jargon in UI labels.

## Review Focus

- LLM returns boundaries that are not on word times, overlap, run past the end, or are under 10 s → snapped, clamped, dropped; test in Task 4.
- LLM returns malformed questions (wrong type, 1 score level, unknown weight id) → invalid questions dropped, run continues with built-ins; test in Task 4.
- No AI configured, or the AI call fails → fallback chunks, `ai: null`, run completes; test in Task 4 and Task 9.
- A clip with no speech (action moment) → no captions, no Laya, still rendered; test in Task 8.
- A requested device or encoder is missing on this machine → falls back with a clear message, never crashes; test in Task 1.

---

### Task 1: Hardware detection, device routing and hardware encoders

**Files:**
- Create: `clipper/hardware.py`, `tests/test_hardware.py`
- Modify: `clipper/transcribe.py` (ROCm routes to transformers), `clipper/device.py` (labels)

**Interfaces:**
- Produces: `hardware.detect() -> dict` = `{"devices": [{"id", "label", "available", "detail", "hint"}], "encoders": [{"id", "label", "available"}]}`; `hardware.is_rocm() -> bool`; `hardware.probe_encoders(ffmpeg: Path) -> list[str]` (working ffmpeg encoder names); `hardware.pick_encoder(requested: str, available: list[str]) -> str` (ffmpeg encoder name); `hardware.encoder_args(name: str) -> list[str]` (quality args per encoder).
- `transcribe._uses_faster_whisper(device: str, rocm: bool) -> bool`.

- [ ] **Step 1: Failing tests** — `pick_encoder("auto", ["h264_qsv"]) == "h264_qsv"`; `pick_encoder("nvenc", [])` falls back to `"libx264"` with a `RuntimeWarning`; `encoder_args("h264_nvenc")` contains `-cq`; `encoder_args("libx264")` contains `-crf`; `_uses_faster_whisper("cuda", rocm=True) is False`, `("cuda", False) is True`, `("cpu", False) is True`, `("xpu", False) is False`; `detect()` with injected probes lists five devices (auto, cuda, xpu, mps, cpu) and marks unavailable ones with a hint.
- [ ] **Step 2: Run** `pytest tests/test_hardware.py -q` → fails (module missing).
- [ ] **Step 3: Implement.** Encoder priority for `auto`: nvenc, amf, qsv, videotoolbox, libx264. Probe = `ffmpeg -hide_banner -f lavfi -i color=c=black:s=256x256:d=0.1 -frames:v 1 -c:v <enc> -f null -`, return code 0 means it works; cache per process. Encoder args: libx264 `-preset medium -crf 20`; nvenc `-preset p5 -rc vbr -cq 21 -b:v 0`; amf `-quality quality -rc cqp -qp_i 20 -qp_p 22`; qsv `-preset medium -global_quality 21`; videotoolbox `-q:v 60`. Device labels: cuda → "NVIDIA GPU (CUDA)" or "AMD GPU (ROCm)" when `torch.version.hip`; GPU name from `torch.cuda.get_device_name(0)` / `torch.xpu.get_device_name(0)`. Hints: cuda "Install a CUDA build of torch (pip install torch --index-url https://download.pytorch.org/whl/cu124)", xpu ".../whl/xpu", AMD "ROCm torch is Linux-only; on Windows AMD GPUs run models on CPU and use AMF for video".
- [ ] **Step 4: Run tests** → pass; full suite passes.
- [ ] **Step 5: Commit** `feat: detect GPUs and hardware video encoders per vendor`.

### Task 2: Fit layout, wide-graphics detection, sentence-aware captions

**Files:**
- Create: `clipper/layout.py`, `tests/test_layout.py`
- Modify: `clipper/filters.py` (fit chain), `clipper/captions.py` (cue breaking, uppercase, fit margin), `tests/test_captions.py`, `tests/test_filters.py` if present

**Interfaces:**
- `filters.build_filter_chain(width, height, vertical, subtitle_path, crop_x="center", layout="crop") -> str | None` — `layout` in `crop|fit`.
- `layout.wide_content(ffmpeg: Path, video: Path, start: float, end: float) -> bool`; `layout.side_detail(gray: bytes, w: int, h: int) -> tuple[float, float]` (edge density outside / inside the 9:16 band); `layout.resolve(layout: str, width: int, height: int, detector) -> str`.
- `captions.build_cues(words, max_chars=42, max_seconds=3.0)` now breaks after `.?!` first, after `,;:` when the line is past half length, and never between a number and the next word.
- `captions.render_ass(cues, height, speakers=None, uppercase=False, layout="crop")`.

- [ ] **Step 1: Failing tests** — fit chain contains `split`, `boxblur`, `overlay=(W-w)/2:(H-h)/2` and ends with the subtitles filter; crop chain unchanged; `resolve("auto", 1920, 1080, lambda: True) == "fit"`, `lambda: False` → `"crop"`, `resolve("auto", 1080, 1920, ...) == "crop"` (already vertical, detector not called); `side_detail` on a synthetic frame with high-contrast stripes only at the edges returns side > centre; cues from "To complete it you need 97.2 hours. Obviously boosts help." break after "hours." and never end a cue on "97.2"; `render_ass(..., uppercase=True)` text is uppercase; fit layout puts captions lower (MarginV smaller than crop's).
- [ ] **Step 2: Run** → fail.
- [ ] **Step 3: Implement.** Fit chain: `split=2[bg][fg];[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:2[bgb];[fg]scale=1080:-2[fgs];[bgb][fgs]overlay=(W-w)/2:(H-h)/2[v]` then `;[v]subtitles=...` when captions. Detection: extract 6 frames at `sheet_times(start, end)` as `gray` rawvideo 192x108 (`-vf scale=192:108,format=gray -f rawvideo`), horizontal-gradient edge mask (|p[x+1]-p[x]| > 40), density outside the centre band `[w/2 - w*81/192/2*... ]` i.e. band width = h*9/16, vs inside; frame counts as wide when side density >= 0.04 and side >= 0.5 x centre; clip is wide when >= 2 of 6 frames are. Calibrate thresholds on the Valorant run (graphic at ~191 s vs face at ~10 s) and record the numbers in the test.
- [ ] **Step 4: Run tests** → pass.
- [ ] **Step 5: Commit** `feat: fit layout keeps wide graphics; captions break at sentences`.

### Task 3: AI client (Claude SDK + OpenAI-compatible)

**Files:**
- Create: `clipper/ai.py`, `tests/test_ai.py`
- Delete: `clipper/llm.py`, `tests/test_llm.py` (if present) — its presets move into `ai.py`
- Modify: `pyproject.toml` (add `anthropic`), `.env.example`

**Interfaces:**
- `ai.AIConfig(provider: str, model: str, api_key: str | None, base_url: str | None)`.
- `ai.config_from_env() -> AIConfig | None` — `AI_PROVIDER` (anthropic|openai|deepseek|kimi|openrouter|ollama|custom), `AI_MODEL` (default `claude-opus-5` for anthropic), `ANTHROPIC_API_KEY` / `AI_API_KEY`, `AI_BASE_URL`. Returns None when nothing is configured.
- `ai.available_providers() -> list[dict]` for the UI (id, label, configured).
- `ai.complete_json(config, system: str, user: str, schema: dict, max_tokens: int) -> dict` — Claude: `client.beta.messages.stream(model, max_tokens, system, messages, output_config={"format": {"type": "json_schema", "schema": schema}}, betas=["server-side-fallback-2026-07-01"], fallbacks="default")` → `get_final_message()`; raise `AIError` on `stop_reason in ("refusal", "max_tokens")`; OpenAI-compatible: POST `/chat/completions` with `response_format={"type": "json_object"}` and the schema pasted into the system prompt. Both parse with `parse_json` (strips ``` fences).
- `ai.AIError(RuntimeError)`.

- [ ] **Step 1: Failing tests** — `config_from_env` with no vars → None; `AI_PROVIDER=deepseek` resolves base URL; anthropic without key → `ValueError` naming `ANTHROPIC_API_KEY`; `complete_json` with an injected fake Anthropic client returns the parsed dict and sends `output_config.format.schema`; a refusal raises `AIError`; OpenAI-compatible path with a fake `httpx.post` returns parsed JSON; `parse_json` strips fences.
- [ ] **Step 2-4:** run → fail → implement (client injectable via `_anthropic_client` module function) → pass.
- [ ] **Step 5: Commit** `feat: one AI client for Claude and OpenAI-compatible providers`.

### Task 4: Step 3-4 — segmentation into candidates + Laya questions

**Files:**
- Create: `clipper/questions.py`, `clipper/segment.py`, `tests/test_questions.py`, `tests/test_segment.py`

**Interfaces:**
- `questions.BUILTIN: dict` (clipworthy, hook_strength, self_contained, ends_cleanly — wording from `profiles/core.yaml`); `questions.BUILTIN_WEIGHTS = {"clipworthy": .40, "hook_strength": .25, "self_contained": .10, "ends_cleanly": .05}`; `questions.AI_SHARE = 0.20`.
- `questions.from_ai(items: list[dict]) -> tuple[dict, list[str]]` — AI question list → Laya question dict + list of problems (invalid ones dropped).
- `questions.combined_weights(ai_weights: dict[str, float], ai_questions: dict) -> dict[str, float]`.
- `segment.transcript_lines(transcript) -> str` (`[12.3-18.9] text` per segment, speech only).
- `segment.SCHEMA: dict` (JSON schema, `additionalProperties: False` everywhere).
- `segment.build_prompt(transcript, duration, user_prompt) -> tuple[str, str]`.
- `segment.snap(candidates, words, duration) -> list[dict]`.
- `segment.fallback_candidates(transcript, duration) -> list[dict]`.
- `segment.run_segment(run: Run, user_prompt: str = "", config=..., complete=ai.complete_json) -> dict` → writes `segments.json`: `{"content_type", "summary", "ai": {"provider","model"} | None, "ai_error": str | None, "questions": {...}, "weights": {...}, "candidates": [{"id","start","end","duration","category","reason","hook_line","text","est_tokens","truncated"}], "problems": [...]}`.

- [ ] **Step 1: Failing tests** — `from_ai` accepts a 4-level score, a 3-option choice, a noul, and drops a 1-level score and an unknown type with problems listed; ids are slugged and never collide with built-ins; `combined_weights` sums to 1 and ignores choice questions; `snap` moves 12.31→first word start >=, 40.02→last word end <=, clamps past-end, drops < 10 s, removes overlaps (keeps earlier), sorts by start; `est_tokens` > 300 sets `truncated`; `fallback_candidates` makes 20-45 s chunks ending on sentence ends; `run_segment` with a fake `complete` writes the file with AI fields; with `complete` raising `AIError` writes fallback + `ai_error`; with no config writes `ai: None`.
- [ ] **Step 2-4:** run → fail → implement → pass.
- [ ] **Step 5: Commit** `feat: AI reads the transcript once and proposes clips and Laya questions`.

### Task 5: Step 5 — Laya scores the candidates

**Files:**
- Create: `clipper/rate.py`, `tests/test_rate.py`
- Modify: `clipper/agent.py` (`load_agent(language, questions: dict, device)` — calibration check takes questions, not a Profile)

**Interfaces:**
- `rate.clip_state(candidate, content_type) -> dict` (`{"content_type", "clip": {"text"}, "duration"}`, clip text first).
- `rate.normalize(answer, qdef) -> float | None` (score / (k-1), noul prob).
- `rate.rate_one(candidate, questions, weights, agent) -> dict` → `{"id", "answers", "score", "confidence", "uncertain", "failed", "error"}`.
- `rate.run_rate(run, device=None, agent=None, progress=None) -> dict` → writes `scored.json` `{"laya_model", "candidates": [candidate + rating]}`; returns `{"scored", "failed", "device"}`.

- [ ] **Step 1: Failing tests** — with a fake agent returning fixed answers, score equals the hand-computed weighted mean; confidence is the mean over weighted questions; a question below its type floor sets `uncertain`; an agent exception marks only that candidate `failed`; `run_rate` with zero candidates writes an empty file and returns device "none"; progress is called per candidate.
- [ ] **Step 2-4:** run → fail → implement → pass.
- [ ] **Step 5: Commit** `feat: Laya answers the AI's questions for each candidate`.

### Task 6: Step 6 — selection

**Files:**
- Create: `clipper/select.py`, `tests/test_select.py`

**Interfaces:**
- `select.select(scored: list[dict], action: list[dict], top_n=5, min_score=0.30, action_n=2) -> list[dict]` rows: `{"id", "start", "end", "duration", "category", "score", "confidence", "uncertain", "title_hint", "source", "include", "fill_in", "reason"}`; ids `c<n>` for Laya candidates, `a<n>` for action.
- `select.run_select(run, top_n=5, min_score=0.30) -> dict` writes `selection.json` `{"rule": {...}, "clips": [...]}`.
- `select.apply_choices(run, choices: list[dict]) -> dict` — updates `include`/`fill_in` per id, rejects unknown ids.

- [ ] **Step 1: Failing tests** — top-N by score, ties by confidence; below min score not included; failed candidates never selected; action rows appended with `source="action"`, `category="action"`; `apply_choices` flips flags and raises `ValueError` for an unknown id.
- [ ] **Step 2-4:** run → fail → implement → pass.
- [ ] **Step 5: Commit** `feat: selection keeps the top clips for preview`.

### Task 7: Steps 7-9 — style, AI fill-in, final edit prompt

**Files:**
- Create: `clipper/style.py`, `clipper/fillin.py`, `clipper/prompts.py`, tests for each

**Interfaces:**
- `style.DEFAULT: dict` = `{"layout": "auto", "vertical": True, "captions": "burn", "caption_case": "sentence", "encoder": "auto", "notes": ""}`; `style.path() -> Path` (`CLIPPER_STYLE` or `./style.json`); `style.load() -> dict`; `style.save(data) -> dict` (validates enums, notes <= 4000 chars).
- `fillin.SCHEMA`; `fillin.clip_slice(transcript, start, end) -> str` (clip-local `[s-e] text`); `fillin.fill_in(config, clip, transcript, notes, complete=ai.complete_json) -> dict` → `{"title", "hook", "description", "caption_quote", "punch_ins": [{"at", "reason"}]}` with `at` clamped into the clip.
- `prompts.edit_spec(clip, style, fill, content_type, stem) -> dict`; `prompts.render_prompt(spec, style_notes, transcript_slice) -> str` (markdown).

- [ ] **Step 1: Failing tests** — `style.save` rejects `layout="zoom"`; round-trips; `clip_slice` gives clip-local times and only words inside; `fill_in` with a fake `complete` clamps punch-ins and is never given text outside the clip (assert on the user message); `render_prompt` contains the notes verbatim, the range, the title and the transcript slice; with fill-in off the title comes from `title_hint`.
- [ ] **Step 2-4:** run → fail → implement → pass.
- [ ] **Step 5: Commit** `feat: style prompt, optional AI fill-in and a final edit prompt per clip`.

### Task 8: Step 10 — cut and edit each clip

**Files:**
- Create: `clipper/edit.py`, `tests/test_edit.py`
- Delete: `clipper/render.py`, `clipper/plan.py`, `tests/test_render.py`, `tests/test_plan.py` (logic moves into `edit.py`: `slugify`, `staged_subtitles`, `display_size`, min-length check)

**Interfaces:**
- `edit.cut_command(ffmpeg, source, start, end, output, encoder) -> list[str]` (no filters).
- `edit.final_command(ffmpeg, source, start, end, output, chain, encoder) -> list[str]`.
- `edit.run_edit(run, clips: list[dict], fills: dict[str, dict], style: dict, config=None, progress=None) -> list[dict]` — for each included clip: folder `clips/NN-slug/`, write `edit.json`, `prompt.md`, `captions.srt`, render `cut.mp4` then `final.mp4`; returns `[{"id", "folder", "final", "prompt", "title", "duration", "layout"}]`.

- [ ] **Step 1: Failing tests** (subprocess monkeypatched) — commands use the chosen encoder's args; an action clip with no words gets no subtitles filter; layout `auto` calls the detector once per clip; a failed ffmpeg removes the partial file and raises with the ffmpeg tail; clips under 10 s raise before rendering; folder names are `01-<slug>`.
- [ ] **Step 2-4:** run → fail → implement → pass.
- [ ] **Step 5: Commit** `feat: every kept clip is cut, edited and packaged with its prompt`.

### Task 9: Pipelines, job runner, CLI; remove the old pipeline

**Files:**
- Create: `clipper/pipeline.py`, `tests/test_pipeline.py`
- Modify: `clipper/web/jobs.py` (job = named stage list), `clipper/cli.py`, `tests/test_jobs.py`, `tests/test_cli.py`
- Delete: `clipper/window.py`, `clipper/classify.py`, `clipper/merge.py`, `clipper/rank.py`, `clipper/score.py`, `clipper/stages/score.py`, `clipper/profiles/` (keep nothing), and their tests

**Interfaces:**
- `pipeline.ANALYZE = ("ingest", "transcribe", "segment", "rate", "action", "select")`, `pipeline.MAKE = ("style", "fillin", "edit")`.
- `pipeline.Steps` dataclass of injectable callables (ingest, transcribe, segment, rate, action, select, fillin, edit).
- `pipeline.analyze_steps(settings) -> list[tuple[str, Callable]]`, `pipeline.make_steps(settings) -> list[...]`.
- `JobRunner.start(run, kind: "analyze" | "make", settings, video=None)`; status gains `kind`, `stages` keyed by that kind's names, `result`.
- CLI: `clipper analyze VIDEO [--run] [--device] [--model] [--prompt] [--top N]`, `clipper make RUN [--fill-in] [--only c1,c3]`, `clipper hardware`, `clipper web`; `ingest`/`transcribe`/`action` kept.

- [ ] **Step 1: Failing tests** — analyze runs the six stages in order and writes `settings.json`; a failed stage stops the job with `failed_stage`; make refuses a run without `selection.json`; make with no included clips fails with a clear message; CLI `analyze` calls stages with device/model/prompt; `hardware` prints the device table.
- [ ] **Step 2-4:** run → fail → implement → pass; delete old modules/tests; full suite green.
- [ ] **Step 5: Commit** `refactor: two pipelines (analyze, make clips) replace the window pipeline`.

### Task 10: Web UI

**Files:**
- Modify: `clipper/web/server.py`, `clipper/web/index.html`, `tests/test_web.py`

**Interfaces (HTTP):**
- `GET /api/config` → `{"hardware": detect(), "models", "ai": {"providers", "active"}, "hf_token", "style": style.load()}`.
- `PUT /api/upload?name=` (unchanged), `POST /api/analyze` (settings), `GET /api/status`.
- `GET /api/selection?run=` → selection + segments summary (content type, AI questions).
- `PUT /api/style` → saved style.
- `POST /api/make` → `{"run", "choices": [{"id", "include", "fill_in"}]}`.
- `GET /media/<run>/<path>` → video/prompt files inside the run folder only, with HTTP Range support for `<video>` seeking; path traversal refused.

- [ ] **Step 1: Failing tests** — config lists five devices; analyze validates device/encoder/model; selection 404s for an unknown run; make with an unknown clip id → 400; media refuses `..` and serves a byte range with 206.
- [ ] **Step 2-4:** run → fail → implement → pass.
- [ ] **Step 5: UI.** Sections: 1 Video + settings (Hardware select showing detected GPU names, Whisper model, AI provider line, "What are you looking for", clips to keep); 2 Progress; 3 Preview table (checkbox, time, category, score, confidence, title hint, AI fill-in toggle, ▶ plays the range from the original); 4 Style (layout, vertical, captions, caption case, encoder, notes — Save); 5 Make clips button; 6 Results (video player per clip, links to prompt.md and edit.json).
- [ ] **Step 6: Commit** `feat: web UI for the two-phase flow, preview and style`.

### Task 11: Docs, skill, real end-to-end run

**Files:**
- Modify: `README.md`, `.env.example`, `.claude/skills/clipper/SKILL.md` (now the AI editor for clip folders)

- [ ] **Step 1:** README: the 10-step flow, AI setup (Claude default `claude-opus-5`, OpenAI-compatible alternatives, no-AI fallback), hardware table (NVIDIA / AMD Linux ROCm / AMD Windows / Intel Arc / Apple / CPU with torch install commands), encoder notes.
- [ ] **Step 2:** SKILL.md: input is a clip folder (`cut.mp4`, `final.mp4`, `edit.json`, `prompt.md`); apply the prompt with ffmpeg (punch-ins, pacing) writing `edited.mp4`; later an MCP video server can replace ffmpeg.
- [ ] **Step 3:** Real run on the Valorant video in `.venv-gpu` through the web UI flow (analyze → preview → make) and inspect a rendered frame of a fit-layout clip.
- [ ] **Step 4: Commit** `docs: AI-first flow, hardware setup, editor skill`.
