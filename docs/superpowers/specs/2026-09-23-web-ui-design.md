# Clipper Web UI — Design

**Date:** 2026-09-23
**Status:** Approved in conversation; this document records it.
**Depends on:** the Laya pipeline (`2026-09-22-laya-claude-clipper-design.md`), Tasks 1–20 complete.

## 1. Goal

A single local page where the user drops a video, sets a few options, and starts a
run. The page runs the pipeline up to and including scoring and shows progress. It
stops where the command line stops: planning is still done by Claude, and rendering
is still `clipper render`.

Out of scope: rendering from the browser, viewing clips, multiple simultaneous
jobs, remote access, authentication.

## 2. User flow

1. `clipper web` starts a server on `127.0.0.1:8765` (`--port` to override) and
   opens the default browser.
2. The page shows:
   - a drop zone (also clickable to pick a file);
   - **Video type**: Auto (default), Podcast, Talking head, Lecture,
     Stream (gaming, just chatting);
   - **Device**: auto (default), xpu, cpu;
   - **Whisper model**: large-v3 (default), medium, small;
   - **Speaker diarization** toggle, default on. If `HF_TOKEN` is not set, the
     page shows an inline warning and the toggle defaults to off;
   - **Vertical 9:16** toggle, default on;
   - **Prompt** textarea, optional ("what are you looking for?");
   - **Start**, disabled until a file is chosen.
3. On Start the page uploads the file, then shows a stage list: upload, ingest,
   transcribe, windows, profile, score. Each stage is pending, running, done or
   failed. Upload shows a percentage.
4. On success it shows the chosen profile (and, for Auto, the vote counts), the
   number of candidates and uncertain windows, the run folder, and the next step:
   "Ask Claude to plan this run."
5. On failure it shows the failing stage and the error message, the same text
   the CLI prints.

While a job runs, Start is disabled. A second browser tab sees the same job.

## 3. Architecture

```
clipper/web/
  server.py    HTTP routes on http.server.ThreadingHTTPServer; serves index.html
  jobs.py      Job state and the background stage runner
  index.html   The page: plain HTML, CSS and JS, no build step
clipper/classify.py   Auto profile selection with Laya
clipper/cli.py        New `web` subcommand
```

Standard library only. No new dependencies.

### 3.1 Routes

| Method | Path | Behaviour |
|---|---|---|
| GET | `/` | `index.html` |
| GET | `/api/config` | `{"profiles": [...], "devices": [...], "models": [...], "hf_token": bool}` |
| PUT | `/api/upload?name=<filename>` | Streams the raw request body to disk in 1 MiB chunks. Creates the run directory. Returns `{"run": "<run dir name>"}` |
| POST | `/api/start` | JSON body: `{"run", "profile", "device", "model", "diarize", "vertical", "prompt"}`. Starts the job. `409` if a job is already running |
| GET | `/api/status` | Current job snapshot, or `{"state": "idle"}` |

Upload uses a raw `PUT` body rather than multipart so the server needs no form
parser and never holds the file in memory. The filename is taken from the query
string, reduced to its basename, and slugified for the run name with
`default_run_name`. The file keeps its original extension inside the run
directory as `source<ext>`.

### 3.2 Jobs

`jobs.py` holds one `Job` at a time behind a lock. A job has `run`, `settings`,
`stages` (ordered name → status), `error`, `result`, `started`, `finished`.
Every state change is written to `runs/<name>/job.json` atomically via
`Run.write_json`, so a finished or failed job can be inspected after a restart.
The in-memory job is the source of truth while the server runs.

The runner executes, in order, in one background thread:

1. **ingest**: `ingest(ffmpeg, ffprobe, video, run)` after `preflight(require_subtitles=False)`.
2. **transcribe**: `transcribe(wav, run, model=..., hf_token=HF_TOKEN if diarize else None)`.
3. **windows**: the same code path the CLI's `window` subcommand uses.
4. **profile**: if the profile is `auto`, load the agent and call `choose_profile`;
   otherwise mark the stage done immediately.
5. **score**: `run_score(run, profile, agent=(agent, meta))`, reusing the agent
   loaded in step 4 when there is one, so the checkpoint loads once.

Stage functions are injected into the runner, so tests pass fakes and no test
touches ffmpeg, WhisperX or Laya.

Before step 1 the runner writes `runs/<name>/settings.json`:

```json
{"profile": "auto", "device": "auto", "model": "large-v3", "diarize": true,
 "vertical": true, "prompt": "funny moments, skip the sponsor read"}
```

After step 4 it adds `"profile_chosen"` and, for Auto, `"profile_votes"`.

### 3.3 Auto profile selection

`clipper/classify.py`:

- `sample_windows(windows, n=12) -> list[dict]`: up to `n` windows evenly
  spaced across the source, so a cold open doesn't decide the whole video.
- `choose_profile(windows, agent, profiles) -> dict`: asks Laya one `choice`
  question per sampled window: "What kind of content is this?", with one
  criterion per selectable profile and a one-line description of each. Returns
  `{"profile": <winner>, "votes": {name: count}, "sampled": k}`.
- The state is `window_state(window, "auto")`, the same builder scoring uses.
- Ties go to the profile listed first in the fixed order podcast,
  talking_head, lecture, stream.
- A window whose call raises or returns an unknown key casts no vote. If no
  window votes, the result falls back to `podcast` with a `"fallback": true`
  flag, and the page says so.

The agent is loaded with `load_agent(language, load_profile("core"), device=...)`.
Calibration provenance is computed against `core`; the score stage then records
the same `meta` it received.

### 3.4 The prompt

The prompt is stored in `settings.json` and nowhere else. It does not reach
Laya, so scores stay comparable across runs. `.claude/skills/clipper/SKILL.md`
gains a section telling Claude to read `settings.json` before planning: honour
the prompt when selecting and trimming, and set `"vertical"` on each clip from
the setting.

## 4. Error handling

- Every stage runs inside `try`. On an exception the stage is marked failed, the
  message is stored in `error`, later stages stay pending, and the job ends. The
  message is `str(exc)`; the known error types already carry actionable text.
- An upload that fails mid-stream deletes the partial file.
- `/api/start` for a run with no uploaded video returns `400`.
- The server binds `127.0.0.1` only.

## 5. Testing

- `tests/test_classify.py`: sampling spread and bounds; majority vote; tie order;
  failed and unknown answers don't vote; empty-vote fallback. Fake agent.
- `tests/test_jobs.py`: stages run in order; failure stops later stages and
  records the error; `settings.json` and `job.json` are written; Auto reuses one
  agent for profile and score; a second start while running is refused. Fake
  stage functions.
- `tests/test_web.py`: the real server on an ephemeral port with a fake runner:
  config, streamed upload lands on disk intact, start → status, `409` on double
  start, `400` on a missing video.
- Manual: one real run through the page with a real video on the Arc GPU before
  calling it done.
