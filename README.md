# Laya Claude Clipper

Turns a long video into short, upload-ready clips.

1. **Video input**: upload a podcast, stream or tutorial.
2. **Whisper** transcribes it with word-level timestamps.
3. **The AI reads the whole transcript once** (one call per video).
4. It decides **what kind of video** it is, proposes **candidate clips** with
   start and end times, and writes the **typed questions** Laya should answer
   about each clip.
5. **Laya** (a local typed-decision model) answers those questions for every
   candidate, with a score, a choice or a yes/no plus a confidence. It is fast
   and free, so it runs on every candidate.
6. **Preview**: the best clips are ticked; you untick, tick and play them
   before any editing time is spent.
7. **Your style**: written once (layout, captions, encoder, and free-text
   notes on tone, pacing and brand rules) and reused for every clip.
8. **AI fill-in** (optional, per clip): the AI reads only that clip's words
   and writes its title, hook, quote and punch-in moments.
9. **A final edit prompt per clip**: `edit.json` plus `prompt.md`.
10. **Clips edited**: each clip is cut from the original and edited
    (`final.mp4`); an AI editor can apply `prompt.md` on top.

Design: `docs/superpowers/specs/2026-09-23-ai-first-pipeline-design.md`

## Setup

    pip install -e ".[dev]"
    cp .env.example .env

ffmpeg must be on PATH and must include libass (`ffmpeg -filters | grep subtitles`).

### The AI (steps 3-4 and 8)

Claude is the default: put `ANTHROPIC_API_KEY=...` in `.env` and it uses
`claude-opus-5`. Any OpenAI-compatible server works too:

| Provider | `.env` |
|---|---|
| Claude | `ANTHROPIC_API_KEY=...` (optional `AI_MODEL=`) |
| DeepSeek, Kimi, OpenAI, OpenRouter | `AI_PROVIDER=deepseek`, `AI_MODEL=deepseek-chat`, `AI_API_KEY=...` |
| Ollama (local, free) | `AI_PROVIDER=ollama`, `AI_MODEL=qwen3` |
| Anything else | `AI_PROVIDER=custom`, `AI_BASE_URL=http://host/v1`, `AI_MODEL=...` |

Without an AI the app still runs: the transcript is cut into 20-45 s
sentence-aligned chunks and Laya asks its four built-in questions. If the AI
call fails, the run falls back the same way and says why.

### Hardware

`clipper hardware` shows what this machine has. The web page lists the same
choices and greys out what is missing.

| Machine | Whisper and Laya | Video encoding | torch to install |
|---|---|---|---|
| NVIDIA GPU | `cuda` (faster-whisper, float16) | NVENC | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| AMD GPU on Linux | `cuda` via ROCm (transformers Whisper) | AMF / software | `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2` |
| AMD GPU on Windows | `cpu` (no ROCm torch on Windows) | AMF | the default CPU torch |
| Intel Arc | `xpu` (transformers Whisper) | Quick Sync | `pip install torch --index-url https://download.pytorch.org/whl/xpu` |
| Apple silicon | `mps` (transformers Whisper) | VideoToolbox | the default torch |
| Anything | `cpu` (faster-whisper, int8) | x264 | the default CPU torch |

`auto` picks the first GPU it finds (cuda, xpu, mps), else the CPU. Measured on
an Intel Arc 140T: large-v3 transcribes 4 minutes of speech in about 18 s
(214 s on the CPU); Laya rates a clip in about 3 s. The first run on
xpu/mps/ROCm downloads `openai/whisper-<model>` (about 3 GB for large-v3).

## Use

    clipper web

Drop a video, choose the hardware and what you are looking for, press
**Analyze**. Check the preview, set your style, press **Make clips**. Each
clip lands in `runs/<name>/clips/NN-title/`:

| File | What it is |
|---|---|
| `final.mp4` | the edited clip: layout, captions, encoder |
| `cut.mp4` | the untouched cut from the original |
| `prompt.md` | the edit instruction for an AI editor (your style notes, title, hook, punch-ins, words) |
| `edit.json` | the same, structured |
| `captions.srt` | the captions |

**Layouts.** `fit` (default) keeps the whole picture, centred over a blurred
copy of itself, so on-screen text and motion graphics are never cut off.
`crop` fills the vertical frame and cuts the sides (best for one face in the
middle).

From the command line:

    clipper analyze "episode47.mp4" --prompt "useful tips" --top 5
    clipper make runs/2026-09-23-episode47 --only c1,c3 --fill-in

Gameplay without commentary works too: the `action` step finds stretches of
8 s or more where nobody talks, ranked by loudness, motion and scene cuts,
and they appear in the preview as `a0`, `a1`, ...

### Editing further with Claude Code

`.claude/skills/clipper/SKILL.md` makes Claude Code the AI editor: point it at
a clip folder and it applies `prompt.md` (punch-ins, pacing, the style notes)
to `cut.mp4` and writes `edited.mp4`.

## Tests

    pytest              # offline; no model download
    pytest -m model     # loads the real Laya checkpoint
