# Laya Clipper

**Turn a long video into short, upload-ready vertical clips. It runs on your
own machine and works with any AI provider, or with none.**

A free, self-hosted alternative to tools like Opus Clip. Whisper transcribes
the video, an LLM of your choice proposes clips, and
[Laya](https://huggingface.co/convaiinnovations/laya), a small local model,
scores every candidate. The scoring is transparent: you see every question
Laya was asked and every answer it gave. Then ffmpeg cuts, reframes and
captions the clips, and an optional AI director adds animated graphics
with [Remotion](https://www.remotion.dev).

- **Local first.** Transcription, scoring, editing and rendering run on
  your hardware. Only the LLM calls leave your machine, and a local
  [Ollama](https://ollama.com) model can do those too.
- **Any model.** Claude, Qwen (Alibaba Model Studio / Token Plan),
  DeepSeek, Kimi, OpenRouter, OpenAI, Ollama or any OpenAI-compatible
  server. No AI at all also works.
- **Explainable ranking.** Tick *Show how Laya decided* to see why each
  clip ranks where it does.
- **Finished clips.** 9:16 layouts, six animated caption looks, an on-screen
  hook title, punch-in zooms, and optional motion graphics (counters,
  kinetic text, lower thirds) placed by an AI that looks at the frames.

---

## How it works

```
video ─► Whisper ─► LLM: kind of video, candidate clips, Laya questions
                     └► LLM: does each clip have a real start and end? (fixes them)
      ─► Laya scores every candidate (fixed layer + variable layer)
      ─► preview: tick the clips you want
      ─► your style (layout, captions, notes) + optional AI fill-in per clip
      ─► ffmpeg: cut, reframe, captions, hook title, zooms  →  final.mp4
      ─► optional AI director: looks at the clip, adds Remotion graphics
```

1. **Transcribe.** Whisper (faster-whisper or transformers) gives
   word-level timestamps. Stretches with no speech go to an `action` step
   that ranks them by loudness, motion and scene cuts, so gameplay without
   commentary still gets clips.
2. **Propose.** The LLM reads the whole transcript once. It names the kind
   of video, proposes candidate clips, and writes typed questions for Laya
   that fit this video and what you asked for ("the funniest moments").
3. **Fix the boundaries.** A second, small LLM call sees the transcript as
   numbered sentences. It checks that each clip opens on its own topic and
   ends on a conclusion, and moves the clip to sentences that do. When a
   story was split in two, it merges the halves back.
4. **Score with Laya.** Laya never writes text. It picks a level, an option
   or yes/no, with a probability for each. It answers two layers:
   - the **fixed layer**, asked of every clip in every video: *worth
     watching*, *hook strength*, *has a start*, *has an end* (80% of the
     score). Start, end and hook read only the clip's first and last
     sentence, where Laya can actually tell good from bad.
   - the **variable layer**: up to six questions the LLM wrote for this
     video (20%).
5. **Preview.** The best clips are ticked. Play them, untick or tick
   others, and open the Laya decision table for any clip.
6. **Style and fill-in.** Your style (layout, caption look, notes on tone
   and brand) is written once and reused. The optional AI fill-in reads only
   one clip's words and writes its title, on-screen hook, quote and punch-in
   moments.
7. **Edit.** ffmpeg cuts each clip from the original and makes the 9:16
   layout (`fit`, `crop`, `black`, `square`, `split`). It burns in the
   animated captions and the hook title, and zooms at the punch-ins.
8. **Direct (optional).** The chosen AI works on each finished clip as a
   tool-calling agent. It reads the clip's script, looks at frames
   (`look`), places graphics from a fixed Remotion kit (`add_graphics`),
   checks them drawn on real frames (`preview`) and renders (`render`). It
   fills in validated props and never writes code, so any model can do it
   safely.

Each clip lands in `runs/<name>/clips/NN-title/` with `final.mp4`, the raw
`cut.mp4`, captions (`.srt`, `.ass`), `edit.json` and a `prompt.md` for any
other AI editor.

---

## Install

**You need:** Python 3.11+, [ffmpeg](https://ffmpeg.org) on `PATH` built
with libass (check with `ffmpeg -filters | grep subtitles`), and about 5 GB
of disk for the models. For the AI director you also need
[Node.js](https://nodejs.org) 18+.

```bash
git clone https://github.com/jinlh060109-cyber/laya-clipper.git
cd laya-clipper
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # then add a key, or pick the AI in the web page
```

**GPU (optional, much faster).** Install the torch build for your hardware
*before* the line above:

| Machine | Device | torch |
|---|---|---|
| NVIDIA | `cuda` | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| AMD on Linux | `cuda` (ROCm) | `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2` |
| Intel Arc | `xpu` | `pip install torch --index-url https://download.pytorch.org/whl/xpu` |
| Apple silicon | `mps` | the default torch |
| Anything else | `cpu` | the default torch |

`clipper hardware` shows what the app found.

**AI director (optional):**

```bash
cd remotion && npm install
```

The first render downloads Chrome Headless Shell (about 100 MB).

**Run it:**

```bash
clipper web          # opens http://127.0.0.1:8765
```

On Windows you can double-click `Start Clipper.bat` instead. Drop a video,
say what you are looking for, press **Analyze**, check the preview, set your
style, press **Make clips**.

On the command line:

```bash
clipper analyze "episode47.mp4" --prompt "useful tips" --top 5
clipper make runs/2026-09-23-episode47 --fill-in --director --caption-style bold
clipper providers    # the AI providers and which have a key
```

---

## Choosing the AI

Pick it in the web page (**AI provider** panel, with **Test connection**)
or with `--ai <id> --ai-model <model>`. Each provider keeps its own key in
`.env`. Keys are never sent back to the page.

| Provider (`--ai`) | Key in `.env` | Default model |
|---|---|---|
| `anthropic` Claude | `ANTHROPIC_API_KEY` | `claude-opus-5` |
| `alibaba_token_plan` Alibaba Cloud Token Plan | `ALIBABA_TOKEN_PLAN_API_KEY` | `qwen3.8-max` |
| `alibaba` Model Studio pay-as-you-go | `DASHSCOPE_API_KEY` | `qwen-plus` |
| `deepseek`, `kimi`, `openrouter`, `openai` | `DEEPSEEK_API_KEY`, `MOONSHOT_API_KEY`, ... | set `AI_MODEL` |
| `ollama` (local, free) | none | set `AI_MODEL`, e.g. `qwen3` |
| `custom` (any OpenAI-compatible server) | `AI_API_KEY` + `AI_BASE_URL` | set `AI_MODEL` |
| `none` | | no AI |

Without an AI the app still works. The transcript is cut into 20-45 s
sentence-aligned chunks and Laya scores them with its fixed layer. The AI
director needs a model that can call tools. With a vision model it also
sees the frames; other models plan from the script alone.

**What is free.** Whisper, Laya, ffmpeg and the app itself are free and
local. The LLM calls cost whatever your provider charges, or nothing with
Ollama. Remotion is free for individuals, non-profits and companies of up
to three people; bigger companies need a
[Remotion company license](https://www.remotion.dev/license).

---

## Features in detail

**Caption looks.** `classic`, `bold`, `boxed`, `minimal`, `one_word` and
`neon`. Each is previewed on a real frame before anything is rendered.

**Layouts.** `fit` (whole picture over a blurred copy, the default), `crop`
(fills the frame), `black` (letterbox), `square`, and `split` (left and
right halves stacked, for two-person podcasts).

**The two text boxes.** *What are you looking for?* changes which clips are
chosen: the LLM proposes clips and writes Laya questions for that goal.
*Style notes* never change the choice; they tell whoever edits a clip how
it should sound and look.

**How Laya decided.** For each clip, the table lists every question and its
layer, what Laya read (the whole clip or only its edges), the range, Laya's
probabilities, the value, the confidence, the weight, and the contribution.
The rows add up to the score. Clips Laya thinks have no start or end are
tagged in the preview. Clips the boundary check moved are tagged *range
fixed by AI*.

**Built-in AI skills.** The instructions sent to the LLM live as Markdown
in `clipper/skills/`:
- `laya_questions.md`: how to write good Laya questions, adapted from
  TypeSafe's [typesafe-ai skill](https://github.com/typesafe-ai/skills).
- `clip_boundaries.md`: how to judge a clip's start and end.
- `director.md`: how to direct motion graphics.

**Speed** (Intel Arc 140T, `qwen3.8-max`):
- Whisper large-v3 transcribes 4 minutes of speech in about 18 s.
- Laya scores a clip in about 3 s.
- The AI director takes 40-90 s per clip.

**Claude Code.** `.claude/skills/clipper/SKILL.md` lets Claude Code apply a
clip's `prompt.md` as a further edit.

---

## Tests

```bash
pytest                      # offline unit tests, no models
pytest -m model             # loads the real Laya checkpoint
pytest -m e2e tests/e2e     # the real app in Chromium: Whisper, Laya, ffmpeg, Remotion
```

The end-to-end suite needs `pip install playwright && playwright install
chromium`, the models, and `npm install` in `remotion/`. The AI test runs
when `ALIBABA_TOKEN_PLAN_API_KEY` (or `CLIPPER_E2E_AI_PROVIDER` plus that
provider's key) is set. Every run leaves its evidence in `e2e-artifacts/`.

---

## Limitations

- Scoring reads the transcript. Purely visual moments are found only by the
  simple `action` step (loudness, motion, scene cuts).
- Laya's 512-token context fits clips of about 90 s. Longer text is cut.
- The English Laya checkpoint is the one that has been tested. For other
  languages the app switches to Laya's multilingual checkpoint automatically;
  it has seen much less testing here.
- Clip boundaries are only as good as the LLM. In our tests it judged
  starts right 15 times in 18 and ends about 13 times in 18.
- Speaker diarization needs a Hugging Face token (`HF_TOKEN`).

---

## Future path

Where this is going, roughly in order. Contributions and ideas are welcome.

**Better ranking**
- **Genre-specialised Laya checkpoints.** Laya(gaming), Laya(podcast) and
  so on, fine-tuned on labelled clips of that genre. A new genre starts on
  the generic checkpoint plus the fixed layer, until it has enough labels.
- **Multimodal scoring.** Add cheap visual signals (motion, faces,
  on-screen action) to what Laya reads, so visual moments rank as well as
  spoken ones.
- **Local LLMs, measured honestly.** A `--compare` flag that runs a local
  model (Ollama) and a paid one side by side on the same video and reports
  the quality gap in clip choice and boundaries.

**Learning from real performance**
- **Retention data in.** Import YouTube Studio, TikTok and Instagram
  exports as CSV first. Direct "connect your account" OAuth (YouTube
  Analytics, Instagram Graph, TikTok Business) comes later.
- **Normalised signal.** Views depend on posting time and algorithm luck,
  so performance is normalised (per channel, per week, retention rather
  than raw views) before it becomes training data.
- **Self-improvement loop.** A periodic `retrain on the last N labelled
  clips` script, rather than live fine-tuning. Laya learns what works for
  your audience, which a closed platform cannot offer.
- **Swipe labelling.** A quick yes/no swipe UI on the preview to make
  training data fast.
- **Community checkpoint hub.** Share fine-tuned Laya checkpoints per
  niche, LoRA-hub style.

**Output and workflow**
- **Batch processing.** Queue several videos and analyze them in one go.
- **Platform profiles.** One click for TikTok, Shorts or Reels: length
  limits, safe zones and caption style per platform.
- **Cross-video duplicate detection.** Flag stories and anecdotes a creator
  has already clipped from their back catalogue.
- **Config file.** Models, thresholds, weights and prompts in one file
  instead of `.env` plus code.

**Packaging**
- A one-line install script and a Docker image (CPU and CUDA).
- A demo video and GIF in this README.
- Clearer hardware requirements per setup.

---

## Credits

- [Laya](https://huggingface.co/convaiinnovations/laya) by ConvAI
  Innovations: typed-decision model (Apache-2.0 weights).
- [WhisperX](https://github.com/m-bain/whisperX) /
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper):
  transcription.
- [Remotion](https://www.remotion.dev): motion graphics
  ([its own license](https://www.remotion.dev/license)).
- [ffmpeg](https://ffmpeg.org) with libass: cutting, layouts, captions.
- TypeSafe's [typesafe-ai skill](https://github.com/typesafe-ai/skills)
  (MIT): the basis of the Laya question skill (copy in `skills/`).

## License

[MIT](LICENSE) for this project's code. Models and libraries keep their
own licenses (see Credits).
