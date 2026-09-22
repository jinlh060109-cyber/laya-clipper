# Laya Claude Clipper

Turns a long video into short, upload-ready clips.

Laya scores every 30-second window of the transcript for clip-worthiness,
Claude selects and trims the clips and writes their copy, and ffmpeg renders
them with captions. Laya runs locally: no API key, no per-clip cost.

Design: `docs/superpowers/specs/2026-09-22-laya-claude-clipper-design.md`

## Requirements

- Python 3.11+, ffmpeg with libass
- A GPU is strongly recommended for scoring. Measured on a 90-minute source:
  13 minutes on an Intel Arc 140T, 78 minutes on CPU.
- Intel Arc users: install a torch build with XPU support. Laya's own device
  detection cannot see Arc, so set `CLIPPER_LAYA_DEVICE=xpu` or rely on
  clipper's `auto`, which probes for it.

## Setup

    pip install -e ".[dev]"
    cp .env.example .env    # optionally HF_TOKEN (diarization), CLIPPER_LAYA_DEVICE

ffmpeg must be on PATH and must include libass. Check:

    ffmpeg -filters | grep subtitles

## Use

    clipper all "episode47.mp4" --profile podcast --vertical

Then ask Claude to run the plan stage, and:

    clipper plan runs/2026-09-21-episode47 --check
    clipper render runs/2026-09-21-episode47 --vertical

Profiles: `podcast`, `talking_head`, `lecture`, `stream`.
Re-scoring with a different profile reuses the cached transcript and windows;
only the Laya pass runs again.

## Tests

    pytest              # offline; no model download
    pytest -m model     # loads the real Laya checkpoint
