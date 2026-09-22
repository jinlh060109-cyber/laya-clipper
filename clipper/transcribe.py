from __future__ import annotations

import os
from pathlib import Path

from clipper.run import Run

DEFAULT_SPEAKER = "SPEAKER_00"


def _fill_timings(words: list[dict], seg_start: float, seg_end: float) -> list[dict]:
    """Give every word a start and end, interpolating across unaligned runs."""
    filled = [dict(w) for w in words]
    known = [i for i, w in enumerate(filled) if "start" in w and "end" in w]
    if not known:
        span = (seg_end - seg_start) / max(1, len(filled))
        for i, w in enumerate(filled):
            w["start"] = seg_start + i * span
            w["end"] = seg_start + (i + 1) * span
            w["score"] = 0.0
        return filled
    for i, word in enumerate(filled):
        if "start" in word and "end" in word:
            continue
        prev = max((k for k in known if k < i), default=None)
        nxt = min((k for k in known if k > i), default=None)
        lo = filled[prev]["end"] if prev is not None else seg_start
        hi = filled[nxt]["start"] if nxt is not None else seg_end
        gap = [j for j in range(len(filled)) if j not in known
               and (prev is None or j > prev) and (nxt is None or j < nxt)]
        share = (hi - lo) / max(1, len(gap))
        slot = gap.index(i)
        word["start"] = lo + slot * share
        word["end"] = lo + (slot + 1) * share
        word["score"] = 0.0
    return filled


def normalize_transcript(result: dict, model: str, diarized: bool, language: str) -> dict:
    segments = result.get("segments") or []
    if not segments:
        raise ValueError(
            "Transcription produced no speech. The audio may be silent or music only."
        )
    out_segments = []
    for seg in segments:
        speaker = seg.get("speaker", DEFAULT_SPEAKER)
        words = _fill_timings(seg.get("words", []), float(seg["start"]), float(seg["end"]))
        out_segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "speaker": speaker,
            "text": seg.get("text", "").strip(),
            "words": [{
                "word": w["word"],
                "start": float(w["start"]),
                "end": float(w["end"]),
                "score": float(w.get("score", 0.0)),
                "speaker": w.get("speaker", speaker),
            } for w in words],
        })
    return {"language": language, "model": model,
            "diarized": diarized, "segments": out_segments}


def transcribe(wav: Path, run: Run, model: str = "large-v3",
               device: str | None = None, hf_token: str | None = None) -> dict:
    import torch
    import whisperx

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = "float16" if device == "cuda" else "int8"

    audio = whisperx.load_audio(str(wav))
    asr = whisperx.load_model(model, device, compute_type=compute_type)
    result = asr.transcribe(audio, batch_size=16)
    language = result["language"]

    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(result["segments"], align_model, metadata,
                            audio, device, return_char_alignments=False)

    token = hf_token or os.environ.get("HF_TOKEN")
    diarized = False
    if token:
        from whisperx.diarize import DiarizationPipeline
        pipeline = DiarizationPipeline(use_auth_token=token, device=device)
        result = whisperx.assign_word_speakers(pipeline(audio), result)
        diarized = True
    else:
        print("HF_TOKEN not set; skipping diarization. All speech labelled SPEAKER_00.")

    transcript = normalize_transcript(result, model, diarized, language)
    run.write_json("transcript.json", transcript)
    return transcript
