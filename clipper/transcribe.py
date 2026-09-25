from __future__ import annotations

from pathlib import Path

from clipper.device import resolve_device
from clipper.run import Run

DEFAULT_SPEAKER = "SPEAKER_00"
SAMPLE_RATE = 16000


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
        # Gameplay without commentary is a real input, not an error: the
        # action stage finds its moments from picture and sound instead.
        return {"language": language, "model": model,
                "diarized": diarized, "segments": []}
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


def segments_from_chunks(chunks: list[dict], texts: list[str]) -> list[dict]:
    """Pair each voice chunk with what Whisper heard in it, like whisperx does."""
    return [{"start": chunk["start"], "end": chunk["end"], "text": text}
            for chunk, text in zip(chunks, texts) if text.strip()]


def _faster_whisper_asr(audio, model: str, device: str) -> dict:
    import whisperx

    compute_type = "float16" if device == "cuda" else "int8"
    asr = whisperx.load_model(model, device, compute_type=compute_type)
    result = asr.transcribe(audio, batch_size=16)
    del asr  # see _free_device_memory
    return result


def _transformers_asr(audio, model: str, device: str, batch_size: int = 8) -> dict:
    """Whisper for GPUs CTranslate2 cannot use (Intel Arc, Apple). The same
    recipe as whisperx: cut the audio into <=30 s voice chunks, then
    transcribe the chunks in batches."""
    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration
    from whisperx.vads import Pyannote

    vad_options = {"onset": 0.500, "offset": 0.363}
    vad = Pyannote(torch.device(device), token=None, vad_onset=vad_options["onset"],
                   vad_offset=vad_options["offset"])
    chunks = Pyannote.merge_chunks(
        vad({"waveform": Pyannote.preprocess_audio(audio), "sample_rate": SAMPLE_RATE}),
        30, **vad_options)

    processor = AutoProcessor.from_pretrained(f"openai/whisper-{model}")
    whisper = WhisperForConditionalGeneration.from_pretrained(
        f"openai/whisper-{model}", dtype=torch.float16).to(device).eval()

    def features(pieces):
        # The attention mask marks the padding of chunks under 30 s; without
        # it transformers warns that it cannot tell padding from speech.
        batch = processor(pieces, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                          return_attention_mask=True)
        return (batch.input_features.to(device, torch.float16),
                batch.attention_mask.to(device))

    texts: list[str] = []
    with torch.inference_mode():
        # Like whisperx: the language is read from the first 30 s of audio.
        token = whisper.detect_language(features([audio[:30 * SAMPLE_RATE]])[0])[0]
        language = processor.tokenizer.convert_ids_to_tokens(int(token)).strip("<|>")
        for i in range(0, len(chunks), batch_size):
            pieces = [audio[int(c["start"] * SAMPLE_RATE):int(c["end"] * SAMPLE_RATE)]
                      for c in chunks[i:i + batch_size]]
            inputs, mask = features(pieces)
            # A static KV cache keeps tensor shapes fixed. With the default
            # growing cache an Intel GPU compiles new kernels at every decoding
            # step: minutes of warm-up. torch.compile would need a C compiler.
            ids = whisper.generate(inputs, attention_mask=mask, task="transcribe",
                                   language=language, cache_implementation="static",
                                   disable_compile=True)
            # Whisper's tokenizer is BPE: transformers ignores the clean-up
            # anyway, so say so instead of being warned about it every batch.
            texts += processor.batch_decode(ids, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
    del whisper, vad  # see _free_device_memory
    return {"segments": segments_from_chunks(chunks, texts), "language": language}


def uses_faster_whisper(device: str, rocm: bool) -> bool:
    """faster-whisper (CTranslate2) runs on NVIDIA CUDA and the CPU only.
    An AMD ROCm build of torch also calls its GPU `cuda`, but CTranslate2
    cannot use it, so those runs go through transformers like Intel and Apple."""
    return device == "cpu" or (device == "cuda" and not rocm)


def _free_device_memory(device: str) -> None:
    """See hardware.free_device_memory. The stages below `del` their models
    explicitly: a library that catches a failed import (torchcodec, here)
    keeps its traceback, and with it every frame below, locals included."""
    from clipper.hardware import free_device_memory

    free_device_memory(device)


def transcribe(wav: Path, run: Run, model: str = "large-v3",
               device: str | None = None, hf_token: str | None = None) -> dict:
    device = resolve_device(device)
    try:
        # The models live in _transcribe's frame, so they are unreferenced here.
        return _transcribe(wav, run, model, device, hf_token)
    finally:
        _free_device_memory(device)


def _import_whisperx():
    """whisperx, with pyannote's audio module loaded without its torchcodec
    warning. pyannote decodes files through torchcodec, which needs FFmpeg
    4-7 as shared DLLs; this app never hands it a file. whisperx.load_audio
    decodes with the ffmpeg program, and voice detection and diarization get
    the waveform in memory, the route pyannote's own warning recommends. So
    the warning (a long traceback on every run with FFmpeg 8+ or a static
    build) says nothing about this app, and only this warning is silenced."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"\s*torchcodec is not installed correctly",
                                category=UserWarning)
        try:
            import pyannote.audio.core.io  # noqa: F401 - the module that warns on import
        except ImportError:
            pass  # whisperx reports a missing pyannote itself, if it needs it
        import whisperx
    return whisperx


def _transcribe(wav: Path, run: Run, model: str, device: str,
                hf_token: str | None) -> dict:
    whisperx = _import_whisperx()
    audio = whisperx.load_audio(str(wav))
    from clipper.hardware import is_rocm

    fast = uses_faster_whisper(device, is_rocm() if device == "cuda" else False)
    asr = _faster_whisper_asr if fast else _transformers_asr
    result = asr(audio, model, device)
    language = result["language"]

    # Nothing was said (gameplay without commentary): there is nothing to
    # align or attribute, and whisperx's aligner does not accept an empty list.
    if result["segments"]:
        try:
            align_model, metadata = whisperx.load_align_model(language_code=language,
                                                              device=device)
        except ValueError:
            # whisperx has no aligner for this language (or Whisper misread game
            # audio as, say, Welsh). Keep the words; their timings are spread
            # evenly across each segment by normalize_transcript.
            print(f"No word aligner for language {language!r}; word timings are estimated.")
            result = {"segments": [
                {**seg, "words": [{"word": w} for w in seg.get("text", "").split()]}
                for seg in result["segments"]]}
        else:
            result = whisperx.align(result["segments"], align_model, metadata,
                                    audio, device, return_char_alignments=False)
            del align_model  # see _free_device_memory

    # The caller decides: None means "do not diarize" (--no-diarize, the web
    # toggle), so there is deliberately no fallback to $HF_TOKEN here.
    diarized = False
    if hf_token and result["segments"]:
        from whisperx.diarize import DiarizationPipeline
        pipeline = DiarizationPipeline(use_auth_token=hf_token, device=device)
        result = whisperx.assign_word_speakers(pipeline(audio), result)
        del pipeline  # see _free_device_memory
        diarized = True
    else:
        print("Diarization off (no HF_TOKEN, or switched off). All speech labelled SPEAKER_00.")

    transcript = normalize_transcript(result, model, diarized, language)
    run.write_json("transcript.json", transcript)
    return transcript
