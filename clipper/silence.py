from __future__ import annotations

ISLAND_WORDS = 5
ISLAND_SECONDS = 3.0
# Real speech runs at two or three words a second; Whisper's inventions over
# game audio stretch a word or two across half a minute.
MIN_WORDS_PER_SECOND = 1.0


def _utterances(words: list[dict], min_gap: float) -> list[dict]:
    """Runs of words with no gap of `min_gap` or more between them."""
    out: list[dict] = []
    for word in sorted(words, key=lambda w: w["start"]):
        # Words without a score predate alignment scores; trust them.
        aligned = word.get("score", 1.0) > 0
        if out and word["start"] - out[-1]["end"] < min_gap:
            out[-1]["end"] = max(out[-1]["end"], word["end"])
            out[-1]["words"] += 1
            out[-1]["aligned"] = out[-1]["aligned"] or aligned
        else:
            out.append({"start": word["start"], "end": word["end"],
                        "words": 1, "aligned": aligned})
    return out


def _is_speech(utterance: dict) -> bool:
    """Whether an utterance is someone really talking.

    Not speech: a short isolated utterance (over game music usually Whisper
    inventing "Thanks for watching!", and a lone "let's go!" mid-fight is part
    of the action), a run too sparse to be talk, or one where the aligner
    placed none of the words.
    """
    length = utterance["end"] - utterance["start"]
    if utterance["words"] <= ISLAND_WORDS and length <= ISLAND_SECONDS:
        return False
    if length > 0 and utterance["words"] / length < MIN_WORDS_PER_SECOND:
        return False
    return utterance["aligned"]


def silent_spans(transcript: dict, duration: float, min_gap: float = 8.0) -> list[list[float]]:
    """Stretches of at least `min_gap` seconds in which nobody really speaks."""
    words = [w for seg in transcript.get("segments") or [] for w in seg.get("words") or []]
    speech = [(min(u["start"], duration), min(u["end"], duration))
              for u in _utterances(words, min_gap) if _is_speech(u)]
    spans: list[list[float]] = []
    cursor = 0.0
    for start, end in speech:
        if start - cursor >= min_gap:
            spans.append([round(cursor, 3), round(start, 3)])
        cursor = max(cursor, end)
    if duration - cursor >= min_gap:
        spans.append([round(cursor, 3), round(duration, 3)])
    return spans
