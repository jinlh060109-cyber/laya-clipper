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
        if out and word["start"] - out[-1]["end"] < min_gap:
            out[-1]["end"] = max(out[-1]["end"], word["end"])
            out[-1]["words"].append(word)
        else:
            out.append({"start": word["start"], "end": word["end"], "words": [word]})
    return out


def _is_speech(utterance: dict) -> bool:
    """Whether an utterance is someone really talking.

    Not speech: a short isolated utterance (over game music usually Whisper
    inventing "Thanks for watching!", and a lone "let's go!" mid-fight is part
    of the action), or a run too sparse to be talk.
    """
    count = len(utterance["words"])
    length = utterance["end"] - utterance["start"]
    if count <= ISLAND_WORDS and length <= ISLAND_SECONDS:
        return False
    return not (length > 0 and count / length < MIN_WORDS_PER_SECOND)


def _all_words(transcript: dict) -> list[dict]:
    return [w for seg in transcript.get("segments") or [] for w in seg.get("words") or []]


def speech_only(transcript: dict, min_gap: float = 8.0) -> dict:
    """The transcript with only the words that are someone really talking.

    Laya reads this; everything else is judged by the action stage instead.
    """
    kept = {id(w) for u in _utterances(_all_words(transcript), min_gap)
            if _is_speech(u) for w in u["words"]}
    segments = []
    for seg in transcript.get("segments") or []:
        words = [w for w in seg.get("words") or [] if id(w) in kept]
        if words:
            segments.append({**seg, "words": words})
    return {**transcript, "segments": segments}


def silent_spans(transcript: dict, duration: float, min_gap: float = 8.0) -> list[list[float]]:
    """Stretches of at least `min_gap` seconds in which nobody really speaks."""
    speech = [(min(u["start"], duration), min(u["end"], duration))
              for u in _utterances(_all_words(transcript), min_gap) if _is_speech(u)]
    spans: list[list[float]] = []
    cursor = 0.0
    for start, end in speech:
        if start - cursor >= min_gap:
            spans.append([round(cursor, 3), round(start, 3)])
        cursor = max(cursor, end)
    if duration - cursor >= min_gap:
        spans.append([round(cursor, 3), round(duration, 3)])
    return spans
