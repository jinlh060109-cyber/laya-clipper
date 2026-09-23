from __future__ import annotations

ISLAND_WORDS = 5
ISLAND_SECONDS = 3.0


def _utterances(words: list[dict], min_gap: float) -> list[tuple[float, float, int]]:
    """(start, end, word count) runs of words with no gap of `min_gap` or more."""
    out: list[list[float]] = []
    for word in sorted(words, key=lambda w: w["start"]):
        if out and word["start"] - out[-1][1] < min_gap:
            out[-1][1] = max(out[-1][1], word["end"])
            out[-1][2] += 1
        else:
            out.append([word["start"], word["end"], 1])
    return [(s, e, int(n)) for s, e, n in out]


def silent_spans(transcript: dict, duration: float, min_gap: float = 8.0) -> list[list[float]]:
    """Stretches of at least `min_gap` seconds in which nobody really speaks.

    A short isolated utterance does not count as speech: over game music it is
    usually Whisper inventing "Thanks for watching!", and a lone "let's go!"
    mid-fight is part of the action, not a reason to split it.
    """
    words = [w for seg in transcript.get("segments") or [] for w in seg.get("words") or []]
    speech = [(min(s, duration), min(e, duration))
              for s, e, n in _utterances(words, min_gap)
              if not (n <= ISLAND_WORDS and e - s <= ISLAND_SECONDS)]
    spans: list[list[float]] = []
    cursor = 0.0
    for start, end in speech:
        if start - cursor >= min_gap:
            spans.append([round(cursor, 3), round(start, 3)])
        cursor = max(cursor, end)
    if duration - cursor >= min_gap:
        spans.append([round(cursor, 3), round(duration, 3)])
    return spans
