"""Step 4b: the AI makes sure every clip has a real start and a real end.

The first AI call picks clips from a long transcript and often starts one on
an answer ("The answer is 1,162,500 XP") or stops it on a setup ("The next
major factor is..."). This second, small call sees the transcript as
numbered sentences and each clip as a sentence range, says whether it has a
start and an end, and moves the range to sentence boundaries that give it
both. On 18 hand-labelled clips of a real video it judged the start right
15 times and the end right 12-14 times (Laya on the whole text: chance).

Every change is checked: the new range must be 10-90 s and must not run into
an earlier clip; a later clip it runs into is merged in (the proposal split
one thought in two). Otherwise the clip keeps its range. Laya's fixed layer then
scores the result, so a clip that still has no start or end ranks lower.
"""
from __future__ import annotations

from clipper.ai import AIConfig, AIError, complete_json
from clipper.captions import join_words
from clipper.skills import body as skill

MIN_CLIP, MAX_CLIP = 10.0, 90.0
_ENDS = (".", "?", "!", "。", "？", "！")

SCHEMA = {
    "type": "object",
    "properties": {"clips": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "has_start": {"type": "boolean"},
                       "has_end": {"type": "boolean"}, "start": {"type": "integer"},
                       "end": {"type": "integer"}, "why": {"type": "string"}},
        "required": ["id", "has_start", "has_end", "start", "end", "why"],
        "additionalProperties": False}}},
    "required": ["clips"],
    "additionalProperties": False,
}


def sentences(words: list[dict]) -> list[list[dict]]:
    """The words grouped into sentences, in time order."""
    out, current = [], []
    for word in sorted(words, key=lambda w: w["start"]):
        current.append(word)
        if word["word"].strip().endswith(_ENDS):
            out.append(current)
            current = []
    if current:
        out.append(current)
    return out


def sentence_range(clip: dict, sents: list[list[dict]]) -> tuple[int, int]:
    """The first and last sentence a clip touches."""
    first = next(i for i, s in enumerate(sents) if s[-1]["end"] > clip["start"])
    last = max(i for i, s in enumerate(sents) if s[0]["start"] < clip["end"])
    return first, max(first, last)


def _prompt(sents: list[list[dict]], ranges: list[tuple[int, int]], content_type: str) -> str:
    lines = "\n".join(f"[{i}] ({s[0]['start']:.0f}s) {join_words([w['word'] for w in s])}"
                      for i, s in enumerate(sents))
    clips = "\n".join(f"clip {k}: sentences {a}-{b}" for k, (a, b) in enumerate(ranges))
    return f"Kind of video: {content_type}\n\nTranscript:\n{lines}\n\nClips:\n{clips}"


def check(config: AIConfig, candidates: list[dict], words: list[dict], content_type: str,
          rebuild, complete=complete_json) -> tuple[list[dict], str | None]:
    """The clips with checked, possibly moved boundaries, and an error if the
    check could not run (the clips are then unchanged). `rebuild(start, end,
    old)` makes a clip for a new range."""
    sents = sentences(words)
    if not candidates or not sents:
        return candidates, None
    ranges = [sentence_range(c, sents) for c in candidates]
    try:
        answer = complete(config, skill("clip_boundaries"),
                          _prompt(sents, ranges, content_type), SCHEMA, 8000)
    except AIError as exc:
        return candidates, str(exc)
    verdicts = {v["id"]: v for v in answer.get("clips") or []
                if isinstance(v, dict) and isinstance(v.get("id"), int)}

    out: list[dict] = []
    merged: set[int] = set()  # later clips swallowed by an earlier one
    for k, clip in enumerate(candidates):
        if k in merged:
            continue
        verdict = verdicts.get(k)
        if verdict is None:
            out.append({**clip, "boundary": None})
            continue
        a, b = verdict.get("start", ranges[k][0]), verdict.get("end", ranges[k][1])
        note = {"has_start": bool(verdict.get("has_start")),
                "has_end": bool(verdict.get("has_end")),
                "why": str(verdict.get("why") or ""), "moved": False,
                "was": [clip["start"], clip["end"]]}
        valid = isinstance(a, int) and isinstance(b, int) and 0 <= a <= b < len(sents)
        if valid and (a, b) != ranges[k]:
            start, end = sents[a][0]["start"], sents[b][-1]["end"]
            # A later clip the new range runs into is the other half of the
            # same thought (the proposal split one story in two): it is merged
            # in and the clip grows to cover both. An earlier clip, already
            # settled, blocks the move instead.
            inside = [j for j in range(k + 1, len(candidates)) if j not in merged
                      and candidates[j]["start"] < end - 0.05
                      and candidates[j]["end"] > start + 0.05]
            if inside:
                end = max(end, *(candidates[j]["end"] for j in inside))
            # Earlier clips as already checked, later ones as proposed.
            neighbours = out + [c for j, c in enumerate(candidates[k + 1:], start=k + 1)
                                if j not in merged and j not in inside]
            clear = all(end <= n["start"] or start >= n["end"] for n in neighbours)
            if MIN_CLIP <= end - start <= MAX_CLIP and clear:
                clip = rebuild(start, end, clip)
                note["moved"] = True
                if inside:
                    merged.update(inside)
                    note["merged"] = [candidates[j]["id"] for j in inside]
                    note["why"] += f" (Merged with the clip{'s' if len(inside) > 1 else ''} it split off.)"
            else:
                note["why"] += " (The better range was not used: "
                note["why"] += ("it would overlap another clip.)" if not clear
                                else f"it would be {end - start:.0f} s long.)")
        out.append({**clip, "boundary": note})
    return out, None
