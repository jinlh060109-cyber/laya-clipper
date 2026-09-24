import json

from clipper import segment
from clipper.ai import AIConfig, AIError
from clipper.run import Run


def _transcript(sentences, gap=0.3, pause=0.6):
    """Sentences spoken back to back; each word lasts `gap` seconds."""
    segments, t = [], 0.0
    for sentence in sentences:
        words = []
        for token in sentence.split():
            words.append({"word": token, "start": round(t, 2), "end": round(t + gap * 0.9, 2),
                          "score": 0.9, "speaker": "SPEAKER_00"})
            t += gap
        segments.append({"start": words[0]["start"], "end": words[-1]["end"],
                         "speaker": "SPEAKER_00", "text": sentence, "words": words})
        t += pause
    return {"language": "en", "segments": segments}


SENTENCE = "this is a fairly ordinary sentence with exactly twelve words in it."
LONG = _transcript([SENTENCE] * 30)  # about 4.2 s per sentence, 126 s total


def _words(t):
    return [w for s in t["segments"] for w in s["words"]]


def test_transcript_lines_carry_timestamps():
    lines = segment.transcript_lines(_transcript(["hello there friend.", "second one here."]))
    first, second = lines.splitlines()
    assert first == "[0.00-0.87] hello there friend."
    assert second.startswith("[1.50-")


def test_snap_moves_boundaries_onto_words_and_fills_in_text():
    words = _words(LONG)
    got = segment.snap([{"start": 0.1, "end": 20.0, "category": "tip", "reason": "r",
                         "hook_line": "h"}], words, 130.0)
    clip = got[0]
    assert clip["start"] == words[0]["start"] or clip["start"] == words[1]["start"]
    assert any(abs(clip["end"] - w["end"]) < 1e-9 for w in words)
    assert clip["end"] < 20.0 + 0.3  # finishes the word the end point falls inside
    assert clip["text"].startswith("this is") or clip["text"].startswith("is a")
    assert clip["est_tokens"] > 0 and clip["truncated"] is False


def test_snap_drops_short_clips_clamps_the_end_and_removes_overlaps():
    words = _words(LONG)
    got = segment.snap([
        {"start": 50.0, "end": 55.0, "category": "a", "reason": "", "hook_line": ""},   # 5 s
        {"start": 10.0, "end": 40.0, "category": "b", "reason": "", "hook_line": ""},
        {"start": 30.0, "end": 60.0, "category": "c", "reason": "", "hook_line": ""},   # overlaps b
        {"start": 110.0, "end": 999.0, "category": "d", "reason": "", "hook_line": ""},
    ], words, 130.0)
    assert [c["category"] for c in got] == ["b", "d"]
    assert got[1]["end"] <= words[-1]["end"]
    assert [c["id"] for c in got] == [0, 1]


def test_a_clip_too_long_for_laya_is_kept_but_marked():
    words = _words(LONG)
    got = segment.snap([{"start": 0.0, "end": 125.0, "category": "x", "reason": "",
                         "hook_line": ""}], words, 130.0)
    assert got[0]["truncated"] is True


def test_fallback_chunks_end_on_sentences_and_stay_between_20_and_45_seconds():
    chunks = segment.fallback_candidates(LONG, 130.0)
    assert chunks
    for chunk in chunks[:-1]:
        assert 20.0 <= chunk["duration"] <= 45.0
        assert chunk["text"].endswith(".")
    assert all(c["category"] == "chunk" for c in chunks)


def _run(tmp_path):
    run = Run.create(tmp_path, "r")
    run.write_json("transcript.json", LONG)
    run.write_json("source.json", {"duration": 130.0})
    return run


AI_ANSWER = {
    "content_type": "tutorial", "summary": "How long things take.",
    "questions": [{"id": "useful", "type": "score", "instructions": "How useful?",
                   "levels": ["no", "some", "very"], "choices": [], "true_means": "",
                   "false_means": ""},
                  {"id": "broken", "type": "score", "instructions": "x", "levels": ["one"],
                   "choices": [], "true_means": "", "false_means": ""}],
    "weights": [{"question_id": "useful", "weight": 1.0}],
    "candidates": [{"start": 10.0, "end": 40.0, "category": "tip", "reason": "clear tip",
                    "hook_line": "Here is the trick."}],
}
CLAUDE = AIConfig("anthropic", "claude-opus-5", "k", None)


def test_when_the_ai_fails_the_run_falls_back_to_chunks(tmp_path):
    run = _run(tmp_path)

    def complete(*a, **k):
        raise AIError("Claude rejected the API key.")

    out = segment.run_segment(run, CLAUDE, complete=complete)
    assert out["ai"] is None and "API key" in out["ai_error"]
    assert out["candidates"] and all(c["category"] == "chunk" for c in out["candidates"])
    assert set(out["questions"]) == {"clipworthy", "hook_strength", "self_contained", "ends_cleanly"}


def test_when_the_ai_proposes_nothing_usable_the_run_falls_back(tmp_path):
    run = _run(tmp_path)
    answer = {**AI_ANSWER, "candidates": [{"start": 1.0, "end": 3.0, "category": "x",
                                           "reason": "", "hook_line": ""}]}
    out = segment.run_segment(run, CLAUDE, complete=lambda *a, **k: answer)
    assert out["candidates"][0]["category"] == "chunk"
    assert "no usable clips" in out["ai_error"]


def test_a_video_with_no_speech_has_no_candidates(tmp_path):
    run = Run.create(tmp_path, "r")
    run.write_json("transcript.json", {"language": "en", "segments": []})
    run.write_json("source.json", {"duration": 60.0})
    called = []
    out = segment.run_segment(run, CLAUDE, complete=lambda *a, **k: called.append(1))
    assert out["candidates"] == [] and called == []


def test_schema_is_strict_everywhere():
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(segment.SCHEMA)
    json.dumps(segment.SCHEMA)
