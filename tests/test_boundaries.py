"""The AI boundary check: clips get moved to sentences with a real start and end."""
from clipper import boundaries
from clipper.ai import AIError


def _words(sentences):
    """Two seconds per word, sentences back to back."""
    out, t = [], 0.0
    for sentence in sentences:
        for token in sentence.split():
            out.append({"word": token, "start": t, "end": t + 1.8})
            t += 2.0
    return out


SENTS = ["So how much XP do you need?",          # 0: 7 words, 0-14 s
         "The answer is a lot of XP.",            # 1: 7 words, 14-28 s
         "That means you play every day.",        # 2: 6 words, 28-40 s
         "The next major factor is time."]        # 3: 6 words, 40-52 s
WORDS = _words(SENTS)


def _rebuild(start, end, old):
    return {**old, "start": start, "end": end, "duration": end - start, "text": "moved"}


def _answer(clips):
    return lambda config, system, user, schema, tokens: {"clips": clips}


def test_sentences_split_on_end_punctuation():
    sents = boundaries.sentences(WORDS)
    assert [len(s) for s in sents] == [7, 7, 6, 6]


def test_a_clip_is_moved_to_the_sentences_the_ai_picks():
    clip = {"id": "c1", "start": 14.0, "end": 51.8, "duration": 37.8}
    out, error = boundaries.check(None, [clip], WORDS, "gaming", _rebuild, _answer([
        {"id": 0, "has_start": False, "has_end": False, "start": 0, "end": 2,
         "why": "Starts on the answer."}]))
    assert error is None
    assert out[0]["start"] == 0.0 and out[0]["end"] == WORDS[19]["end"]
    assert out[0]["boundary"]["moved"] and out[0]["boundary"]["was"] == [14.0, 51.8]
    assert not out[0]["boundary"]["has_start"]


def test_a_range_that_would_run_into_an_earlier_clip_is_not_used():
    first = {"id": "c1", "start": 0.0, "end": 27.8, "duration": 27.8}
    second = {"id": "c2", "start": 28.0, "end": 51.8, "duration": 23.8}
    out, _ = boundaries.check(None, [first, second], WORDS, "gaming", _rebuild, _answer([
        {"id": 1, "has_start": False, "has_end": True, "start": 1, "end": 3, "why": "x"}]))
    assert out[1]["start"] == 28.0 and not out[1]["boundary"]["moved"]
    assert "overlap" in out[1]["boundary"]["why"]
    assert out[0]["boundary"] is None  # no verdict for it


def test_a_range_under_ten_seconds_is_not_used():
    words = _words(SENTS + ["Done."])  # sentence 4: one word, 2 s
    clip = {"id": "c1", "start": 0.0, "end": 27.8, "duration": 27.8}
    out, _ = boundaries.check(None, [clip], words, "gaming", _rebuild, _answer([
        {"id": 0, "has_start": True, "has_end": True, "start": 4, "end": 4, "why": "x"}]))
    assert out[0]["start"] == 0.0 and "long" in out[0]["boundary"]["why"]


def test_an_ai_failure_leaves_the_clips_as_they_were():
    def broken(*_):
        raise AIError("down")
    clip = {"id": "c1", "start": 0.0, "end": 27.8, "duration": 27.8}
    out, error = boundaries.check(None, [clip], WORDS, "gaming", _rebuild, broken)
    assert out == [clip] and error == "down"


def test_the_prompt_numbers_sentences_and_clip_ranges():
    seen = {}

    def spy(config, system, user, schema, tokens):
        seen["user"], seen["system"] = user, system
        return {"clips": []}
    clip = {"id": "c1", "start": 14.0, "end": 39.8, "duration": 25.8}
    boundaries.check(None, [clip], WORDS, "gaming", _rebuild, spy)
    assert "[1] (14s) The answer is a lot of XP." in seen["user"]
    assert "clip 0: sentences 1-2" in seen["user"]
    assert "has_start" in seen["system"]


def test_a_story_split_in_two_is_merged_back_into_one_clip():
    first = {"id": "c1", "start": 0.0, "end": 27.8, "duration": 27.8}
    second = {"id": "c2", "start": 28.0, "end": 39.8, "duration": 11.8}
    out, _ = boundaries.check(None, [first, second], WORDS, "comedy", _rebuild, _answer([
        {"id": 0, "has_start": True, "has_end": False, "start": 0, "end": 2, "why": "Punchline is later."},
        {"id": 1, "has_start": False, "has_end": True, "start": 0, "end": 2, "why": "Needs the setup."}]))
    assert len(out) == 1
    assert out[0]["start"] == 0.0 and out[0]["end"] == WORDS[19]["end"]
    assert out[0]["boundary"]["merged"] == ["c2"] and "Merged" in out[0]["boundary"]["why"]


def test_two_fixes_that_overlap_become_one_clip_covering_both():
    """What the real AI did on a camping story: each half was fixed toward
    the other, into ranges that overlap."""
    first = {"id": "c1", "start": 0.0, "end": 13.8, "duration": 13.8}   # sentence 0
    second = {"id": "c2", "start": 28.0, "end": 51.8, "duration": 23.8}  # sentences 2-3
    out, _ = boundaries.check(None, [first, second], WORDS, "comedy", _rebuild, _answer([
        {"id": 0, "has_start": True, "has_end": False, "start": 0, "end": 2, "why": "a"},
        {"id": 1, "has_start": False, "has_end": True, "start": 1, "end": 3, "why": "b"}]))
    assert len(out) == 1
    assert out[0]["start"] == 0.0 and out[0]["end"] == WORDS[-1]["end"]
    assert out[0]["boundary"]["merged"] == ["c2"]
