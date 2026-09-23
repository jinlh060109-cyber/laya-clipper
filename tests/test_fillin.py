from clipper import fillin
from clipper.ai import AIConfig

TRANSCRIPT = {"segments": [
    {"words": [{"word": "Before.", "start": 5.0, "end": 5.5, "speaker": "S"}]},
    {"words": [{"word": "Here", "start": 10.0, "end": 10.3, "speaker": "S"},
               {"word": "is", "start": 10.4, "end": 10.5, "speaker": "S"},
               {"word": "the", "start": 10.6, "end": 10.7, "speaker": "S"},
               {"word": "trick.", "start": 10.8, "end": 11.2, "speaker": "S"}]},
    {"words": [{"word": "After.", "start": 45.0, "end": 45.5, "speaker": "S"}]},
]}
CLIP = {"id": "c0", "start": 10.0, "end": 30.0, "duration": 20.0, "category": "tip",
        "title_hint": "Here is the trick."}
CONFIG = AIConfig("anthropic", "claude-opus-5", "k", None)


def test_clip_slice_is_clip_local_and_only_the_clip():
    assert fillin.clip_slice(TRANSCRIPT, 10.0, 30.0) == "[0.00-1.20] Here is the trick."


def test_fill_in_reads_only_the_clip_and_clamps_punch_ins():
    seen = {}

    def complete(config, system, user, schema, max_tokens):
        seen.update(system=system, user=user, schema=schema)
        return {"title": "The trick", "hook": "Watch this.", "description": "A tip.",
                "caption_quote": "Here is the trick.",
                "punch_ins": [{"at": 0.8, "reason": "the reveal"}, {"at": 99.0, "reason": "late"}]}

    out = fillin.fill_in(CONFIG, CLIP, TRANSCRIPT, "Calm tone.", complete=complete)
    assert "Before" not in seen["user"] and "After" not in seen["user"]
    assert "Calm tone." in seen["user"]
    assert seen["schema"] == fillin.SCHEMA
    assert out["title"] == "The trick"
    assert [p["at"] for p in out["punch_ins"]] == [0.8, 20.0]
