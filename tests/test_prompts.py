from clipper import prompts
from clipper.style import DEFAULT

CLIP = {"id": "c0", "start": 10.0, "end": 30.0, "duration": 20.0, "category": "tip",
        "score": 0.61, "confidence": 0.4, "uncertain": False, "title_hint": "Here is the trick.",
        "source": "ai", "reason": "clear tip"}
FILL = {"title": "The trick", "hook": "Watch this.", "description": "A tip.",
        "caption_quote": "Here is the trick.", "punch_ins": [{"at": 0.8, "reason": "reveal"}]}
STYLE = {**DEFAULT, "notes": "Brand: always say 'the crew'. Fast cuts."}


def test_without_fill_in_the_title_comes_from_the_hint():
    spec = prompts.edit_spec(CLIP, STYLE, None, "tutorial", "01-x")
    assert spec["title"] == "Here is the trick." and spec["fill_in_used"] is False
    assert spec["punch_ins"] == []


def test_prompt_has_the_style_verbatim_the_clip_and_its_words():
    spec = prompts.edit_spec(CLIP, STYLE, FILL, "tutorial", "01-the-trick")
    text = prompts.render_prompt(spec, STYLE["notes"], "[0.00-1.20] Here is the trick.")
    assert "Brand: always say 'the crew'. Fast cuts." in text
    assert "0:10" in text and "0:30" in text
    assert "The trick" in text and "reveal" in text
    assert "[0.00-1.20] Here is the trick." in text
    assert "final.mp4" in text


def test_an_action_clip_is_not_described_as_scored_by_laya():
    action = {**CLIP, "id": "a0", "source": "action", "category": "action", "score": 0.53,
              "confidence": 0.0}
    text = prompts.render_prompt(prompts.edit_spec(action, STYLE, None, "gaming", "01-x"), "", "")
    assert "Laya" not in text
    assert "Action score 0.53" in text
