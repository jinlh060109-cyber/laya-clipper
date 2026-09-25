"""The AI director: its checks on the plan, the placement slots, and one
full agent loop with a scripted model (no AI, no Remotion)."""
import json

import pytest

from clipper import director
from clipper.captions import picture_band, reserved_bands


def test_unknown_components_and_bad_props_are_refused_with_a_reason():
    plan, notes = director.add_items([], [
        {"component": "explosion", "from": 1, "to": 3, "props": {}},
        {"component": "kinetic_text", "from": 1, "to": 3, "props": {"text": "x" * 60}},
        {"component": "pointer", "from": 1, "to": 3, "props": {"x": 2, "y": 0.5}},
        {"component": "counter", "from": 1, "to": 20, "props": {"value": 3}},
        {"component": "lower_third", "from": 1, "to": 3, "props": {}},
    ], 30.0)
    assert plan == []
    assert "Unknown component" in notes[0] and "under 40" in notes[1]
    assert "0 to 1" in notes[2] and "0.8-8 s" in notes[3] and "needs title" in notes[4]


def test_two_text_graphics_in_the_same_place_at_once_are_refused():
    plan, notes = director.add_items([], [
        {"component": "kinetic_text", "from": 1, "to": 3, "props": {"text": "A", "position": "top"}},
        {"component": "counter", "from": 2, "to": 4, "props": {"value": 5, "position": "top"}},
        {"component": "counter", "from": 2, "to": 4, "props": {"value": 5, "position": "lower"}},
    ], 30.0)
    assert [p["component"] for p in plan] == ["kinetic_text", "counter"]
    assert "would sit on the kinetic_text" in notes[1]


def test_a_progress_bar_may_run_the_whole_clip_and_times_are_clamped():
    plan, _ = director.add_items([], [{"component": "progress_bar", "from": -2, "to": 99,
                                       "props": {}}], 30.0)
    assert plan[0]["from"] == 0 and plan[0]["to"] == 30.0


def test_slots_use_the_empty_bands_of_a_fit_layout_and_the_picture_of_a_crop():
    fit = director.slots(reserved_bands("fit", 1920), picture_band("fit", 1920), 1920)
    top, bottom = picture_band("fit", 1920)
    assert fit["top"]["y"] < top and fit["lower"]["y"] > bottom
    crop_bands = reserved_bands("crop", 1920)
    crop = director.slots(crop_bands, picture_band("crop", 1920), 1920)
    assert all(crop_bands["title_bottom"] < s["y"] < crop_bands["caption_top"] for s in crop.values())


class ScriptedChat:
    """A model that looks, plans, previews and renders."""

    def __init__(self, config, system, tools):
        self.tools = {t["name"] for t in tools}
        self.vision = True
        self.replies = []
        self.said = []
        self.turns = iter([
            [{"id": "1", "name": "look", "arguments": {"times": [1, 5]}}],
            [{"id": "2", "name": "add_graphics", "arguments": {"items": [
                {"component": "counter", "from": 4, "to": 7, "props": {"value": 813, "suffix": " games"}},
                {"component": "sticker", "from": 1, "to": 3, "props": {}}]}}],
            [{"id": "3", "name": "preview", "arguments": {"times": [5]}}],
            [{"id": "4", "name": "render", "arguments": {"summary": "One counter."}}],
        ])

    def say(self, text):
        self.said.append(text)

    def step(self, max_tokens):
        return "", next(self.turns)

    def reply(self, results):
        self.replies.append(results)


def test_the_agent_loop_looks_plans_previews_and_renders(tmp_path, monkeypatch):
    (tmp_path / "final.mp4").write_bytes(b"video")
    monkeypatch.setattr(director, "frame_jpeg", lambda ffmpeg, video, at, png=None: b"jpeg")
    drawn, rendered = [], []

    def fake_node(args, what, timeout=900):
        props = json.loads(open(args[1], encoding="utf-8").read())
        drawn.append(props)
        for t in args[3].split(","):
            (tmp_path / "x").mkdir(exist_ok=True)
        return "done"
    monkeypatch.setattr(director, "_node", fake_node)
    monkeypatch.setattr(director, "render_over", lambda ffmpeg, folder, props, enc: rendered.append(props))
    chats = []

    def factory(*a):
        chats.append(ScriptedChat(*a))
        return chats[-1]
    bands = reserved_bands("fit", 1920)
    record = director.direct(None if False else type("C", (), {"provider": "p", "model": "m"})(),
                             tmp_path, "[0.00-4.00] you need 813 games", {"id": "c1", "duration": 30.0},
                             "", ffmpeg="ffmpeg", width=1080, height=1920, fps=30, bands=bands,
                             encoder=[], picture=picture_band("fit", 1920), chat_factory=factory)
    chat = chats[0]
    assert chat.tools == {"look", "add_graphics", "remove_graphic", "preview", "render"}
    assert "813 games" in chat.said[0] and "empty band" in chat.said[0]
    look = chat.replies[0][0][1]
    assert [b["type"] for b in look] == ["text", "image", "image"]
    assert "refused 'sticker': sticker needs text." in chat.replies[1][0][1]
    assert drawn and drawn[0]["items"][0]["props"]["value"] == 813  # preview drew the plan
    assert rendered and rendered[0]["slots"]["top"] < 600
    assert record["rendered"] and record["summary"] == "One counter."
    saved = json.loads((tmp_path / "director.json").read_text(encoding="utf-8"))
    assert [c["tool"] for c in saved["calls"]] == ["look", "add_graphics", "preview", "render"]


def test_director_needs_an_ai(tmp_path):
    with pytest.raises(ValueError, match="needs an AI"):
        director.run_director(None, {}, [], None)
