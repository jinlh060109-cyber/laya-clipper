import pytest

from clipper.profiles.loader import Profile, ProfileError, available_profiles, load_profile


def test_core_profile_loads_with_expected_questions():
    p = load_profile("core")
    assert p.name == "core"
    assert p.questions["clipworthy"]["type"] == "score"
    assert len(p.questions["clipworthy"]["criteria"]) == 5
    assert p.questions["hook_type"]["type"] == "choice"
    assert p.questions["open_loop"]["type"] == "noul"


def test_podcast_extends_core_and_adds_its_own():
    core, podcast = load_profile("core"), load_profile("podcast")
    assert "disagreement" in podcast.questions
    assert "clipworthy" in podcast.questions          # inherited
    assert podcast.weights["clipworthy"] == core.weights["clipworthy"]
    assert podcast.weights["disagreement"] == 0.06    # added


def test_uncertain_confidence_is_a_per_primitive_map():
    p = load_profile("core")
    assert p.uncertain_confidence == {"score": 0.10, "choice": 0.15, "noul": 0.55}
    assert "uncertain_confidence" not in p.thresholds
    assert p.thresholds["candidate"] == 0.55


def test_scalar_uncertain_confidence_is_rejected(tmp_path, monkeypatch):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\nquestions: {}\nthresholds:\n  uncertain_confidence: 0.55\n",
                   encoding="utf-8")
    monkeypatch.setattr("clipper.profiles.loader.PROFILE_DIR", tmp_path)
    with pytest.raises(ProfileError, match="per-primitive map"):
        load_profile("bad")


def test_unknown_question_type_is_rejected(tmp_path, monkeypatch):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\nquestions:\n  q:\n    type: freeform\n    instructions: hi\n",
                   encoding="utf-8")
    monkeypatch.setattr("clipper.profiles.loader.PROFILE_DIR", tmp_path)
    with pytest.raises(ProfileError, match="freeform"):
        load_profile("bad")


def test_weight_referencing_a_missing_question_is_rejected(tmp_path, monkeypatch):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\nquestions: {}\nweights:\n  ghost: 0.5\n", encoding="utf-8")
    monkeypatch.setattr("clipper.profiles.loader.PROFILE_DIR", tmp_path)
    with pytest.raises(ProfileError, match="ghost"):
        load_profile("bad")


def test_score_criteria_must_be_a_list_and_choice_a_mapping():
    p = load_profile("core")
    assert isinstance(p.questions["clipworthy"]["criteria"], list)
    assert isinstance(p.questions["hook_type"]["criteria"], dict)


def test_all_five_profiles_are_available_and_load():
    names = available_profiles()
    assert set(names) >= {"core", "podcast", "talking_head", "lecture", "stream"}
    for n in names:
        load_profile(n)


def test_unknown_profile_names_what_exists():
    with pytest.raises(ProfileError, match="podcast"):
        load_profile("nope")


@pytest.mark.model
def test_every_question_fits_layas_question_head():
    """Laya truncates instructions and can drop options to fit head_max_len.

    Neither failure raises — both silently degrade the question — so assert the
    fit here rather than discovering it as bad scores later.
    """
    from laya.common import build_sequence, render_options
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    import os

    d = snapshot_download("convaiinnovations/laya",
                          allow_patterns=["rl_agent_config.json", "model.safetensors",
                                          "tokenizer/*", "encoder/*"])
    tok = AutoTokenizer.from_pretrained(os.path.join(d, "tokenizer"))
    state = "SPEAKER_01: " + ("word " * 400)

    for name in available_profiles():
        profile = load_profile(name)
        for qid, q in profile.questions.items():
            crit = q.get("criteria")
            if q["type"] == "choice" and isinstance(crit, list):
                crit = {c: None for c in crit}
            internal = {"t": q["type"], "ins": q["instructions"], "crit": crit}
            seq, markers = build_sequence(tok, state, internal, 512, 192)
            expected = len(render_options(internal))
            assert len(markers) == expected, (
                f"{name}/{qid}: {expected - len(markers)} option(s) dropped; shorten the criteria"
            )
            head = tok(f"{q['type']} question: {q['instructions']}",
                       add_special_tokens=False)["input_ids"]
            opt_ids = [[0] + tok(" " + o, add_special_tokens=False)["input_ids"][:48]
                       for o in render_options(internal)]
            budget = 192 - sum(len(o) for o in opt_ids)
            assert len(head) <= max(8, budget), (
                f"{name}/{qid}: instructions truncated by {len(head) - max(8, budget)} tokens"
            )
