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


def test_noul_with_unquoted_boolean_keys_is_rejected(tmp_path, monkeypatch):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: bad\nquestions:\n  q:\n    type: noul\n    instructions: hi\n"
        "    criteria:\n      true: yes\n      false: no\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("clipper.profiles.loader.PROFILE_DIR", tmp_path)
    with pytest.raises(ProfileError, match="parse as booleans"):
        load_profile("bad")


def test_all_five_profiles_are_available_and_load():
    names = available_profiles()
    assert set(names) >= {"core", "podcast", "talking_head", "lecture", "stream"}
    for n in names:
        load_profile(n)


def test_unknown_profile_names_what_exists():
    with pytest.raises(ProfileError, match="podcast"):
        load_profile("nope")


def test_noul_criteria_keys_are_the_strings_true_and_false():
    """Unquoted `true:`/`false:` in YAML parse as Python booleans, not strings.

    laya.common.render_options looks up crit.get("true")/crit.get("false") with
    string keys, so boolean keys miss silently and Laya falls back to generic
    text -- the hand-written criteria never reach the model. This must hold for
    every noul question in every profile.
    """
    for name in available_profiles():
        profile = load_profile(name)
        for qid, q in profile.questions.items():
            if q["type"] != "noul":
                continue
            keys = set(q["criteria"].keys())
            assert keys == {"true", "false"}, (
                f"{name}/{qid}: criteria keys are {keys!r}, expected string "
                f"keys {{'true', 'false'}} (unquoted YAML true:/false: parse as booleans)"
            )
            assert all(isinstance(k, str) for k in keys), (
                f"{name}/{qid}: criteria keys must be strings, got {[type(k) for k in keys]}"
            )


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
            options = render_options(internal)
            expected = len(options)
            assert len(markers) == expected, (
                f"{name}/{qid}: {expected - len(markers)} option(s) dropped; shorten the criteria"
            )
            if q["type"] == "noul":
                # A false assurance in the original guard: it only checked option
                # count and token budget, and Laya's generic true/false fallback
                # text is *shorter* than hand-written criteria -- so silently
                # falling back to it made these assertions MORE likely to pass,
                # not less. Assert the profile's own wording actually made it
                # into what Laya renders, not a fallback substituted because the
                # criteria keys weren't the strings "true"/"false".
                rendered_text = " ".join(str(o) for o in options)
                for value in crit.values():
                    assert value in rendered_text, (
                        f"{name}/{qid}: rendered options do not contain this profile's "
                        f"own criteria text {value!r}; Laya likely substituted its "
                        f"generic true/false fallback because the criteria keys were "
                        f"not the strings \"true\"/\"false\"."
                    )
            head = tok(f"{q['type']} question: {q['instructions']}",
                       add_special_tokens=False)["input_ids"]
            opt_ids = [[0] + tok(" " + o, add_special_tokens=False)["input_ids"][:48]
                       for o in options]
            budget = 192 - sum(len(o) for o in opt_ids)
            assert len(head) <= max(8, budget), (
                f"{name}/{qid}: instructions truncated by {len(head) - max(8, budget)} tokens"
            )
