# Laya Claude Clipper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the clipper pipeline with Laya as the local scoring engine, replacing TypeSafe Jev entirely.

**Architecture:** Every stage is a CLI subcommand reading and writing JSON artifacts in a run directory. Scoring loads a local Laya checkpoint once per run and calls `agent.system_one(state, questions)` once per window — the whole question bundle in a single batched forward pass. No network, no API key, no retry machinery. Device resolution is isolated in its own module because Laya cannot auto-detect Intel Arc.

**Tech Stack:** Python 3.11+, `laya` 0.3.5 (torch + transformers + safetensors + huggingface_hub), WhisperX, PyYAML, ffmpeg via subprocess, pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-09-22-laya-claude-clipper-design.md`

## Global Constraints

- Python `>=3.11`.
- Scoring requires **no credential of any kind**. `TYPESAFE_API_KEY` is removed.
- `pytest` must pass offline, with no network and without downloading a model. The `model` marker is deselected by default and replaces the previous `live` marker.
- Every artifact is JSON on disk. Nothing passes between stages in memory.
- Score primitives normalize as `score / (k - 1)`; `noul` is used as-is; `choice` is never weighted.
- `uncertain_confidence` is a **per-primitive map**, never a scalar.
- Provenance (`laya_model`) is written in `scores.json` and carried verbatim into `candidates.json` and `plan.json`.
- Default repo is `convaiinnovations/laya`, overridable by `CLIPPER_LAYA_REPO`.
- Device is resolved by `clipper/device.py` and passed **explicitly** to `laya.load`. Laya's own auto-detect is never relied upon.
- Commit after every task. Never commit `runs/` or `.env`.

## Status and Provenance

Tasks 1–6 are **complete and review-clean** on branch `feat/clipper-pipeline` at commit `2c1cfa9`: project scaffold and ffmpeg preflight, run directory and artifact layer, rolling-baseline energy normalization, ingest, transcribe, windowing. Suite: 67 passing.

This plan renumbers the remaining work because the old Task 8 ("Jev question building and async scoring client") splits into three under Laya. Mapping to the superseded plan:

| This plan | Old plan | Status |
|---|---|---|
| Task 7 | Task 7 | Rewritten — `uncertain_confidence` map, structural guard, dependency swap |
| Task 8 | — | **New** — device resolution |
| Task 9 | Task 8 (first half) | Rewritten — question building, normalization, fake-agent scoring |
| Task 10 | Task 8 (second half) | Rewritten — agent loading, checkpoint selection, provenance |
| Task 11 | Task 9 | Rewritten — composite over normalized values, per-type uncertainty |
| Task 12 | Task 10 | Carried forward unchanged |
| Task 13 | Task 11 | Rewritten — synchronous, `laya_model` provenance |
| Tasks 14–20 | Tasks 12–18 | Carried forward; three need small patches (§Tasks 14–20) |

### Untrusted partial output

Task 7 was killed mid-write in the previous session. These three files are **untracked, incomplete, and validated by no test**:

```
clipper/profiles/__init__.py   (empty)
clipper/profiles/core.yaml     (partial — has a SCALAR uncertain_confidence, which this plan replaces)
clipper/profiles/podcast.yaml  (partial)
```

Treat them as untrusted. Verify every line against Task 7 or overwrite them outright. Do not assume they are correct. Missing entirely: `loader.py`, `talking_head.yaml`, `lecture.yaml`, `stream.yaml`, `tests/test_profiles.py`.

## File Structure

| File | Responsibility |
|---|---|
| `clipper/profiles/loader.py` | YAML load, `extends` resolution, validation, `Profile` dataclass |
| `clipper/profiles/*.yaml` | Question bundles, weights, gates, thresholds per content type |
| `clipper/device.py` | Device resolution. The seam where Intel Arc optimization lands |
| `clipper/score.py` | Question building, state construction, answer normalization, per-window scoring |
| `clipper/agent.py` | Laya checkpoint selection, agent loading, provenance assembly |
| `clipper/rank.py` | Composite score, gates, per-type uncertainty routing |
| `clipper/merge.py` | Candidate merging and reaction-lag backward extension |
| `clipper/stages/score.py` | Stage wiring: reads artifacts, writes `scores.json` + `candidates.json` |

`score.py` and `agent.py` are separate because agent loading touches torch and the filesystem while scoring is pure over a passed-in agent. That split is what lets the entire default test suite run with a fake agent and no model.

---

### Task 7: Profile schema, loader, and the five profile files

**Files:**
- Create: `clipper/profiles/loader.py`, `clipper/profiles/talking_head.yaml`, `clipper/profiles/lecture.yaml`, `clipper/profiles/stream.yaml`
- Overwrite: `clipper/profiles/__init__.py`, `clipper/profiles/core.yaml`, `clipper/profiles/podcast.yaml`
- Modify: `pyproject.toml`, `.env.example`
- Test: `tests/test_profiles.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Profile` frozen dataclass with `name: str`, `questions: dict[str, dict]`, `weights: dict[str, float]`, `penalties: dict[str, float]`, `gates: dict[str, float]`, `thresholds: dict[str, float]`, `uncertain_confidence: dict[str, float]`, `merge: dict[str, float]`, `window: dict[str, float]`. Loader: `load_profile(name: str) -> Profile`, `available_profiles() -> list[str]`. Raises `ProfileError`.

Question entries are `{"type": "score"|"choice"|"noul", "instructions": str, "criteria": ...}` — a list for `score`, an option→description dict for `choice`, a `{"true","false"}` dict for `noul`. This is Laya's own schema; no translation layer exists or should be added.

`uncertain_confidence` is separated out of `thresholds` into its own field because it is a per-primitive map while everything else in `thresholds` is a scalar. Defaults when absent: `{"score": 0.10, "choice": 0.15, "noul": 0.55}`.

- [ ] **Step 1: Swap the dependency and the pytest marker**

In `pyproject.toml`, replace `"typesafe-sdk"` with `"laya"` in `dependencies`, and replace the marker line:

```toml
[tool.pytest.ini_options]
markers = ["model: loads the real Laya checkpoint; deselected by default"]
addopts = "-m 'not model'"
asyncio_mode = "auto"
```

In `.env.example`, delete the `TYPESAFE_API_KEY=sk-...` line and add:

```
CLIPPER_LAYA_DEVICE=
CLIPPER_LAYA_REPO=
```

- [ ] **Step 2: Write the failing test**

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_profiles.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'clipper.profiles.loader'`

- [ ] **Step 4: Write `clipper/profiles/core.yaml`**

Overwrite the untracked partial. Keep every question exactly as listed — they are verified to fit Laya's 192-token question head. The only change from the partial file is the `thresholds` block at the end.

Copy the `questions:` block verbatim from the existing partial `clipper/profiles/core.yaml` (all fifteen: `clipworthy`, `hook_strength`, `hook_type`, `open_loop`, `self_contained`, `needs_context`, `ends_cleanly`, `payoff`, `poster_line`, `concrete_specifics`, `audible_reaction`, `emotional_intensity`, `clip_format`, `opens_with_windup`, `buried_lede`), then replace everything from `weights:` onward with:

```yaml
weights:
  clipworthy: 0.45
  hook_strength: 0.30
  poster_line: 0.08
  open_loop: 0.06
  payoff: 0.05
  concrete_specifics: 0.03
  audible_reaction: 0.03
penalties:
  needs_context: 0.20
  opens_with_windup: 0.15
gates:
  self_contained: 0.35
thresholds:
  candidate: 0.55
  uncertain_confidence:
    score: 0.10
    choice: 0.15
    noul: 0.55
merge:
  max_seconds: 60.0
  backward_extend_seconds: 8.0
  backward_extend_offset: 0.6
window:
  seconds: 30.0
  step: 10.0
  context: 20.0
```

Note the `window:` block moves to the end for readability; key order is irrelevant to YAML.

- [ ] **Step 5: Write the four derived profiles**

`clipper/profiles/podcast.yaml` — overwrite the partial, which is already correct; verify it matches:

```yaml
name: podcast
extends: core
questions:
  disagreement:
    type: noul
    instructions: Two speakers genuinely disagree, rather than politely agreeing.
    criteria:
      true: "A real difference of view is voiced."
      false: "Agreement, or one speaker only."
  personal_story:
    type: noul
    instructions: A speaker recounts something that happened to them personally.
    criteria:
      true: "A first-hand anecdote with specifics."
      false: "Abstract discussion."
  needs_speaker_id:
    type: noul
    instructions: >
      Following this segment requires knowing who is speaking, so an on-screen
      speaker label is needed.
    criteria:
      true: "Two or more voices alternate and identity matters."
      false: "A single voice, or identity is irrelevant."
weights:
  disagreement: 0.06
  personal_story: 0.05
```

`clipper/profiles/talking_head.yaml`:

```yaml
name: talking_head
extends: core
questions:
  hot_take:
    type: noul
    instructions: The speaker states a bold opinion they expect some viewers to reject.
    criteria:
      true: "A clear, contestable claim is made."
      false: "Uncontroversial or purely descriptive."
  direct_address:
    type: noul
    instructions: The speaker addresses the viewer directly rather than a third party.
    criteria:
      true: "Speaks to 'you' as the audience."
      false: "Talks about a subject without addressing anyone."
weights:
  hot_take: 0.07
  direct_address: 0.04
```

`clipper/profiles/lecture.yaml`:

```yaml
name: lecture
extends: core
questions:
  complete_explanation:
    type: noul
    instructions: The segment explains its concept from start to finish without leaving a gap.
    criteria:
      true: "A viewer could act on this without the rest of the lecture."
      false: "The explanation depends on material outside the segment."
  one_concept:
    type: noul
    instructions: The segment carries exactly one idea rather than several partial ones.
    criteria:
      true: "A single concept, fully carried."
      false: "Two or more ideas, none finished."
  actionable:
    type: noul
    instructions: The segment gives something the viewer can do, not only something to know.
    criteria:
      true: "Contains a method, step, or applicable rule."
      false: "Purely conceptual."
  requires_visual:
    type: noul
    instructions: >
      Understanding this segment requires seeing slides, diagrams, or written
      material that a clip would not carry.
    criteria:
      true: "Refers to something on screen that must be seen."
      false: "Fully carried by the audio."
weights:
  complete_explanation: 0.07
  one_concept: 0.06
  actionable: 0.04
penalties:
  requires_visual: 0.25
```

`clipper/profiles/stream.yaml`:

```yaml
name: stream
extends: core
questions:
  reaction_spike:
    type: noul
    instructions: The streamer reacts sharply — surprise, outrage, delight, or disbelief.
    criteria:
      true: "A strong, sudden reaction occurs."
      false: "Even delivery throughout."
  narratable:
    type: noul
    instructions: >
      What happens is followable from speech alone, without watching the
      gameplay.
    criteria:
      true: "The words carry the event."
      false: "Only makes sense if you can see the screen."
  needs_gameplay_context:
    type: noul
    instructions: >
      The moment is incomprehensible without knowing the prior game state.
    criteria:
      true: "Depends on setup the viewer has not seen."
      false: "Self-explanatory."
weights:
  reaction_spike: 0.08
  narratable: 0.05
penalties:
  needs_gameplay_context: 0.20
```

- [ ] **Step 6: Write `clipper/profiles/__init__.py`**

Overwrite the empty partial:

```python
from clipper.profiles.loader import Profile, ProfileError, available_profiles, load_profile

__all__ = ["Profile", "ProfileError", "available_profiles", "load_profile"]
```

- [ ] **Step 7: Implement `clipper/profiles/loader.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROFILE_DIR = Path(__file__).parent

VALID_TYPES = ("score", "choice", "noul")
DEFAULT_UNCERTAIN = {"score": 0.10, "choice": 0.15, "noul": 0.55}


class ProfileError(ValueError):
    """A profile file is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class Profile:
    name: str
    questions: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    penalties: dict = field(default_factory=dict)
    gates: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    uncertain_confidence: dict = field(default_factory=dict)
    merge: dict = field(default_factory=dict)
    window: dict = field(default_factory=dict)


def available_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILE_DIR.glob("*.yaml"))


def _read(name: str) -> dict:
    path = PROFILE_DIR / f"{name}.yaml"
    if not path.exists():
        raise ProfileError(
            f"No profile named {name!r}. Available: {', '.join(available_profiles())}."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ProfileError(f"Profile {name!r} is not a YAML mapping.")
    return data


def _merge_layer(base: dict, layer: dict) -> dict:
    """Child layers extend parent maps key-by-key; scalars replace outright."""
    out = dict(base)
    for key, value in layer.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


def _resolve(name: str, seen: tuple[str, ...] = ()) -> dict:
    if name in seen:
        raise ProfileError(f"Profile {name!r} extends itself: {' -> '.join(seen + (name,))}.")
    data = _read(name)
    parent = data.pop("extends", None)
    if parent is None:
        return data
    return _merge_layer(_resolve(parent, seen + (name,)), data)


def _validate(name: str, data: dict) -> None:
    questions = data.get("questions") or {}
    for qid, q in questions.items():
        qtype = q.get("type")
        if qtype not in VALID_TYPES:
            raise ProfileError(
                f"Profile {name!r} question {qid!r} has type {qtype!r}; "
                f"Laya supports only {', '.join(VALID_TYPES)}."
            )
        if not q.get("instructions"):
            raise ProfileError(f"Profile {name!r} question {qid!r} has no instructions.")
        crit = q.get("criteria")
        if qtype == "score" and not isinstance(crit, list):
            raise ProfileError(f"Profile {name!r} question {qid!r} is a score; criteria must be a list.")
        if qtype == "score" and len(crit) < 2:
            raise ProfileError(f"Profile {name!r} question {qid!r} needs at least 2 score levels.")
        if qtype == "choice" and not isinstance(crit, dict):
            raise ProfileError(f"Profile {name!r} question {qid!r} is a choice; criteria must be a mapping.")

    for block in ("weights", "penalties", "gates"):
        for key in (data.get(block) or {}):
            if key not in questions:
                raise ProfileError(
                    f"Profile {name!r} {block} references {key!r}, which is not a question in this profile."
                )


def load_profile(name: str) -> Profile:
    data = _resolve(name)
    _validate(name, data)

    thresholds = dict(data.get("thresholds") or {})
    uncertain = thresholds.pop("uncertain_confidence", None)
    if uncertain is None:
        uncertain = dict(DEFAULT_UNCERTAIN)
    elif not isinstance(uncertain, dict):
        raise ProfileError(
            f"Profile {name!r}: uncertain_confidence must be a per-primitive map "
            f"({{score, choice, noul}}), not a single value. Laya's confidence uses a "
            f"different formula per primitive, so one threshold cannot serve all three."
        )
    else:
        uncertain = {**DEFAULT_UNCERTAIN, **uncertain}

    return Profile(
        name=data.get("name", name),
        questions=data.get("questions") or {},
        weights=data.get("weights") or {},
        penalties=data.get("penalties") or {},
        gates=data.get("gates") or {},
        thresholds=thresholds,
        uncertain_confidence=uncertain,
        merge=data.get("merge") or {},
        window=data.get("window") or {},
    )
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_profiles.py -v`
Expected: 9 passed

- [ ] **Step 9: Add the structural guard test**

This is the test that keeps a future profile edit from silently degrading a question. It is marked `model` because it needs the real tokenizer.

Append to `tests/test_profiles.py`:

```python
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
```

- [ ] **Step 10: Run the structural guard**

Run: `pytest tests/test_profiles.py -v -m model`
Expected: 1 passed. If it fails, shorten the named question's criteria — do not raise `head_max_len`.

- [ ] **Step 11: Run the whole suite**

Run: `pytest -q`
Expected: 76 passed (67 existing + 9 new), `model` deselected.

- [ ] **Step 12: Commit**

```bash
git add clipper/profiles tests/test_profiles.py pyproject.toml .env.example
git commit -m "feat: profile loader with per-primitive uncertainty thresholds

Replaces typesafe-sdk with laya. uncertain_confidence is a per-primitive
map because Laya computes confidence differently per type; a scalar is
rejected with an explanatory error. Structural guard asserts no question
loses options or instruction tokens to Laya's 192-token question head."
```

---

### Task 8: Device resolution

**Files:**
- Create: `clipper/device.py`
- Test: `tests/test_device.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `resolve_device(requested: str | None = None, probes: dict[str, Callable[[], bool]] | None = None) -> str` returning one of `"cuda" | "xpu" | "mps" | "cpu"`. `DeviceError` for an unrecognized name. `PROBE_ORDER: tuple[str, ...]`. `default_probes() -> dict[str, Callable[[], bool]]`.

This module exists because **Laya's own probe is `cuda -> mps -> cpu` and cannot see an Intel Arc GPU**. Calling `laya.load()` without a device on an Arc machine silently selects CPU, which measured 5.9x slower. It is also the seam where separate Intel Arc optimization work lands; nothing in `score.py` should change when it does.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from clipper.device import PROBE_ORDER, DeviceError, resolve_device


def probes(**kw):
    """Build a probe table; anything unnamed is unavailable."""
    return {name: (lambda v=kw.get(name, False): v) for name in ("cuda", "xpu", "mps")}


def test_auto_prefers_cuda_when_present():
    assert resolve_device("auto", probes(cuda=True, xpu=True)) == "cuda"


def test_auto_picks_xpu_when_cuda_absent():
    """The case that matters on this machine: Intel Arc, no CUDA."""
    assert resolve_device("auto", probes(xpu=True)) == "xpu"


def test_auto_picks_mps_when_only_mps():
    assert resolve_device("auto", probes(mps=True)) == "mps"


def test_auto_falls_back_to_cpu_when_nothing_is_available():
    assert resolve_device("auto", probes()) == "cpu"


def test_none_means_auto():
    assert resolve_device(None, probes(xpu=True)) == "xpu"


def test_env_var_is_read_when_nothing_is_passed(monkeypatch):
    monkeypatch.setenv("CLIPPER_LAYA_DEVICE", "cpu")
    assert resolve_device(None, probes(xpu=True)) == "cpu"


def test_explicit_device_is_honoured_when_available():
    assert resolve_device("xpu", probes(xpu=True)) == "xpu"


def test_explicit_unavailable_device_falls_back_and_warns():
    with pytest.warns(RuntimeWarning, match="xpu"):
        assert resolve_device("xpu", probes()) == "cpu"


def test_explicit_cpu_never_warns(recwarn):
    assert resolve_device("cpu", probes()) == "cpu"
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_unknown_device_name_is_rejected_and_lists_valid_ones():
    with pytest.raises(DeviceError, match="xpu"):
        resolve_device("gpu0", probes())


def test_case_and_whitespace_are_tolerated():
    assert resolve_device("  XPU ", probes(xpu=True)) == "xpu"


def test_probe_order_puts_xpu_ahead_of_cpu():
    assert PROBE_ORDER.index("xpu") < PROBE_ORDER.index("cpu")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_device.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'clipper.device'`

- [ ] **Step 3: Implement `clipper/device.py`**

```python
from __future__ import annotations

import os
import warnings
from typing import Callable

PROBE_ORDER: tuple[str, ...] = ("cuda", "xpu", "mps", "cpu")
VALID = PROBE_ORDER + ("auto",)


class DeviceError(ValueError):
    """An unrecognized device name."""


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _xpu() -> bool:
    """Intel GPUs (Arc). Laya's own probe does not look here."""
    try:
        import torch

        return bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    except Exception:
        return False


def _mps() -> bool:
    try:
        import torch

        return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    except Exception:
        return False


def default_probes() -> dict[str, Callable[[], bool]]:
    return {"cuda": _cuda, "xpu": _xpu, "mps": _mps}


def resolve_device(requested: str | None = None,
                   probes: dict[str, Callable[[], bool]] | None = None) -> str:
    """Resolve a device name to pass explicitly to `laya.load`.

    Never rely on Laya's internal auto-detection: it probes cuda -> mps -> cpu
    and cannot see an Intel Arc GPU, so it silently returns CPU on this machine.
    """
    probes = probes if probes is not None else default_probes()

    if requested is None:
        requested = os.environ.get("CLIPPER_LAYA_DEVICE") or "auto"
    requested = requested.strip().lower() or "auto"

    if requested not in VALID:
        raise DeviceError(
            f"Unknown device {requested!r}. Valid values: {', '.join(VALID)}."
        )

    if requested == "auto":
        for name in PROBE_ORDER:
            if name == "cpu" or probes.get(name, lambda: False)():
                return name
        return "cpu"

    if requested == "cpu":
        return "cpu"

    if probes.get(requested, lambda: False)():
        return requested

    warnings.warn(
        f"Device {requested!r} was requested but is not available; falling back to CPU. "
        f"Scoring on CPU measured ~5.9x slower (about 78 minutes for a 90-minute "
        f"source, against 13 on an Intel Arc GPU).",
        RuntimeWarning,
        stacklevel=2,
    )
    return "cpu"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_device.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add clipper/device.py tests/test_device.py
git commit -m "feat: XPU-aware device resolution

Laya probes cuda -> mps -> cpu and cannot see Intel Arc, so laya.load()
with no device silently picks CPU. Resolution lives here and is passed
explicitly. This module is the seam for Arc optimization work."
```

---

### Task 9: Question building, state, and answer normalization

**Files:**
- Create: `clipper/score.py`
- Test: `tests/test_score.py`

**Interfaces:**
- Consumes: `Profile` (Task 7), window dicts (Task 6).
- Produces: `build_questions(profile: Profile) -> dict[str, dict]`, `window_state(window: dict, profile_name: str) -> dict`, `normalize_answer(answer: dict, qdef: dict) -> float | None`, `answers_to_dict(response: dict, profile: Profile) -> dict[str, dict]`, `score_windows(windows: list[dict], profile: Profile, agent) -> list[dict]`.

Score record: `{"id": int, "answers": {...}, "failed": bool}`. Each answer is `{"type", "value", "normalized", "confidence", "probabilities", "legend"}`. `value` is the raw Laya return — the float or the chosen option string. `normalized` is the 0..1 value the composite consumes, or `None` for a `choice`.

`agent` is any object with `system_one(state, questions) -> dict`. The entire default test suite uses a fake; the real Laya agent arrives from Task 10.

**Why `value` and `normalized` are both kept:** spec §13 requires `scores.json` to retain every raw answer so a weighting change is a recomputation rather than a re-run. The composite needs one scale. Storing both satisfies each.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from clipper.profiles.loader import load_profile
from clipper.score import (answers_to_dict, build_questions, normalize_answer,
                           score_windows, window_state)


class FakeAgent:
    """Stands in for laya.Agent. Records calls; returns canned answers."""

    def __init__(self, answers=None, fail_on=None):
        self.calls = []
        self._answers = answers or {}
        self._fail_on = fail_on or set()

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        if len(self.calls) - 1 in self._fail_on:
            raise RuntimeError("simulated inference failure")
        answers = {}
        for qid, q in questions.items():
            if qid in self._answers:
                answers[qid] = self._answers[qid]
            elif q["type"] == "score":
                answers[qid] = {"type": "score", "score": 2.0, "confidence": 0.15,
                                "legend": {"0": "a"}, "probabilities": {"0": 0.2}}
            elif q["type"] == "choice":
                key = next(iter(q["criteria"]))
                answers[qid] = {"type": "choice", "choice": key, "confidence": 0.2,
                                "probabilities": {key: 0.5}}
            else:
                answers[qid] = {"type": "noul", "noul": 0.7, "confidence": 0.7,
                                "probabilities": {}}
        return {"model": "laya-rl-agent", "answers": answers,
                "usage": {"input_tokens": 226, "output_tokens": 0}}


def win(i=0, start=0.0, end=30.0, text="hello world"):
    return {"id": i, "start": start, "end": end, "text": text, "preceding": "before",
            "position": 0.4, "energy_mean": 0.44, "energy_peak": 0.86,
            "energy_peak_offset": 0.71}


def test_build_questions_passes_profile_yaml_through_unchanged():
    p = load_profile("core")
    qs = build_questions(p)
    assert qs["clipworthy"]["type"] == "score"
    assert qs["clipworthy"]["instructions"] == p.questions["clipworthy"]["instructions"]
    assert qs["clipworthy"]["criteria"] == p.questions["clipworthy"]["criteria"]
    assert set(qs) == set(p.questions)


def test_window_state_puts_window_text_first():
    """Laya truncates state from the right, so the judged text must lead."""
    keys = list(window_state(win(), "podcast").keys())
    assert keys.index("window") < keys.index("preceding")
    assert keys[-1] == "preceding"


def test_window_state_carries_energy_and_position():
    s = window_state(win(), "podcast")
    assert s["profile"] == "podcast"
    assert s["window"]["start"] == 0.0
    assert s["energy_peak_offset"] == 0.71


def test_normalize_score_divides_by_levels_minus_one():
    qdef = {"type": "score", "criteria": ["a", "b", "c", "d", "e"]}
    assert normalize_answer({"type": "score", "score": 4.0}, qdef) == 1.0
    assert normalize_answer({"type": "score", "score": 0.0}, qdef) == 0.0
    assert normalize_answer({"type": "score", "score": 2.0}, qdef) == 0.5


def test_normalize_score_with_three_levels():
    qdef = {"type": "score", "criteria": ["a", "b", "c"]}
    assert normalize_answer({"type": "score", "score": 1.0}, qdef) == 0.5


def test_normalize_noul_passes_probability_through():
    assert normalize_answer({"type": "noul", "noul": 0.73}, {"type": "noul"}) == 0.73


def test_normalize_choice_is_none_because_choice_is_never_weighted():
    qdef = {"type": "choice", "criteria": {"a": "x", "b": "y"}}
    assert normalize_answer({"type": "choice", "choice": "a"}, qdef) is None


def test_normalize_score_is_clamped_to_unit_range():
    qdef = {"type": "score", "criteria": ["a", "b"]}
    assert normalize_answer({"type": "score", "score": 1.4}, qdef) == 1.0
    assert normalize_answer({"type": "score", "score": -0.2}, qdef) == 0.0


def test_answers_to_dict_keeps_raw_value_and_normalized():
    p = load_profile("core")
    resp = FakeAgent().system_one({}, build_questions(p))
    out = answers_to_dict(resp, p)
    assert out["clipworthy"]["value"] == 2.0          # raw, as Laya returned it
    assert out["clipworthy"]["normalized"] == 0.5     # rebased for the composite
    assert out["open_loop"]["value"] == 0.7
    assert out["open_loop"]["normalized"] == 0.7
    assert out["hook_type"]["normalized"] is None


def test_answers_to_dict_carries_confidence_per_answer():
    p = load_profile("core")
    out = answers_to_dict(FakeAgent().system_one({}, build_questions(p)), p)
    assert out["clipworthy"]["confidence"] == 0.15
    assert out["open_loop"]["confidence"] == 0.7


def test_score_windows_calls_the_agent_once_per_window():
    p = load_profile("core")
    agent = FakeAgent()
    records = score_windows([win(0), win(1), win(2)], p, agent)
    assert len(agent.calls) == 3
    assert [r["id"] for r in records] == [0, 1, 2]
    assert all(r["failed"] is False for r in records)


def test_score_windows_sends_the_whole_bundle_in_one_call():
    """One call per window, not one per question."""
    p = load_profile("core")
    agent = FakeAgent()
    score_windows([win(0)], p, agent)
    _, questions = agent.calls[0]
    assert len(questions) == len(p.questions)


def test_a_failing_window_is_marked_and_the_run_continues():
    p = load_profile("core")
    agent = FakeAgent(fail_on={1})
    records = score_windows([win(0), win(1), win(2)], p, agent)
    assert [r["failed"] for r in records] == [False, True, False]
    assert records[1]["answers"] == {}
    assert "simulated inference failure" in records[1]["error"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_score.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'clipper.score'`

- [ ] **Step 3: Implement `clipper/score.py`**

```python
from __future__ import annotations

from clipper.profiles.loader import Profile


def build_questions(profile: Profile) -> dict[str, dict]:
    """Profile YAML is already Laya's schema, so this is a pass-through copy.

    Kept as a named seam so a future profile feature has one place to land.
    """
    return {qid: dict(q) for qid, q in profile.questions.items()}


def window_state(window: dict, profile_name: str) -> dict:
    """Build the text-only state Laya scores.

    Field order is load-bearing: Laya truncates state from the RIGHT at roughly
    395 tokens, so the window being judged leads and `preceding` trails, ensuring
    any overflow sheds context rather than the material itself.
    """
    return {
        "profile": profile_name,
        "window": {
            "start": window["start"],
            "end": window["end"],
            "text": window["text"],
        },
        "position": window.get("position"),
        "energy_mean": window.get("energy_mean"),
        "energy_peak": window.get("energy_peak"),
        "energy_peak_offset": window.get("energy_peak_offset"),
        "preceding": window.get("preceding", ""),
    }


def normalize_answer(answer: dict, qdef: dict) -> float | None:
    """Rebase one Laya answer onto 0..1.

    `score` returns an expected value over level INDICES (0..k-1), not Jev's
    2..10, so it divides by k-1. `noul` is already a probability. `choice` is
    categorical and never weighted, so it has no normalized form.
    """
    qtype = qdef.get("type")
    if qtype == "score":
        levels = len(qdef.get("criteria") or [])
        if levels < 2:
            return None
        raw = float(answer.get("score", 0.0)) / (levels - 1)
        return min(1.0, max(0.0, raw))
    if qtype == "noul":
        return min(1.0, max(0.0, float(answer.get("noul", 0.0))))
    return None


def answers_to_dict(response: dict, profile: Profile) -> dict[str, dict]:
    """Flatten a `system_one` response, keeping raw values alongside normalized ones."""
    out: dict[str, dict] = {}
    for qid, answer in (response.get("answers") or {}).items():
        qdef = profile.questions.get(qid, {})
        qtype = answer.get("type")
        if qtype == "choice":
            value = answer.get("choice")
        elif qtype == "score":
            value = answer.get("score")
        else:
            value = answer.get("noul")
        out[qid] = {
            "type": qtype,
            "value": value,
            "normalized": normalize_answer(answer, qdef),
            "confidence": answer.get("confidence"),
            "probabilities": answer.get("probabilities", {}),
            "legend": answer.get("legend"),
        }
    return out


def score_windows(windows: list[dict], profile: Profile, agent) -> list[dict]:
    """Score every window with one `system_one` call each.

    `agent` is anything exposing `system_one(state, questions)`. Scoring is
    synchronous: Laya is local and compute-bound, so there is nothing to overlap.
    A window whose call raises is marked `failed` and the run continues — losing
    3 of 540 windows is not a reason to discard an eight-minute transcription.
    """
    questions = build_questions(profile)
    records: list[dict] = []
    for window in windows:
        state = window_state(window, profile.name)
        try:
            response = agent.system_one(state, questions)
        except Exception as exc:  # noqa: BLE001 - any inference failure is per-window
            records.append({"id": window["id"], "answers": {}, "failed": True,
                            "error": f"{type(exc).__name__}: {exc}"})
            continue
        records.append({"id": window["id"],
                        "answers": answers_to_dict(response, profile),
                        "failed": False})
    return records
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_score.py -v`
Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add clipper/score.py tests/test_score.py
git commit -m "feat: Laya question building, state, and answer normalization

One system_one call per window carries the whole bundle. Score primitives
rebase through score/(k-1) because Laya returns 0..k-1, not Jev's 2..10;
raw values are retained alongside normalized ones so a weighting change
stays a recomputation."
```

---

### Task 10: Agent loading, checkpoint selection, and provenance

**Files:**
- Create: `clipper/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `resolve_device` (Task 8), `Profile` (Task 7).
- Produces: `checkpoint_for_language(language: str | None) -> str | None` (returns `None` for the repo root, `"multilingual"` otherwise), `used_buckets(profile: Profile) -> set[str]`, `uncalibrated_buckets(agent, profile: Profile) -> list[str]`, `load_agent(language, profile, device=None, repo=None, loader=None) -> tuple[object, dict]` returning the agent and the `laya_model` provenance dict, `LAYA_REPO_DEFAULT: str`, `AgentError`.

`loader` defaults to `laya.load` and is injected in tests so the default suite never downloads or loads a model.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from clipper.agent import (AgentError, checkpoint_for_language, load_agent,
                           uncalibrated_buckets, used_buckets)
from clipper.profiles.loader import load_profile


class FakeLoaded:
    """Stands in for a loaded laya.Agent."""

    def __init__(self, device="xpu", dtype="torch.bfloat16", temps=None):
        self.device = device
        self.dtype = dtype
        self.temperature_by_options_raw = temps if temps is not None else {"choice:11+": 0.1006}

    def system_one(self, state, questions):
        return {"model": "laya-rl-agent", "answers": {}, "usage": {}}


def fake_loader(recorder, **overrides):
    def _load(repo, device=None, subfolder=None, **kw):
        recorder.append({"repo": repo, "device": device, "subfolder": subfolder})
        return FakeLoaded(**overrides)
    return _load


def test_english_selects_the_repo_root():
    assert checkpoint_for_language("en") is None


def test_non_english_selects_multilingual():
    assert checkpoint_for_language("hi") == "multilingual"
    assert checkpoint_for_language("de") == "multilingual"


def test_missing_language_selects_multilingual_defensively():
    assert checkpoint_for_language(None) == "multilingual"


def test_language_region_suffix_still_counts_as_english():
    assert checkpoint_for_language("en-US") is None


def test_load_agent_passes_device_explicitly():
    """Laya's own probe cannot see Arc, so the device must be passed, not inferred."""
    calls = []
    _, meta = load_agent("en", load_profile("core"), device="xpu",
                         loader=fake_loader(calls))
    assert calls[0]["device"] == "xpu"
    assert calls[0]["subfolder"] is None
    assert meta["device"] == "xpu"


def test_load_agent_selects_multilingual_subfolder_for_non_english():
    calls = []
    _, meta = load_agent("fr", load_profile("core"), device="cpu",
                         loader=fake_loader(calls))
    assert calls[0]["subfolder"] == "multilingual"
    assert meta["checkpoint"] == "multilingual"


def test_provenance_records_repo_package_and_dtype():
    _, meta = load_agent("en", load_profile("core"), device="xpu",
                         loader=fake_loader([]))
    assert meta["repo"] == "convaiinnovations/laya"
    assert meta["checkpoint"] == "root"
    assert meta["dtype"] == "torch.bfloat16"
    assert meta["package"]


def test_silent_cpu_fallback_is_detected_and_warned():
    """Laya mutates agent.device and continues on placement failure."""
    loader = fake_loader([], device="cpu")
    with pytest.warns(RuntimeWarning, match="fell back"):
        _, meta = load_agent("en", load_profile("core"), device="xpu", loader=loader)
    assert meta["device"] == "cpu"
    assert meta["device_requested"] == "xpu"


def test_no_warning_when_the_device_is_what_was_asked_for(recwarn):
    load_agent("en", load_profile("core"), device="xpu", loader=fake_loader([]))
    assert not [w for w in recwarn if "fell back" in str(w.message)]


def test_used_buckets_reflects_the_profiles_actual_questions():
    buckets = used_buckets(load_profile("core"))
    assert "score:3-5" in buckets       # 5-level scores
    assert "choice:6-10" in buckets     # hook_type (6), clip_format (8)
    assert "noul:2" in buckets
    assert "choice:11+" not in buckets  # nothing has 11+ options


def test_the_clamped_bucket_is_not_reachable_so_calibration_holds():
    """choice:11+ ships clamped, but no question reaches it."""
    agent = FakeLoaded(temps={"choice:11+": 0.1006})
    assert uncalibrated_buckets(agent, load_profile("core")) == []


def test_a_clamped_bucket_that_is_reachable_is_reported():
    agent = FakeLoaded(temps={"choice:6-10": 0.1006})
    assert uncalibrated_buckets(agent, load_profile("core")) == ["choice:6-10"]


def test_in_range_temperatures_are_not_flagged():
    agent = FakeLoaded(temps={"choice:6-10": 1.2, "noul:2": 0.9})
    assert uncalibrated_buckets(agent, load_profile("core")) == []


def test_calibration_flag_lands_in_provenance():
    _, meta = load_agent("en", load_profile("core"), device="cpu", loader=fake_loader([]))
    assert meta["confidence_calibrated"] is True
    assert meta["uncalibrated_buckets"] == []


def test_a_loader_failure_is_reported_with_the_repo_named():
    def boom(repo, **kw):
        raise OSError("no such file")
    with pytest.raises(AgentError, match="convaiinnovations/laya"):
        load_agent("en", load_profile("core"), device="cpu", loader=boom)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'clipper.agent'`

- [ ] **Step 3: Implement `clipper/agent.py`**

```python
from __future__ import annotations

import os
import warnings

from clipper.device import resolve_device
from clipper.profiles.loader import Profile

LAYA_REPO_DEFAULT = "convaiinnovations/laya"

# Laya clamps fitted temperatures into this range; outside it, confidence from
# that bucket is not calibrated and must not silently drive routing.
TEMP_MIN, TEMP_MAX = 0.5, 5.0


class AgentError(RuntimeError):
    """The Laya checkpoint could not be loaded."""


def checkpoint_for_language(language: str | None) -> str | None:
    """Repo root for English, the multilingual checkpoint otherwise.

    Returns the `subfolder` argument for `laya.load`: None means the repo root.
    """
    if not language:
        return "multilingual"
    return None if language.split("-")[0].lower() == "en" else "multilingual"


def _bucket(qtype: str, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return f"{qtype}:{size}"


def used_buckets(profile: Profile) -> set[str]:
    """The temperature buckets this profile's questions actually reach."""
    out: set[str] = set()
    for q in profile.questions.values():
        qtype = q.get("type")
        crit = q.get("criteria")
        if qtype == "score":
            k = len(crit or [])
        elif qtype == "choice":
            k = len(crit or {})
        else:
            k = 2
        out.add(_bucket(qtype, k))
    return out


def uncalibrated_buckets(agent, profile: Profile) -> list[str]:
    """Buckets this profile reaches whose shipped temperature was out of range.

    Evaluated against the buckets actually used, not against whether Laya warned
    at all: the shipped checkpoint clamps `choice:11+`, which no profile reaches,
    so a flag keyed on the warning would be permanently and uselessly false.
    """
    raw = getattr(agent, "temperature_by_options_raw", {}) or {}
    reachable = used_buckets(profile)
    bad = []
    for bucket, temp in raw.items():
        if bucket not in reachable:
            continue
        try:
            t = float(temp)
        except (TypeError, ValueError):
            continue
        if t < TEMP_MIN or t > TEMP_MAX:
            bad.append(bucket)
    return sorted(bad)


def _package_version() -> str:
    try:
        import laya

        return getattr(laya, "__version__", "unknown")
    except Exception:
        return "unknown"


def load_agent(language: str | None, profile: Profile, device: str | None = None,
               repo: str | None = None, loader=None) -> tuple[object, dict]:
    """Load one Laya agent for a run and assemble its provenance.

    The agent is loaded once and reused across every window; reloading is pure
    overhead (~8.5 s warm).
    """
    repo = repo or os.environ.get("CLIPPER_LAYA_REPO") or LAYA_REPO_DEFAULT
    requested = resolve_device(device)
    subfolder = checkpoint_for_language(language)

    if loader is None:
        try:
            import laya

            loader = laya.load
        except ImportError as exc:
            raise AgentError(
                "laya is not installed. Install it with: pip install laya"
            ) from exc

    with warnings.catch_warnings():
        # Laya's own temperature warning is re-derived below against the buckets
        # this profile actually reaches, so suppress the blanket one.
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            agent = loader(repo, device=requested, subfolder=subfolder)
        except Exception as exc:  # noqa: BLE001
            raise AgentError(
                f"Could not load Laya checkpoint {repo!r}"
                f"{f' (subfolder {subfolder!r})' if subfolder else ''}: {exc}"
            ) from exc

    actual = str(getattr(agent, "device", requested))
    if actual != requested:
        warnings.warn(
            f"Laya fell back from {requested!r} to {actual!r} while placing the model. "
            f"Scoring will be roughly 5.9x slower. Check the torch build supports "
            f"this device.",
            RuntimeWarning,
            stacklevel=2,
        )

    bad = uncalibrated_buckets(agent, profile)
    meta = {
        "package": _package_version(),
        "repo": repo,
        "checkpoint": subfolder or "root",
        "revision": _revision(repo),
        "device": actual,
        "device_requested": requested,
        "dtype": str(getattr(agent, "dtype", "unknown")),
        "confidence_calibrated": not bad,
        "uncalibrated_buckets": bad,
    }
    return agent, meta


def _revision(repo: str) -> str:
    """Best-effort commit sha of the cached checkpoint; 'unknown' if unavailable."""
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo, local_files_only=True,
                                 allow_patterns=["rl_agent_config.json"])
        return os.path.basename(path.rstrip("/\\"))
    except Exception:
        return "unknown"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent.py -v`
Expected: 15 passed

- [ ] **Step 5: Add a model-marked smoke test**

Append to `tests/test_agent.py`:

```python
@pytest.mark.model
def test_real_agent_loads_and_scores_one_window():
    from clipper.score import build_questions, score_windows

    profile = load_profile("core")
    agent, meta = load_agent("en", profile)
    assert meta["checkpoint"] == "root"
    assert meta["revision"] != "unknown"

    window = {"id": 0, "start": 0.0, "end": 30.0, "position": 0.1,
              "energy_mean": 0.4, "energy_peak": 0.8, "energy_peak_offset": 0.5,
              "preceding": "Earlier in the episode.",
              "text": "SPEAKER_01: The first version is supposed to embarrass you."}
    records = score_windows([window], profile, agent)
    assert records[0]["failed"] is False
    answers = records[0]["answers"]
    assert set(answers) == set(profile.questions)
    assert 0.0 <= answers["clipworthy"]["normalized"] <= 1.0
    assert 0.0 <= answers["open_loop"]["normalized"] <= 1.0
    assert answers["hook_type"]["value"] in profile.questions["hook_type"]["criteria"]
```

- [ ] **Step 6: Run the smoke test**

Run: `pytest tests/test_agent.py -v -m model`
Expected: 1 passed. Takes roughly 10 s for the load plus 1.5 s for the window on XPU.

- [ ] **Step 7: Commit**

```bash
git add clipper/agent.py tests/test_agent.py
git commit -m "feat: Laya agent loading, checkpoint selection, provenance

Checkpoint is chosen per run from the transcript language. Device is
passed explicitly and read back afterwards, because Laya silently falls
back to CPU on placement failure and that difference is 13 vs 78 minutes.
Calibration is evaluated against buckets the profile actually reaches."
```

---

### Task 11: Composite scoring, gates, and per-type uncertainty

**Files:**
- Create: `clipper/rank.py`
- Test: `tests/test_rank.py`

**Interfaces:**
- Consumes: score records (Task 9), `Profile` (Task 7).
- Produces: `composite(answers: dict, profile: Profile) -> float`, `passes_gate(answers: dict, profile: Profile) -> bool`, `is_uncertain(answers: dict, profile: Profile) -> bool`, `rank(scores: list[dict], profile: Profile) -> list[dict]` adding `composite`, `gated`, `uncertain` to each record.

Two things distinguish this from the superseded Jev version:

1. `composite` reads `answer["normalized"]`, never `answer["value"]`. Reading the raw value would silently feed a 0..4 score into weights calibrated for 0..1.
2. `is_uncertain` compares each weighted question's confidence against **its own primitive's threshold**. A single threshold cannot work: `score`/`choice` confidence is `1 - H(p)/log k` and runs 0.11–0.25 in practice, while `noul` confidence is `max(p, 1-p)` and is floored at 0.5. One number marks either everything or nothing uncertain.

Spec §7: `buried_lede` annotates but never penalizes — it describes a fixable in-point, not bad material.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from clipper.profiles.loader import Profile, load_profile
from clipper.rank import composite, is_uncertain, passes_gate, rank


def ans(normalized=None, confidence=1.0, qtype="noul", value=None):
    return {"type": qtype, "value": value, "normalized": normalized,
            "confidence": confidence, "probabilities": {}, "legend": None}


def tiny_profile(**kw):
    base = dict(
        name="tiny",
        questions={"a": {"type": "score", "criteria": ["0", "1", "2", "3", "4"],
                         "instructions": "x"},
                   "b": {"type": "noul", "instructions": "x"},
                   "g": {"type": "noul", "instructions": "x"},
                   "c": {"type": "choice", "criteria": {"x": "1", "y": "2"},
                         "instructions": "x"}},
        weights={"a": 0.6, "b": 0.4},
        penalties={},
        gates={},
        thresholds={"candidate": 0.55},
        uncertain_confidence={"score": 0.10, "choice": 0.15, "noul": 0.55},
        merge={}, window={},
    )
    base.update(kw)
    return Profile(**base)


def test_composite_uses_normalized_not_raw_values():
    """A raw 0..4 score must never reach weights calibrated for 0..1."""
    p = tiny_profile()
    answers = {"a": ans(normalized=1.0, value=4.0, qtype="score"),
               "b": ans(normalized=1.0)}
    assert composite(answers, p) == pytest.approx(1.0)


def test_composite_is_a_weighted_sum():
    p = tiny_profile()
    answers = {"a": ans(normalized=0.5, qtype="score"), "b": ans(normalized=0.25)}
    assert composite(answers, p) == pytest.approx(0.6 * 0.5 + 0.4 * 0.25)


def test_penalties_subtract():
    p = tiny_profile(penalties={"g": 0.2})
    answers = {"a": ans(normalized=1.0, qtype="score"), "b": ans(normalized=1.0),
               "g": ans(normalized=1.0)}
    assert composite(answers, p) == pytest.approx(0.8)


def test_composite_is_clamped_to_unit_range():
    p = tiny_profile(penalties={"g": 2.0})
    answers = {"a": ans(normalized=1.0, qtype="score"), "b": ans(normalized=1.0),
               "g": ans(normalized=1.0)}
    assert composite(answers, p) == 0.0


def test_a_missing_answer_contributes_zero_rather_than_raising():
    p = tiny_profile()
    assert composite({"a": ans(normalized=1.0, qtype="score")}, p) == pytest.approx(0.6)


def test_a_choice_answer_never_contributes_to_the_composite():
    p = tiny_profile(weights={"a": 0.6, "b": 0.4, "c": 0.5})
    answers = {"a": ans(normalized=0.0, qtype="score"), "b": ans(normalized=0.0),
               "c": ans(normalized=None, qtype="choice", value="x")}
    assert composite(answers, p) == 0.0


def test_gate_passes_at_or_above_the_floor():
    p = tiny_profile(gates={"g": 0.35})
    assert passes_gate({"g": ans(normalized=0.35)}, p) is True
    assert passes_gate({"g": ans(normalized=0.36)}, p) is True


def test_gate_fails_below_the_floor():
    p = tiny_profile(gates={"g": 0.35})
    assert passes_gate({"g": ans(normalized=0.34)}, p) is False


def test_a_missing_gate_answer_fails_closed():
    p = tiny_profile(gates={"g": 0.35})
    assert passes_gate({}, p) is False


def test_score_confidence_is_judged_against_the_score_threshold():
    """0.12 is confident FOR A SCORE; a single 0.55 threshold would reject it."""
    p = tiny_profile()
    answers = {"a": ans(normalized=0.5, confidence=0.12, qtype="score"),
               "b": ans(normalized=0.5, confidence=0.9)}
    assert is_uncertain(answers, p) is False


def test_score_confidence_below_its_own_threshold_is_uncertain():
    p = tiny_profile()
    answers = {"a": ans(normalized=0.5, confidence=0.05, qtype="score"),
               "b": ans(normalized=0.5, confidence=0.9)}
    assert is_uncertain(answers, p) is True


def test_noul_confidence_is_judged_against_the_noul_threshold():
    """noul confidence is floored at 0.5, so 0.52 is genuinely a coin flip."""
    p = tiny_profile()
    answers = {"a": ans(normalized=0.5, confidence=0.9, qtype="score"),
               "b": ans(normalized=0.5, confidence=0.52)}
    assert is_uncertain(answers, p) is True


def test_only_weighted_questions_drive_uncertainty():
    """An unweighted question's confidence does not route the window."""
    p = tiny_profile()
    answers = {"a": ans(normalized=0.5, confidence=0.9, qtype="score"),
               "b": ans(normalized=0.5, confidence=0.9),
               "g": ans(normalized=0.5, confidence=0.01)}
    assert is_uncertain(answers, p) is False


def test_rank_annotates_every_record():
    p = tiny_profile(gates={"g": 0.35})
    scores = [{"id": 0, "failed": False,
               "answers": {"a": ans(normalized=1.0, qtype="score"),
                           "b": ans(normalized=1.0), "g": ans(normalized=1.0)}}]
    out = rank(scores, p)
    assert out[0]["composite"] == pytest.approx(1.0)
    assert out[0]["gated"] is True
    assert out[0]["uncertain"] is False


def test_rank_passes_failed_records_through_untouched():
    p = tiny_profile()
    out = rank([{"id": 3, "failed": True, "answers": {}, "error": "boom"}], p)
    assert out[0]["failed"] is True
    assert "composite" not in out[0]


def test_real_core_profile_ranks_a_plausible_window():
    p = load_profile("core")
    answers = {k: ans(normalized=0.8, confidence=0.9,
                      qtype=p.questions[k]["type"]) for k in p.questions}
    assert 0.0 <= composite(answers, p) <= 1.0
    assert passes_gate(answers, p) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rank.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'clipper.rank'`

- [ ] **Step 3: Implement `clipper/rank.py`**

```python
from __future__ import annotations

from clipper.profiles.loader import Profile


def _normalized(answers: dict, key: str) -> float:
    """The 0..1 value for one question, or 0.0 if absent or non-numeric.

    Reads `normalized`, never `value`: Laya's raw score is 0..k-1, and feeding
    that into weights calibrated for 0..1 would silently inflate every composite.
    """
    answer = answers.get(key)
    if not answer:
        return 0.0
    value = answer.get("normalized")
    if not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def composite(answers: dict, profile: Profile) -> float:
    total = sum(weight * _normalized(answers, key)
                for key, weight in profile.weights.items())
    total -= sum(weight * _normalized(answers, key)
                 for key, weight in profile.penalties.items())
    return min(1.0, max(0.0, total))


def passes_gate(answers: dict, profile: Profile) -> bool:
    """Fails closed: a missing gate answer disqualifies rather than passing."""
    return all(_normalized(answers, key) >= floor
               for key, floor in profile.gates.items())


def is_uncertain(answers: dict, profile: Profile) -> bool:
    """True when any question driving the composite is below its type's threshold.

    Per-primitive because Laya computes confidence two different ways:
    `score`/`choice` use 1 - H(p)/log k and run low by construction, while
    `noul` uses max(p, 1-p) and cannot go below 0.5. A single threshold marks
    either every window or no window uncertain.
    """
    for key in profile.weights:
        answer = answers.get(key)
        if not answer:
            continue
        floor = profile.uncertain_confidence.get(answer.get("type"))
        if floor is None:
            continue
        confidence = answer.get("confidence")
        if not isinstance(confidence, (int, float)):
            continue
        if float(confidence) < floor:
            return True
    return False


def rank(scores: list[dict], profile: Profile) -> list[dict]:
    out: list[dict] = []
    for record in scores:
        if record.get("failed"):
            out.append(record)
            continue
        answers = record["answers"]
        out.append({**record,
                    "composite": composite(answers, profile),
                    "gated": passes_gate(answers, profile),
                    "uncertain": is_uncertain(answers, profile)})
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rank.py -v`
Expected: 16 passed

- [ ] **Step 5: Commit**

```bash
git add clipper/rank.py tests/test_rank.py
git commit -m "feat: composite scoring with per-primitive uncertainty routing

Composite reads normalized values only. Uncertainty compares each
weighted question against its own primitive's threshold, because score
confidence runs 0.11-0.25 while noul confidence is floored at 0.5 -- one
threshold would route every window, or none, to Claude."
```

---

### Task 12: Candidate merging and backward extension

**Files:**
- Create: `clipper/merge.py`
- Test: `tests/test_merge.py`

**Interfaces:**
- Consumes: ranked scores (Task 11), windows (Task 6), `Profile` (Task 7).
- Produces: `merge_candidates(windows: list[dict], ranked: list[dict], profile: Profile) -> dict` returning `{"candidates": [...], "uncertain": [...]}`.

Candidate dict: `{"id", "start", "end", "peak_composite", "mean_composite", "window_ids", "curve", "clip_format", "signals", "backward_extended"}`.

**Carry this task forward verbatim from `.superpowers/sdd/2026-09-21-jev-claude-clipper/task-10-brief.md`.** It consumes only `composite`, `gated`, `uncertain` and the window geometry — none of which changed shape in the Laya swap — so the brief applies unaltered.

Two adjustments the executor must make while transcribing it:

1. Wherever the brief reads a signal off an answer for the `signals` block, read `answer["normalized"]` rather than `answer["value"]`, matching Task 11's `_normalized`. For `clip_format`, keep `answer["value"]` — it is a `choice`, so it has no normalized form.
2. Spec §7 constraints hold unchanged: candidates cap at **60 s**; a candidate is extended backwards when its peak window has a late `energy_peak_offset` (`merge.backward_extend_offset`, default 0.6) by at most `merge.backward_extend_seconds` (default 8.0); **energy alone never promotes a candidate** — a spike must be corroborated by a transcript-derived signal.

---

### Task 13: Score stage wiring

**Files:**
- Create: `clipper/stages/score.py`, `clipper/stages/__init__.py`
- Test: `tests/test_stage_score.py`

**Interfaces:**
- Consumes: `Run` (Task 2), `load_profile` (Task 7), `load_agent` (Task 10), `score_windows` (Task 9), `rank` (Task 11), `merge_candidates` (Task 12).
- Produces: `run_score(run: Run, profile_name: str, device: str | None = None, agent=None) -> dict`.

Writes `scores.json` as `{"profile", "laya_model", "windows": [...]}` and `candidates.json` as `{"profile", "laya_model", "candidates": [...], "uncertain": [...]}`.

**Synchronous**, unlike the superseded Jev version: Laya is local and compute-bound, so there is nothing to overlap. `agent` is injectable so the default suite never loads a model.

- [ ] **Step 1: Write the failing test**

```python
import json

import pytest

from clipper.run import Run
from clipper.stages.score import run_score


class FakeAgent:
    def __init__(self):
        self.calls = 0

    def system_one(self, state, questions):
        self.calls += 1
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "score":
                answers[qid] = {"type": "score", "score": 3.0, "confidence": 0.3,
                                "legend": {}, "probabilities": {}}
            elif q["type"] == "choice":
                key = next(iter(q["criteria"]))
                answers[qid] = {"type": "choice", "choice": key, "confidence": 0.3,
                                "probabilities": {}}
            else:
                answers[qid] = {"type": "noul", "noul": 0.9, "confidence": 0.9,
                                "probabilities": {}}
        return {"model": "laya-rl-agent", "answers": answers, "usage": {}}


META = {"package": "0.3.5", "repo": "convaiinnovations/laya", "checkpoint": "root",
        "revision": "abc123", "device": "xpu", "device_requested": "xpu",
        "dtype": "torch.bfloat16", "confidence_calibrated": True,
        "uncalibrated_buckets": []}


@pytest.fixture
def run_with_windows(tmp_path):
    run = Run.create(tmp_path, "ep")
    run.write_json("transcript.json", {"language": "en", "model": "large-v3",
                                       "diarized": True, "segments": []})
    run.write_json("windows.json", {"windows": [
        {"id": i, "start": i * 10.0, "end": i * 10.0 + 30.0,
         "text": f"SPEAKER_01: line {i}", "preceding": "", "position": i / 10,
         "energy_mean": 0.4, "energy_peak": 0.8, "energy_peak_offset": 0.5}
        for i in range(4)]})
    return run


def test_score_writes_both_artifacts(run_with_windows, monkeypatch):
    monkeypatch.setattr("clipper.stages.score.load_agent",
                        lambda *a, **k: (FakeAgent(), META))
    run_score(run_with_windows, "podcast")
    assert run_with_windows.exists("scores.json")
    assert run_with_windows.exists("candidates.json")


def test_provenance_is_written_to_scores_and_carried_to_candidates(run_with_windows,
                                                                   monkeypatch):
    monkeypatch.setattr("clipper.stages.score.load_agent",
                        lambda *a, **k: (FakeAgent(), META))
    run_score(run_with_windows, "podcast")
    scores = run_with_windows.read_json("scores.json")
    candidates = run_with_windows.read_json("candidates.json")
    assert scores["laya_model"]["revision"] == "abc123"
    assert candidates["laya_model"] == scores["laya_model"]


def test_the_agent_is_loaded_once_and_called_once_per_window(run_with_windows,
                                                             monkeypatch):
    agent = FakeAgent()
    loads = []

    def fake_load(*a, **k):
        loads.append(1)
        return agent, META

    monkeypatch.setattr("clipper.stages.score.load_agent", fake_load)
    run_score(run_with_windows, "podcast")
    assert len(loads) == 1
    assert agent.calls == 4


def test_the_transcript_language_selects_the_checkpoint(run_with_windows, monkeypatch):
    seen = {}

    def fake_load(language, profile, device=None, **k):
        seen["language"] = language
        return FakeAgent(), META

    monkeypatch.setattr("clipper.stages.score.load_agent", fake_load)
    run_with_windows.write_json("transcript.json", {"language": "de", "segments": []})
    run_score(run_with_windows, "podcast")
    assert seen["language"] == "de"


def test_raw_answers_are_retained_for_recalibration(run_with_windows, monkeypatch):
    """Spec 13: a weighting change must be a recomputation, not a re-run."""
    monkeypatch.setattr("clipper.stages.score.load_agent",
                        lambda *a, **k: (FakeAgent(), META))
    run_score(run_with_windows, "podcast")
    first = run_with_windows.read_json("scores.json")["windows"][0]
    assert first["answers"]["clipworthy"]["value"] == 3.0
    assert first["answers"]["clipworthy"]["normalized"] == pytest.approx(0.75)
    assert "probabilities" in first["answers"]["clipworthy"]


def test_an_injected_agent_is_used_instead_of_loading(run_with_windows):
    agent = FakeAgent()
    run_score(run_with_windows, "podcast", agent=(agent, META))
    assert agent.calls == 4


def test_missing_windows_artifact_names_the_stage_that_makes_it(tmp_path):
    run = Run.create(tmp_path, "ep")
    run.write_json("transcript.json", {"language": "en", "segments": []})
    with pytest.raises(Exception, match="window"):
        run_score(run, "podcast")


def test_the_returned_summary_reports_counts(run_with_windows, monkeypatch):
    monkeypatch.setattr("clipper.stages.score.load_agent",
                        lambda *a, **k: (FakeAgent(), META))
    summary = run_score(run_with_windows, "podcast")
    assert summary["scored"] == 4
    assert summary["failed"] == 0
    assert "candidates" in summary
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stage_score.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'clipper.stages'`

- [ ] **Step 3: Implement `clipper/stages/__init__.py`**

```python
```

(An empty file; the package marker only.)

- [ ] **Step 4: Implement `clipper/stages/score.py`**

```python
from __future__ import annotations

from clipper.agent import load_agent
from clipper.merge import merge_candidates
from clipper.profiles.loader import load_profile
from clipper.rank import rank
from clipper.run import Run
from clipper.score import score_windows


def run_score(run: Run, profile_name: str, device: str | None = None,
              agent=None) -> dict:
    """Score every window and write scores.json plus candidates.json.

    Synchronous by design: Laya runs locally and is compute-bound, so there is
    no latency to overlap. The agent is loaded once and reused across windows.

    `agent` accepts a prepared `(agent, laya_model)` pair, which is how tests
    run the whole stage without loading a model.
    """
    profile = load_profile(profile_name)
    windows = run.read_json("windows.json")["windows"]
    language = (run.read_json("transcript.json") or {}).get("language")

    if agent is None:
        agent_obj, meta = load_agent(language, profile, device=device)
    else:
        agent_obj, meta = agent

    records = score_windows(windows, profile, agent_obj)
    ranked = rank(records, profile)
    merged = merge_candidates(windows, ranked, profile)

    run.write_json("scores.json", {"profile": profile.name,
                                   "laya_model": meta,
                                   "windows": ranked})
    run.write_json("candidates.json", {"profile": profile.name,
                                       "laya_model": meta,
                                       "candidates": merged["candidates"],
                                       "uncertain": merged["uncertain"]})

    failed = sum(1 for r in ranked if r.get("failed"))
    return {"scored": len(ranked) - failed,
            "failed": failed,
            "candidates": len(merged["candidates"]),
            "uncertain": len(merged["uncertain"]),
            "device": meta["device"]}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_stage_score.py -v`
Expected: 8 passed

- [ ] **Step 6: Run the whole suite**

Run: `pytest -q`
Expected: all green, `model` deselected.

- [ ] **Step 7: Commit**

```bash
git add clipper/stages tests/test_stage_score.py
git commit -m "feat: score stage wiring with Laya provenance

Synchronous: Laya is local and compute-bound. Agent loads once per run.
laya_model provenance is written to scores.json and carried verbatim into
candidates.json so thresholds stay traceable to a checkpoint and device."
```

---

## Tasks 14–20: carried forward

These seven tasks are **unaffected by the scoring-engine swap** — captions, ASS rendering, crop arithmetic, ffmpeg invocation and the writer adapter never touched Jev. Their briefs are already extracted, complete and reviewed in `.superpowers/sdd/2026-09-21-jev-claude-clipper/`. Execute them from those briefs rather than rewriting them here; rewriting reviewed work risks drift for no gain.

| Task | Brief | Patch needed |
|---|---|---|
| 14. Caption cues, rebasing and SRT | `task-12-brief.md` | None |
| 15. ASS rendering with word-level highlighting | `task-13-brief.md` | None |
| 16. Crop arithmetic and filter chain | `task-14-brief.md` | None |
| 17. Render stage | `task-15-brief.md` | **Yes** — see below |
| 18. LLM writer adapter | `task-16-brief.md` | None |
| 19. Plan validation and the Claude skill | `task-17-brief.md` | **Yes** — see below |
| 20. CLI wiring and the `all` pipeline | `task-18-brief.md` | **Yes** — see below |

### Patch for Task 17 (from `task-15-brief.md`)

The brief writes per-clip sidecar metadata, splitting the scoring block out of the clip dict. Replace both occurrences of `"jev"`:

```python
# Before
meta = {k: v for k, v in clip.items() if k != "jev"}
meta["jev"] = clip.get("jev", {})

# After
meta = {k: v for k, v in clip.items() if k != "laya"}
meta["laya"] = clip.get("laya", {})
```

### Patch for Task 19 (from `task-17-brief.md`)

Three changes:

1. The fixture builds a plan with `"jev_model": "jev-1.13.0"`. Replace with the provenance object:

```python
return {"profile": "podcast",
        "laya_model": {"package": "0.3.5", "repo": "convaiinnovations/laya",
                       "checkpoint": "root", "revision": "abc123",
                       "device": "xpu", "dtype": "torch.bfloat16",
                       "confidence_calibrated": True, "uncalibrated_buckets": []},
        "clips": [...]}
```

2. Rename `test_missing_jev_model_is_flagged` to `test_missing_laya_model_is_flagged`, deleting `plan["laya_model"]` and asserting `any("laya_model" in w for w in validate_plan(plan, 5400.0))`.

3. In `validate_plan`, replace the provenance check and add a calibration check:

```python
if not plan.get("laya_model"):
    warnings.append(
        "plan.json has no laya_model; thresholds cannot be traced to a checkpoint."
    )
elif not plan["laya_model"].get("confidence_calibrated", True):
    buckets = ", ".join(plan["laya_model"].get("uncalibrated_buckets") or [])
    warnings.append(
        f"laya_model reports uncalibrated confidence buckets ({buckets}); "
        f"the uncertain bucket may be unreliable."
    )
```

Also update the skill prose the brief writes into `.claude/skills/clipper/SKILL.md`: "Jev flagged these as low-confidence" becomes "Laya flagged these as low-confidence", and "Carry `jev_model` across verbatim" becomes "Carry `laya_model` across verbatim". The example clip block's `"jev": {...}` becomes `"laya": {"clipworthy": 0.72, "hook_strength": 0.61, "confidence": 0.18}` — note the values are now 0..1.

### Patch for Task 20 (from `task-18-brief.md`)

1. The `score` subcommand gains `--device`, passed through to `run_score`:

```python
score_parser.add_argument(
    "--device", default=None,
    choices=["auto", "cuda", "xpu", "mps", "cpu"],
    help="Device for Laya. Default: CLIPPER_LAYA_DEVICE, else auto "
         "(cuda -> xpu -> mps -> cpu). Intel Arc needs xpu; Laya's own "
         "auto-detect cannot see it.")
```

`run_score` is synchronous, so the brief's `asyncio.run(...)` wrapper around the score stage is dropped — call it directly.

2. The README the brief writes needs its title and opening replaced:

```markdown
# Laya Claude Clipper

Turns a long video into short, upload-ready clips.

Laya scores every 30-second window of the transcript for clip-worthiness,
Claude selects and trims the clips and writes their copy, and ffmpeg renders
them with captions. Laya runs locally: no API key, no per-clip cost.

Design: `docs/superpowers/specs/2026-09-22-laya-claude-clipper-design.md`
```

Add a Requirements section naming the device situation, since it is the one thing a new user will get wrong:

```markdown
## Requirements

- Python 3.11+, ffmpeg with libass
- A GPU is strongly recommended for scoring. Measured on a 90-minute source:
  13 minutes on an Intel Arc 140T, 78 minutes on CPU.
- Intel Arc users: install a torch build with XPU support. Laya's own device
  detection cannot see Arc, so set `CLIPPER_LAYA_DEVICE=xpu` or rely on
  clipper's `auto`, which probes for it.
```

3. The brief's spec cross-reference map at its end should point at the new spec path and add the two new modules: "§7 device selection → Task 8. §7 scoring, normalization, provenance → Tasks 9, 10, 11, 13."

---

## Self-Review

**Spec coverage.** Walked every section of `2026-09-22-laya-claude-clipper-design.md`:

| Spec section | Task |
|---|---|
| §3 project layout | Tasks 7–13 create every new module named there |
| §7 profile YAML → Laya schema | Task 7 |
| §7 state, field ordering | Task 9 (`window_state`, ordering test) |
| §7 normalization 0..1 | Task 9 (`normalize_answer`), Task 11 (`_normalized`) |
| §7 ranking, gates | Task 11 |
| §7 merging, backward extension, 60s cap | Task 12 |
| §7 per-type confidence routing | Task 11 (`is_uncertain`) |
| §7 calibration integrity, used buckets | Task 10 (`used_buckets`, `uncalibrated_buckets`) |
| §7 device selection, silent-fallback detection | Task 8, Task 10 |
| §7 checkpoint selection per language | Task 10 (`checkpoint_for_language`), Task 13 |
| §7 provenance `laya_model` | Task 10, Task 13, Task 19 patch |
| §8 plan.json shape | Task 19 patch |
| §9 render, captions, crop | Tasks 14–17 |
| §10 CLI, `CLIPPER_LAYA_DEVICE`, `CLIPPER_LAYA_REPO` | Task 7 (`.env.example`), Task 8, Task 20 |
| §10 `TYPESAFE_API_KEY` removed | Task 7 |
| §11 failure modes | Tasks 7–13 (profile validation, device fallback, load failure, per-window failure), Task 19 (plan validation) |
| §11 `model` marker replaces `live` | Task 7 |
| §11 fake agent, device stub, structural guard, smoke test | Tasks 7, 8, 9, 10 |
| §13 raw answers retained | Task 9, asserted in Task 13 |

No gaps found.

**Placeholder scan.** No "TBD", "TODO", "add error handling", or "similar to Task N". Tasks 12 and 14–20 point at existing complete brief files — a concrete artifact in this workspace, with every required edit spelled out as a diff, not a description of one.

**Type consistency.** Verified across tasks: `Profile.uncertain_confidence` (Task 7) is read by `is_uncertain` (Task 11) and `used_buckets` (Task 10). `answers_to_dict` (Task 9) produces `normalized`, which `_normalized` (Task 11) consumes — the name that would have silently broken the composite if it drifted. `load_agent` (Task 10) returns `(agent, meta)`, matching `run_score`'s unpacking (Task 13). `score_windows(windows, profile, agent)` (Task 9) matches its call in Task 13. `merge_candidates(windows, ranked, profile)` (Task 12) matches Task 13.

One inconsistency found and fixed while reviewing: Task 13's test injected `agent=` as a bare agent while `run_score` unpacked a pair; the test now passes `(agent, META)` and the signature documents it.
