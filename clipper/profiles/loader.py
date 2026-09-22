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
        if qtype == "noul":
            if not isinstance(crit, dict) or set(crit.keys()) != {"true", "false"}:
                raise ProfileError(
                    f"Profile {name!r} question {qid!r} is a noul; criteria must be a mapping "
                    f"with exactly the string keys \"true\" and \"false\", got {crit!r}. "
                    f"Unquoted YAML `true:`/`false:` parse as booleans, not strings — quote "
                    f'them as "true": and "false": in the YAML.'
                )

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
