from __future__ import annotations

import os
import warnings

from clipper.device import resolve_device

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


def _questions(questions) -> dict:
    """A question dict; a legacy Profile is read through its `questions`."""
    return getattr(questions, "questions", questions)


def used_buckets(questions) -> set[str]:
    """The temperature buckets these questions actually reach."""
    out: set[str] = set()
    for q in _questions(questions).values():
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


def uncalibrated_buckets(agent, questions) -> list[str]:
    """Buckets this profile reaches whose shipped temperature was out of range.

    Evaluated against the buckets actually used, not against whether Laya warned
    at all: the shipped checkpoint clamps `choice:11+`, which no profile reaches,
    so a flag keyed on the warning would be permanently and uselessly false.
    """
    raw = getattr(agent, "temperature_by_options_raw", {}) or {}
    reachable = used_buckets(questions)
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


def load_agent(language: str | None, questions, device: str | None = None,
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

    bad = uncalibrated_buckets(agent, questions)
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
