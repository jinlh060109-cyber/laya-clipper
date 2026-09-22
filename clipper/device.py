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
        f"source, against 13 on an Intel Arc GPU). "
        f"Check that your torch build includes support for {requested!r} (e.g. a '+{requested}' build).",
        RuntimeWarning,
        stacklevel=2,
    )
    return "cpu"
