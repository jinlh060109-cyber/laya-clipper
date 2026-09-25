"""What this machine can run on: GPUs for the models, encoders for the video.

Models (Whisper, Laya) run through torch, which names devices `cuda` (NVIDIA,
and AMD on Linux through ROCm), `xpu` (Intel Arc), `mps` (Apple) or `cpu`.
Video encoding runs through ffmpeg, whose hardware encoders are listed by
every build but only work when the matching GPU and driver are present, so
each one is tried with a one-frame test encode.
"""
from __future__ import annotations

import functools
import subprocess
import warnings
from typing import Callable

from clipper.preflight import find_binary

# (setting id, ffmpeg encoder, label), in the order `auto` prefers them.
ENCODERS: tuple[tuple[str, str, str], ...] = (
    ("nvenc", "h264_nvenc", "NVIDIA NVENC"),
    ("amf", "h264_amf", "AMD AMF"),
    ("qsv", "h264_qsv", "Intel Quick Sync"),
    ("videotoolbox", "h264_videotoolbox", "Apple VideoToolbox"),
    ("libx264", "libx264", "Software (x264)"),
)
_BY_ID = {eid: (name, label) for eid, name, label in ENCODERS}
# A hardware encoder whose test encode succeeded proves that GPU is present,
# whatever torch says: Quick Sync -> Intel, NVENC -> NVIDIA, AMF -> AMD.
_GPU_EVIDENCE = {"xpu": "qsv", "cuda": "nvenc", "mps": "videotoolbox"}
_QUALITY = {
    "libx264": ["-preset", "medium", "-crf", "20"],
    "h264_nvenc": ["-preset", "p5", "-rc", "vbr", "-cq", "21", "-b:v", "0"],
    "h264_amf": ["-quality", "quality", "-rc", "cqp", "-qp_i", "20", "-qp_p", "22"],
    "h264_qsv": ["-preset", "medium", "-global_quality", "21"],
    "h264_videotoolbox": ["-q:v", "60"],
}

DEVICE_LABELS = {"cuda": "NVIDIA GPU (CUDA)", "xpu": "Intel GPU (XPU)",
                 "mps": "Apple GPU (Metal)", "cpu": "CPU"}
DEVICE_HINTS = {
    "cuda": ("Needs an NVIDIA GPU and a CUDA build of torch: pip install torch "
             "--index-url https://download.pytorch.org/whl/cu124. AMD GPUs need a "
             "ROCm build of torch, which is Linux-only; on Windows an AMD GPU still "
             "encodes video (AMF) but the models run on the CPU."),
    "xpu": ("Needs an Intel Arc GPU and an XPU build of torch: pip install torch "
            "--index-url https://download.pytorch.org/whl/xpu"),
    "mps": "Needs a Mac with Apple silicon and a recent torch.",
}


def pick_encoder(requested: str, available: list[str]) -> str:
    """The ffmpeg encoder to use. `available` lists encoders that passed a test."""
    if requested not in _BY_ID and requested != "auto":
        raise ValueError(f"Unknown encoder {requested!r}. Valid: auto, "
                         f"{', '.join(_BY_ID)}.")
    if requested == "auto":
        return next((name for _, name, _ in ENCODERS if name in available), "libx264")
    name = _BY_ID[requested][0]
    if name == "libx264" or name in available:
        return name
    warnings.warn(f"The {requested} encoder does not work on this machine; "
                  f"using software encoding (libx264) instead.", RuntimeWarning,
                  stacklevel=2)
    return "libx264"


def encoder_args(name: str) -> list[str]:
    return ["-c:v", name, *_QUALITY[name]]


@functools.lru_cache(maxsize=4)
def probe_encoders(ffmpeg: str) -> list[str]:
    """Hardware encoders that can encode one frame here. Cached per process."""
    working = []
    for _, name, _ in ENCODERS:
        if name == "libx264":
            continue
        cmd = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-f", "lavfi",
               "-i", "color=c=black:s=256x256:d=0.1", "-frames:v", "1",
               "-c:v", name, "-f", "null", "-"]
        try:
            result = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                    timeout=20, check=False)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            working.append(name)
    return working


def free_device_memory(device: str) -> None:
    """Hand cached GPU memory back to the driver. On an integrated GPU (Intel
    Arc) that memory is system RAM, and the next model needs it. Only models
    nothing refers to any more can be freed."""
    if device == "cpu":
        return
    import gc

    import torch

    gc.collect()
    backend = getattr(torch, device, None)
    if backend is not None and hasattr(backend, "empty_cache"):
        backend.empty_cache()


def is_rocm() -> bool:
    """True when torch is an AMD ROCm build (it then reports devices as `cuda`)."""
    try:
        import torch

        return bool(getattr(torch.version, "hip", None))
    except Exception:
        return False


def _gpu_name(kind: str) -> Callable[[], str | None]:
    def probe() -> str | None:
        try:
            import torch

            backend = getattr(torch, kind, None)
            if kind == "mps":
                ok = bool(getattr(torch.backends, "mps", None)
                          and torch.backends.mps.is_available())
                return "Apple GPU" if ok else None
            if backend is None or not backend.is_available():
                return None
            return str(backend.get_device_name(0))
        except Exception:
            return None
    return probe


def detect(gpus: dict[str, Callable[[], str | None]] | None = None,
           rocm: bool | None = None, encoders: list[str] | None = None,
           ffmpeg: str | None = None) -> dict:
    """Everything the settings page needs to offer hardware choices."""
    gpus = gpus or {kind: _gpu_name(kind) for kind in ("cuda", "xpu", "mps")}
    rocm = is_rocm() if rocm is None else rocm
    found = {kind: probe() for kind, probe in gpus.items()}

    if encoders is None:
        # The same ffmpeg the renders use: FFMPEG_PATH, else the one on PATH.
        try:
            ffmpeg = ffmpeg or str(find_binary("ffmpeg", "FFMPEG_PATH"))
            encoders = probe_encoders(ffmpeg)
        except Exception:  # noqa: BLE001 - no ffmpeg: software encoding is still listed
            encoders = []
    working = encoders

    devices = []
    for kind in ("cuda", "xpu", "mps"):
        label = "AMD GPU (ROCm)" if kind == "cuda" and rocm else DEVICE_LABELS[kind]
        name = found.get(kind)
        # ffmpeg's own encoder probes can prove a GPU torch cannot see: a
        # working Quick Sync / NVENC / AMF encoder means that GPU is there,
        # and the missing torch build is the reason it is not offered (a
        # plain `pip install torch` is CPU-only on Windows). Say that
        # instead of "Not found", which reads as "no GPU in this machine".
        detail = name
        if name is None:
            evidence = _GPU_EVIDENCE.get(kind)
            if evidence and _BY_ID[evidence][0] in working:
                detail = (f"GPU found ({_BY_ID[evidence][1]} works) but this torch "
                          "build cannot use it: see the hint below to reinstall torch")
        devices.append({"id": kind, "label": label, "available": bool(name),
                        "detail": detail or "Not found",
                        "hint": "" if name else DEVICE_HINTS[kind]})
    devices.append({"id": "cpu", "label": "CPU", "available": True,
                    "detail": "Always available (slowest)", "hint": ""})
    best = next((d for d in devices if d["available"]), devices[-1])
    auto = {"id": "auto", "label": "Automatic", "available": True,
            "detail": f"Uses {best['label']}" + (f": {best['detail']}" if best["id"] != "cpu" else ""),
            "hint": ""}

    enc = []
    for eid, name, label in ENCODERS:
        ok = name == "libx264" or name in working
        enc.append({"id": eid, "label": label, "available": ok})
    chosen = pick_encoder("auto", working)
    enc.insert(0, {"id": "auto", "label": "Automatic", "available": True,
                   "detail": next(label for _, n, label in ENCODERS if n == chosen)})
    return {"devices": [auto, *devices], "encoders": enc, "rocm": rocm}
