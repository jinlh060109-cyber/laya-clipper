import pytest

from clipper import hardware


def test_auto_encoder_takes_the_first_working_hardware_encoder():
    assert hardware.pick_encoder("auto", ["h264_qsv"]) == "h264_qsv"
    assert hardware.pick_encoder("auto", ["h264_amf", "h264_nvenc"]) == "h264_nvenc"


def test_auto_encoder_without_hardware_uses_the_software_encoder():
    assert hardware.pick_encoder("auto", []) == "libx264"


def test_a_requested_encoder_that_does_not_work_falls_back_with_a_warning():
    with pytest.warns(RuntimeWarning, match="nvenc"):
        assert hardware.pick_encoder("nvenc", []) == "libx264"


def test_a_requested_encoder_that_works_is_used():
    assert hardware.pick_encoder("amf", ["h264_amf"]) == "h264_amf"


def test_unknown_encoder_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown encoder"):
        hardware.pick_encoder("h265_magic", [])


def test_probe_keeps_only_encoders_whose_test_encode_succeeds(monkeypatch):
    import subprocess
    tried = []

    def fake_run(cmd, **kw):
        encoder = cmd[cmd.index("-c:v") + 1]
        tried.append(encoder)
        return subprocess.CompletedProcess(cmd, 0 if encoder == "h264_qsv" else 1, "", "")

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)
    hardware.probe_encoders.cache_clear()
    assert hardware.probe_encoders("ffmpeg") == ["h264_qsv"]
    assert set(tried) == {"h264_nvenc", "h264_amf", "h264_qsv", "h264_videotoolbox"}
    hardware.probe_encoders.cache_clear()


@pytest.mark.parametrize("device, rocm, expected", [
    ("cuda", False, True), ("cpu", False, True),
    ("cuda", True, False), ("xpu", False, False), ("mps", False, False)])
def test_faster_whisper_only_runs_on_nvidia_cuda_and_cpu(device, rocm, expected):
    from clipper.transcribe import uses_faster_whisper
    assert uses_faster_whisper(device, rocm) is expected


def _gpu(name):
    return lambda: name


def test_detect_lists_every_device_and_explains_missing_ones():
    info = hardware.detect(gpus={"cuda": _gpu(None), "xpu": _gpu("Intel(R) Arc(TM) 140T"),
                                 "mps": _gpu(None)},
                           rocm=False, encoders=["h264_qsv"])
    by_id = {d["id"]: d for d in info["devices"]}
    assert list(by_id) == ["auto", "cuda", "xpu", "mps", "cpu"]
    assert by_id["xpu"]["available"] and "Arc" in by_id["xpu"]["detail"]
    assert not by_id["cuda"]["available"] and "torch" in by_id["cuda"]["hint"]
    assert by_id["cpu"]["available"]
    assert "Intel GPU" in by_id["auto"]["detail"]
    enc = {e["id"]: e for e in info["encoders"]}
    assert enc["qsv"]["available"] and not enc["nvenc"]["available"]
    assert enc["auto"]["detail"] == "Intel Quick Sync"


def test_a_working_qsv_encoder_reveals_an_intel_gpu_a_cpu_torch_hides():
    """The case this machine actually hit: ffmpeg encodes with Quick Sync, so
    an Intel GPU is present, but a CPU-only torch build cannot use it."""
    info = hardware.detect(gpus={"cuda": _gpu(None), "xpu": _gpu(None),
                                 "mps": _gpu(None)},
                           rocm=False, encoders=["h264_qsv"])
    xpu = next(d for d in info["devices"] if d["id"] == "xpu")
    assert not xpu["available"]
    assert "GPU found (Intel Quick Sync works)" in xpu["detail"]
    assert "torch" in xpu["detail"]


def test_a_working_nvenc_encoder_reveals_an_nvidia_gpu_a_cpu_torch_hides():
    """The fresh-clone NVIDIA case: plain PyPI torch on Windows is CPU-only,
    but ffmpeg's NVENC probe proves the GPU is there."""
    info = hardware.detect(gpus={"cuda": _gpu(None), "xpu": _gpu(None),
                                 "mps": _gpu(None)},
                           rocm=False, encoders=["h264_nvenc"])
    cuda = next(d for d in info["devices"] if d["id"] == "cuda")
    assert not cuda["available"]
    assert "GPU found (NVIDIA NVENC works)" in cuda["detail"]
    assert "torch" in cuda["detail"]
    # No QSV evidence: the Intel entry must stay a plain "Not found".
    xpu = next(d for d in info["devices"] if d["id"] == "xpu")
    assert xpu["detail"] == "Not found"


def test_amd_on_rocm_is_labelled_as_amd():
    info = hardware.detect(gpus={"cuda": _gpu("AMD Radeon RX 7900 XTX"), "xpu": _gpu(None),
                                 "mps": _gpu(None)}, rocm=True, encoders=[])
    cuda = next(d for d in info["devices"] if d["id"] == "cuda")
    assert cuda["label"] == "AMD GPU (ROCm)" and cuda["available"]


def test_detect_probes_the_ffmpeg_the_app_uses(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(hardware, "find_binary", lambda name, env: tmp_path / "my-ffmpeg")
    monkeypatch.setattr(hardware, "probe_encoders", lambda ffmpeg: seen.append(ffmpeg) or [])
    hardware.detect(gpus={"cuda": _gpu(None), "xpu": _gpu(None), "mps": _gpu(None)}, rocm=False)
    assert seen == [str(tmp_path / "my-ffmpeg")]
