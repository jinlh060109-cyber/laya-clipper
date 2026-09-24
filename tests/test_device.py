import pytest

from clipper.device import DeviceError, resolve_device


def probes(**kw):
    """Build a probe table; anything unnamed is unavailable."""
    return {name: (lambda v=kw.get(name, False): v) for name in ("cuda", "xpu", "mps")}


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


def test_auto_prefers_xpu_over_mps_when_both_are_available():
    """The ordering the suite must be able to detect: an Arc GPU must not
    lose to mps. Without both flags set, a swapped PROBE_ORDER passes."""
    assert resolve_device("auto", probes(xpu=True, mps=True)) == "xpu"


def test_auto_follows_the_documented_priority_when_everything_is_available():
    assert resolve_device("auto", probes(cuda=True, xpu=True, mps=True)) == "cuda"
