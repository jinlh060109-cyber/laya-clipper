from clipper.energy import rolling_baseline

def test_output_length_matches_input():
    assert len(rolling_baseline([0.1] * 100, baseline_seconds=10)) == 100

def test_flat_signal_normalizes_to_midpoint():
    out = rolling_baseline([0.5] * 100, baseline_seconds=10)
    assert all(abs(v - 0.5) < 1e-6 for v in out)

def test_spike_in_quiet_stretch_scores_high():
    rms = [0.05] * 60 + [0.9] + [0.05] * 60
    out = rolling_baseline(rms, baseline_seconds=30)
    assert out[60] > 0.9

def test_loud_speaker_does_not_saturate():
    """A uniformly loud signal must not read as one long highlight."""
    out = rolling_baseline([0.95] * 200, baseline_seconds=30)
    assert max(out) < 0.6

def test_values_are_clamped_to_unit_range():
    rms = [0.0] * 50 + [1000.0] + [0.0] * 50
    out = rolling_baseline(rms, baseline_seconds=20)
    assert all(0.0 <= v <= 1.0 for v in out)

def test_empty_input_returns_empty():
    assert rolling_baseline([], baseline_seconds=30) == []
