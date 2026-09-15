import numpy as np
import pytest

from facet.evaluation.thesis_metrics import compute_metrics


@pytest.mark.parametrize("position", range(4))
def test_metrics_reject_each_mismatched_array(position):
    arrays = [np.ones((2, 3, 16)) for _ in range(4)]
    arrays[position] = np.ones((1, 3, 16))
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_metrics(*arrays, sfreq_hz=4096)


def test_exact_artifact_removal_recovers_clean_signal():
    rng = np.random.default_rng(23)
    clean = rng.normal(0, 1e-5, (2, 3, 16))
    artifact = rng.normal(0, 1e-3, clean.shape)
    metrics = compute_metrics(clean + artifact, clean, artifact, artifact, sfreq_hz=4096)
    assert metrics["clean_mse_reduction_pct"] == pytest.approx(100)
    assert metrics["rms_recovery_ratio"] == pytest.approx(1)
    assert metrics["artifact_mse"] == 0
