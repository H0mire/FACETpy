from __future__ import annotations

import numpy as np
import pytest

from facet.core import ProcessingContext, ProcessingMetadata
from facet.models.nested_gan import NestedGANCorrection


@pytest.mark.unit
def test_nested_gan_correction_applies_center_epochs(tmp_path):
    torch = pytest.importorskip("torch")
    mne = pytest.importorskip("mne")

    class ConstantCenterArtifact(torch.nn.Module):
        def forward(self, x):
            return torch.ones((x.shape[0], 1, x.shape[-1]), dtype=x.dtype, device=x.device) * 0.25

    checkpoint = tmp_path / "constant_nested_gan_artifact.ts"
    scripted = torch.jit.trace(ConstantCenterArtifact(), torch.zeros(1, 7, 1, 8))
    scripted.save(str(checkpoint))

    data = np.ones((2, 80), dtype=np.float64)
    info = mne.create_info(["C3", "C4"], sfreq=1000.0, ch_types="eeg")
    raw = mne.io.RawArray(data.copy(), info, verbose=False)
    metadata = ProcessingMetadata(triggers=np.arange(0, 80, 8, dtype=np.int32))
    context = ProcessingContext(raw=raw, metadata=metadata)

    result = context | NestedGANCorrection(
        checkpoint_path=checkpoint,
        context_epochs=7,
        epoch_samples=8,
        demean_input=False,
        remove_prediction_mean=False,
    )

    expected_noise = np.zeros_like(data)
    # Ten triggers define nine epochs. With seven-context inference, center
    # epochs 3, 4, and 5 are corrected: sample ranges [24, 48).
    expected_noise[:, 24:48] = 0.25
    np.testing.assert_allclose(result.get_estimated_noise(), expected_noise)
    np.testing.assert_allclose(result.get_raw()._data, data - expected_noise)
    runs = result.metadata.custom["deep_learning_runs"]
    assert runs[-1]["model"] == "NestedGANAdapter"
    assert runs[-1]["prediction_metadata"]["context_epochs"] == 7


@pytest.mark.unit
def test_nested_gan_correction_requires_enough_triggers(tmp_path):
    torch = pytest.importorskip("torch")
    mne = pytest.importorskip("mne")

    class Zero(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((x.shape[0], 1, x.shape[-1]), dtype=x.dtype, device=x.device)

    checkpoint = tmp_path / "zero_nested_gan.ts"
    torch.jit.trace(Zero(), torch.zeros(1, 7, 1, 8)).save(str(checkpoint))

    info = mne.create_info(["C3"], sfreq=1000.0, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((1, 32)), info, verbose=False)
    metadata = ProcessingMetadata(triggers=np.arange(0, 24, 8, dtype=np.int32))
    context = ProcessingContext(raw=raw, metadata=metadata)

    correction = NestedGANCorrection(
        checkpoint_path=checkpoint,
        context_epochs=7,
        epoch_samples=8,
    )
    with pytest.raises(Exception):
        correction.validate(context)
