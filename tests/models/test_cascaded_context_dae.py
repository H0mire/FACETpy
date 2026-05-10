from __future__ import annotations

import numpy as np
import pytest

from facet.core import ProcessingContext, ProcessingMetadata
from facet.models.cascaded_context_dae import CascadedContextDenoisingAutoencoderCorrection
from facet.models.cascaded_context_dae.training import (
    CascadedContextDenoisingAutoencoder,
    ChannelWiseContextArtifactDataset,
)


def test_cascaded_context_dae_model_predicts_center_epoch_shape():
    torch = pytest.importorskip("torch")
    model = CascadedContextDenoisingAutoencoder(
        input_size=7 * 8,
        output_size=8,
        hidden_units=(16, 4, 16),
        dropout_rate=0.0,
    )
    y = model(torch.randn(3, 7, 1, 8))
    assert tuple(y.shape) == (3, 1, 8)


def test_cascaded_context_dae_second_stage_receives_center_corrected_context():
    torch = pytest.importorskip("torch")

    class ConstantCenterArtifact(torch.nn.Module):
        def forward(self, x):
            return torch.ones((x.shape[0], 1, x.shape[-1]), dtype=x.dtype, device=x.device) * 2.0

    class CaptureContext(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, x):
            self.seen = x.detach().clone()
            return torch.zeros((x.shape[0], 1, x.shape[-1]), dtype=x.dtype, device=x.device)

    model = CascadedContextDenoisingAutoencoder(
        input_size=7 * 4,
        output_size=4,
        hidden_units=(8, 4, 8),
        dropout_rate=0.0,
    )
    model.stage1 = ConstantCenterArtifact()
    model.stage2 = CaptureContext()
    x = torch.ones(2, 7, 1, 4) * 10.0

    y = model(x)

    expected_context = x.clone()
    expected_context[:, 3:4, :, :] -= 2.0
    torch.testing.assert_close(model.stage2.seen, expected_context)
    torch.testing.assert_close(y, torch.ones(2, 1, 4) * 2.0)


class _ContextBaseDataset:
    sfreq = 2048.0

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        noisy = np.arange(7 * 2 * 4, dtype=np.float32).reshape(7, 2, 4) + idx
        target = noisy[3] * 0.5
        return noisy, target


def test_channel_wise_context_dataset_expands_channels():
    dataset = ChannelWiseContextArtifactDataset(
        _ContextBaseDataset(),
        context_epochs=7,
        demean_input=False,
        demean_target=False,
    )
    assert len(dataset) == 4
    noisy, target = dataset[1]
    assert noisy.shape == (7, 1, 4)
    assert target.shape == (1, 4)
    np.testing.assert_allclose(noisy[:, 0, 0], np.array([4, 12, 20, 28, 36, 44, 52], dtype=np.float32))


def test_cascaded_context_dae_correction_applies_center_epochs(tmp_path):
    torch = pytest.importorskip("torch")
    mne = pytest.importorskip("mne")

    class ConstantCenterArtifact(torch.nn.Module):
        def forward(self, x):
            return torch.ones((x.shape[0], 1, x.shape[-1]), dtype=x.dtype, device=x.device) * 0.25

    checkpoint = tmp_path / "constant_context_artifact.ts"
    scripted = torch.jit.trace(ConstantCenterArtifact(), torch.zeros(1, 7, 1, 8))
    scripted.save(str(checkpoint))

    data = np.ones((2, 80), dtype=np.float64)
    info = mne.create_info(["C3", "C4"], sfreq=1000.0, ch_types="eeg")
    raw = mne.io.RawArray(data.copy(), info, verbose=False)
    metadata = ProcessingMetadata(triggers=np.arange(0, 80, 8, dtype=np.int32))
    context = ProcessingContext(raw=raw, metadata=metadata)

    result = context | CascadedContextDenoisingAutoencoderCorrection(
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
    assert runs[-1]["model"] == "CascadedContextDenoisingAutoencoderAdapter"
    assert runs[-1]["prediction_metadata"]["context_epochs"] == 7
