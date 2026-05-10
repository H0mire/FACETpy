from __future__ import annotations

import numpy as np
import pytest

from facet.core import ProcessingContext, ProcessingMetadata
from facet.models.cascaded_dae import CascadedDenoisingAutoencoderCorrection
from facet.models.cascaded_dae.training import CascadedDenoisingAutoencoder, ChannelWiseArtifactDataset


def test_cascaded_dae_model_preserves_single_channel_shape():
    torch = pytest.importorskip("torch")
    model = CascadedDenoisingAutoencoder(input_size=16, hidden_units=(8, 4, 8), dropout_rate=0.0)
    x = torch.randn(3, 1, 16)
    y = model(x)
    assert tuple(y.shape) == (3, 1, 16)


def test_cascaded_dae_second_stage_receives_stage1_corrected_signal():
    torch = pytest.importorskip("torch")

    class HalfArtifact(torch.nn.Module):
        def forward(self, x):
            return x * 0.5

    class CaptureResidual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, x):
            self.seen = x.detach().clone()
            return torch.zeros_like(x)

    model = CascadedDenoisingAutoencoder(input_size=4, hidden_units=(4, 2, 4), dropout_rate=0.0)
    model.stage1 = HalfArtifact()
    model.stage2 = CaptureResidual()
    x = torch.ones(2, 1, 4)

    y = model(x)

    torch.testing.assert_close(model.stage2.seen, torch.full_like(x, 0.5))
    torch.testing.assert_close(y, torch.full_like(x, 0.5))


class _BaseDataset:
    trigger_aligned = True
    sfreq = 2048.0

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        noisy = np.arange(2 * 4, dtype=np.float32).reshape(2, 4) + idx
        target = noisy * 0.5
        return noisy, target


def test_channel_wise_dataset_expands_channels():
    dataset = ChannelWiseArtifactDataset(_BaseDataset(), demean_input=False, demean_target=False)
    assert len(dataset) == 4
    noisy, target = dataset[1]
    assert noisy.shape == (1, 4)
    assert target.shape == (1, 4)
    np.testing.assert_allclose(noisy[0], np.array([4, 5, 6, 7], dtype=np.float32))


def test_cascaded_dae_correction_applies_torchscript_per_channel(tmp_path):
    torch = pytest.importorskip("torch")
    mne = pytest.importorskip("mne")

    class ConstantArtifact(torch.nn.Module):
        def forward(self, x):
            return torch.ones_like(x) * 0.25

    checkpoint = tmp_path / "constant_artifact.ts"
    scripted = torch.jit.trace(ConstantArtifact(), torch.zeros(1, 1, 8))
    scripted.save(str(checkpoint))

    data = np.ones((2, 16), dtype=np.float64)
    info = mne.create_info(["C3", "C4"], sfreq=1000.0, ch_types="eeg")
    raw = mne.io.RawArray(data.copy(), info, verbose=False)
    context = ProcessingContext(raw=raw, metadata=ProcessingMetadata())

    result = context | CascadedDenoisingAutoencoderCorrection(
        checkpoint_path=checkpoint,
        chunk_size_samples=8,
        demean_input=False,
        remove_prediction_mean=False,
    )

    np.testing.assert_allclose(result.get_raw()._data, data - 0.25)
    np.testing.assert_allclose(result.get_estimated_noise(), np.full_like(data, 0.25))
    runs = result.metadata.custom["deep_learning_runs"]
    assert runs[-1]["model"] == "CascadedDenoisingAutoencoderAdapter"
