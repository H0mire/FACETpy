from __future__ import annotations

import numpy as np
import pytest

from facet.core import ProcessingContext, ProcessingMetadata
from facet.models.dpae import DualPathwayAutoencoderCorrection
from facet.models.dpae.training import (
    ChannelWiseArtifactDataset,
    DualPathwayAutoencoder,
    build_model,
)


def test_dpae_model_preserves_single_channel_shape():
    torch = pytest.importorskip("torch")
    model = DualPathwayAutoencoder(input_size=128, base_filters=4, latent_filters=8)
    x = torch.randn(3, 1, 128)
    y = model(x)
    assert tuple(y.shape) == (3, 1, 128)


def test_dpae_build_model_from_input_shape():
    torch = pytest.importorskip("torch")
    model = build_model(input_shape=(1, 64), base_filters=4, latent_filters=8)
    x = torch.randn(2, 1, 64)
    y = model(x)
    assert tuple(y.shape) == (2, 1, 64)


def test_dpae_residual_init_zero_means_pure_decoder():
    torch = pytest.importorskip("torch")
    model = DualPathwayAutoencoder(
        input_size=64, base_filters=4, latent_filters=8, residual_init=0.0
    )
    assert float(model.residual_scale.detach().cpu()) == 0.0


class _BaseDataset:
    trigger_aligned = True
    sfreq = 2048.0

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        noisy = np.arange(2 * 4, dtype=np.float32).reshape(2, 4) + idx
        target = noisy * 0.25
        return noisy, target


def test_channel_wise_dataset_expands_channels():
    dataset = ChannelWiseArtifactDataset(_BaseDataset(), demean_input=False, demean_target=False)
    assert len(dataset) == 4
    noisy, target = dataset[1]
    assert noisy.shape == (1, 4)
    assert target.shape == (1, 4)
    np.testing.assert_allclose(noisy[0], np.array([4, 5, 6, 7], dtype=np.float32))


def test_dpae_correction_applies_torchscript_per_channel(tmp_path):
    torch = pytest.importorskip("torch")
    mne = pytest.importorskip("mne")

    class ConstantArtifact(torch.nn.Module):
        def forward(self, x):
            return torch.ones_like(x) * 0.5

    checkpoint = tmp_path / "constant_artifact.ts"
    scripted = torch.jit.trace(ConstantArtifact(), torch.zeros(1, 1, 8))
    scripted.save(str(checkpoint))

    data = np.ones((2, 80), dtype=np.float64)
    info = mne.create_info(["C3", "C4"], sfreq=1000.0, ch_types="eeg")
    raw = mne.io.RawArray(data.copy(), info, verbose=False)
    metadata = ProcessingMetadata(triggers=np.arange(0, 80, 8, dtype=np.int32))
    context = ProcessingContext(raw=raw, metadata=metadata)

    result = context | DualPathwayAutoencoderCorrection(
        checkpoint_path=checkpoint,
        epoch_samples=8,
        demean_input=False,
        remove_prediction_mean=False,
    )

    expected_noise = np.zeros_like(data)
    # Ten triggers define nine epochs (sample ranges [0,8) ... [64,72)). The
    # last epoch [72, 80) is also defined, so all nine intervals get filled.
    expected_noise[:, 0:72] = 0.5
    np.testing.assert_allclose(result.get_estimated_noise(), expected_noise)
    np.testing.assert_allclose(result.get_raw()._data, data - expected_noise)
    runs = result.metadata.custom["deep_learning_runs"]
    assert runs[-1]["model"] == "DualPathwayAutoencoderAdapter"
