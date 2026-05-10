from __future__ import annotations

import numpy as np
import pytest

from facet.core import ProcessingContext, ProcessingMetadata
from facet.models.ic_unet import IcUnetCorrection
from facet.models.ic_unet.training import (
    IcUnet1D,
    IcUnetWithIca,
    NiazyContextIcDataset,
    build_dataset,
    build_loss,
    build_model,
)


def test_icunet_core_forward_shape():
    torch = pytest.importorskip("torch")
    model = IcUnet1D(in_channels=4, out_channels=4, base_channels=8)
    x = torch.randn(2, 4, 128)
    y = model(x)
    assert tuple(y.shape) == (2, 4, 128)


def test_icunet_with_ica_outputs_center_artifact_shape():
    torch = pytest.importorskip("torch")
    model = IcUnetWithIca(
        n_channels=4,
        context_epochs=7,
        epoch_samples=64,
        base_channels=8,
        demean_input=False,
        ica_init=None,
    )
    x = torch.randn(2, 4, 7 * 64)
    y = model(x)
    assert tuple(y.shape) == (2, 4, 64)


def test_icunet_with_ica_one_batch_backward_pass():
    torch = pytest.importorskip("torch")
    model = IcUnetWithIca(
        n_channels=3,
        context_epochs=7,
        epoch_samples=32,
        base_channels=8,
        demean_input=False,
        ica_init=None,
    )
    x = torch.randn(2, 3, 7 * 32, requires_grad=False)
    target = torch.zeros(2, 3, 32)
    loss_fn = build_loss("mse")

    optimiser = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimiser.zero_grad()
    prediction = model(x)
    loss = loss_fn(prediction, target)
    loss.backward()

    grads_observed = any(
        param.grad is not None and torch.any(param.grad.abs() > 0)
        for param in model.unet.parameters()
    )
    assert grads_observed, "U-Net parameters did not receive gradients"


class _StubBaseDataset:
    sfreq = 4096.0

    def __len__(self) -> int:
        return 3

    def __getitem__(self, idx: int):
        noisy = (
            np.arange(7 * 4 * 16, dtype=np.float32).reshape(7, 4, 16) + idx
        )
        target = noisy[3] * 0.5
        return noisy, target


def test_niazy_context_ic_dataset_reshapes_to_multichannel_time_series():
    dataset = NiazyContextIcDataset(
        _StubBaseDataset(),
        demean_input=False,
        demean_target=False,
    )
    assert len(dataset) == 3
    assert dataset.input_shape == (4, 7 * 16)
    assert dataset.target_shape == (4, 16)

    noisy, target = dataset[1]
    assert noisy.shape == (4, 7 * 16)
    assert target.shape == (4, 16)

    expected_center_sample = noisy[0, 3 * 16]
    np.testing.assert_allclose(target[0, 0], expected_center_sample * 0.5)


def test_build_model_accepts_input_shape_only():
    pytest.importorskip("torch")
    model = build_model(
        input_shape=(4, 7 * 16),
        target_shape=(4, 16),
        context_epochs=7,
        epoch_samples=16,
        n_channels=4,
        base_channels=8,
        demean_input=False,
        fit_ica=False,
    )
    assert isinstance(model, IcUnetWithIca)
    assert model.n_channels == 4
    assert model.context_epochs == 7
    assert model.epoch_samples == 16


def test_build_loss_resolves_known_names():
    pytest.importorskip("torch")
    assert build_loss("mse") is not None
    assert build_loss("mae") is not None
    assert build_loss("ensemble") is not None
    with pytest.raises(ValueError):
        build_loss("unknown")


def test_ic_unet_correction_applies_center_epochs(tmp_path):
    torch = pytest.importorskip("torch")
    mne = pytest.importorskip("mne")

    class _ConstantArtifact(torch.nn.Module):
        def forward(self, x):
            return torch.ones(
                (x.shape[0], x.shape[1], x.shape[-1] // 7),
                dtype=x.dtype,
                device=x.device,
            ) * 0.25

    checkpoint = tmp_path / "constant_artifact.ts"
    scripted = torch.jit.trace(_ConstantArtifact(), torch.zeros(1, 2, 7 * 8))
    scripted.save(str(checkpoint))

    data = np.ones((2, 80), dtype=np.float64)
    info = mne.create_info(["C3", "C4"], sfreq=1000.0, ch_types="eeg")
    raw = mne.io.RawArray(data.copy(), info, verbose=False)
    metadata = ProcessingMetadata(triggers=np.arange(0, 80, 8, dtype=np.int32))
    context = ProcessingContext(raw=raw, metadata=metadata)

    result = context | IcUnetCorrection(
        checkpoint_path=checkpoint,
        context_epochs=7,
        epoch_samples=8,
        demean_input=False,
        remove_prediction_mean=False,
    )

    expected_noise = np.zeros_like(data)
    expected_noise[:, 24:48] = 0.25
    np.testing.assert_allclose(result.get_estimated_noise(), expected_noise)
    np.testing.assert_allclose(result.get_raw()._data, data - expected_noise)

    runs = result.metadata.custom["deep_learning_runs"]
    assert runs[-1]["model"] == "IcUnetAdapter"
    assert runs[-1]["prediction_metadata"]["context_epochs"] == 7
