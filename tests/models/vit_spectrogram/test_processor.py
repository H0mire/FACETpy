"""Tests for the ViT Spectrogram Inpainter processor and adapter."""

from __future__ import annotations

import numpy as np
import pytest

from facet.core import ProcessingContext, ProcessingMetadata
from facet.models.vit_spectrogram import (
    ViTSpectrogramInpainterAdapter,
    ViTSpectrogramInpainterCorrection,
)
from facet.models.vit_spectrogram.training import (
    ChannelWiseSpectrogramDataset,
    ViTSpectrogramInpainter,
    build_loss,
    build_model,
)


@pytest.mark.unit
def test_build_model_returns_module_with_expected_input_shape():
    torch = pytest.importorskip("torch")
    model = build_model(input_shape=(7, 1, 512))
    assert isinstance(model, torch.nn.Module)
    assert isinstance(model, ViTSpectrogramInpainter)
    output = model(torch.zeros(2, 7, 1, 512))
    assert tuple(output.shape) == (2, 1, 512)


@pytest.mark.unit
def test_build_loss_defaults_to_mse():
    torch = pytest.importorskip("torch")
    loss = build_loss("mse")
    assert isinstance(loss, torch.nn.MSELoss)
    assert isinstance(build_loss("l1"), torch.nn.L1Loss)
    assert isinstance(build_loss("smooth_l1"), torch.nn.SmoothL1Loss)


@pytest.mark.unit
def test_model_forward_then_backward_updates_gradients():
    torch = pytest.importorskip("torch")
    model = ViTSpectrogramInpainter(
        context_epochs=7,
        epoch_samples=512,
        embed_dim=96,
        depth=2,
        n_heads=4,
        mlp_ratio=2.0,
    )
    x = torch.randn(2, 7, 1, 512, requires_grad=False)
    y = model(x)
    loss = (y ** 2).mean()
    loss.backward()
    grad_norms = [p.grad.detach().norm().item() for p in model.parameters() if p.grad is not None]
    assert grad_norms, "no parameter received a gradient"
    assert max(grad_norms) > 0.0


@pytest.mark.unit
def test_model_traces_to_torchscript_cleanly():
    torch = pytest.importorskip("torch")
    model = ViTSpectrogramInpainter(
        context_epochs=7,
        epoch_samples=512,
        embed_dim=96,
        depth=2,
        n_heads=4,
        mlp_ratio=2.0,
    ).eval()
    example = torch.randn(1, 7, 1, 512)
    with torch.no_grad():
        original = model(example)
    scripted = torch.jit.trace(model, example)
    with torch.no_grad():
        traced = scripted(example)
    assert tuple(traced.shape) == tuple(original.shape)
    assert torch.allclose(original, traced, atol=1e-5)


class _StubBase:
    """Tiny stand-in for NPZContextArtifactDataset used by the dataset test."""

    sfreq = 2048.0

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        noisy = np.arange(7 * 2 * 4, dtype=np.float32).reshape(7, 2, 4) + idx
        target = noisy[3] * 0.5
        return noisy, target


@pytest.mark.unit
def test_channel_wise_dataset_expands_channels():
    dataset = ChannelWiseSpectrogramDataset(
        _StubBase(),
        context_epochs=7,
        demean_input=False,
        demean_target=False,
    )
    assert len(dataset) == 4
    noisy, target = dataset[1]
    assert noisy.shape == (7, 1, 4)
    assert target.shape == (1, 4)
    assert dataset.input_shape == (7, 1, 4)
    assert dataset.target_shape == (1, 4)


@pytest.mark.unit
def test_correction_subtracts_artifact_from_center_epochs(tmp_path):
    torch = pytest.importorskip("torch")
    mne = pytest.importorskip("mne")

    class ConstantCleanCenter(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((x.shape[0], 1, x.shape[-1]), dtype=x.dtype, device=x.device)

    checkpoint = tmp_path / "constant_clean.ts"
    scripted = torch.jit.trace(ConstantCleanCenter(), torch.zeros(1, 7, 1, 8))
    scripted.save(str(checkpoint))

    data = np.ones((2, 80), dtype=np.float64) * 1.0
    info = mne.create_info(["C3", "C4"], sfreq=1000.0, ch_types="eeg")
    raw = mne.io.RawArray(data.copy(), info, verbose=False)
    metadata = ProcessingMetadata(triggers=np.arange(0, 80, 8, dtype=np.int32))
    context = ProcessingContext(raw=raw, metadata=metadata)

    result = context | ViTSpectrogramInpainterCorrection(
        checkpoint_path=checkpoint,
        context_epochs=7,
        epoch_samples=8,
        demean_input=False,
        remove_prediction_mean=False,
    )

    expected_noise = np.zeros_like(data)
    # ten triggers → nine epochs; with seven-context inference,
    # center epochs 3, 4, 5 are corrected (samples [24, 48))
    expected_noise[:, 24:48] = 1.0
    np.testing.assert_allclose(result.get_estimated_noise(), expected_noise)
    np.testing.assert_allclose(result.get_raw()._data, data - expected_noise)
    runs = result.metadata.custom["deep_learning_runs"]
    assert runs[-1]["model"] == "ViTSpectrogramInpainterAdapter"
    assert runs[-1]["prediction_metadata"]["context_epochs"] == 7


@pytest.mark.unit
def test_adapter_spec_has_expected_metadata():
    adapter = ViTSpectrogramInpainterAdapter.__new__(ViTSpectrogramInpainterAdapter)
    # bypass __init__ to inspect the class-level spec
    assert adapter.spec.architecture.value == "vision_transformer"
    assert adapter.spec.domain.value == "time_frequency"
    assert adapter.spec.uses_triggers is True
    assert adapter.spec.supports_multichannel is False
