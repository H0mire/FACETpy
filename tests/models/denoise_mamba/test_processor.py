from __future__ import annotations

import numpy as np
import pytest

from facet.core import ProcessingContext, ProcessingMetadata
from facet.models.denoise_mamba import DenoiseMambaCorrection
from facet.models.denoise_mamba.training import (
    ChannelWiseSingleEpochArtifactDataset,
    DenoiseMamba,
    build_dataset,
    build_loss,
    build_model,
)


def test_build_model_returns_module_with_expected_io_shape():
    torch = pytest.importorskip("torch")
    model = build_model(epoch_samples=64, d_model=16, d_state=8, n_blocks=2, dropout=0.0)
    assert isinstance(model, DenoiseMamba)
    out = model(torch.zeros(2, 1, 64))
    assert tuple(out.shape) == (2, 1, 64)


def test_build_loss_defaults_to_mse():
    torch = pytest.importorskip("torch")
    loss_fn = build_loss("mse")
    pred = torch.zeros(3, 1, 8)
    target = torch.ones(3, 1, 8)
    assert pytest.approx(float(loss_fn(pred, target)), abs=1e-6) == 1.0


def test_denoise_mamba_forward_shape_and_dtype():
    torch = pytest.importorskip("torch")
    model = DenoiseMamba(epoch_samples=32, d_model=16, d_state=4, n_blocks=2, dropout=0.0)
    out = model(torch.randn(4, 1, 32))
    assert tuple(out.shape) == (4, 1, 32)
    assert out.dtype == torch.float32


def test_denoise_mamba_one_batch_backward_updates_gradients():
    torch = pytest.importorskip("torch")
    model = DenoiseMamba(epoch_samples=32, d_model=16, d_state=4, n_blocks=2, dropout=0.0)
    x = torch.randn(2, 1, 32)
    target = torch.zeros(2, 1, 32)
    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_denoise_mamba_rejects_multichannel_input():
    torch = pytest.importorskip("torch")
    model = DenoiseMamba(epoch_samples=16, d_model=8, d_state=4, n_blocks=1, dropout=0.0)
    with pytest.raises(ValueError):
        model(torch.zeros(1, 3, 16))


class _FakeNpzCenterAdapter:
    sfreq = 1024.0

    def __init__(self, n_examples: int = 3, n_channels: int = 2, samples: int = 8) -> None:
        rng = np.random.default_rng(0)
        self._noisy = rng.standard_normal((n_examples, n_channels, samples)).astype(np.float32)
        self._artifact = rng.standard_normal((n_examples, n_channels, samples)).astype(np.float32)

    def __len__(self) -> int:
        return int(self._noisy.shape[0])

    def __getitem__(self, idx: int):
        return self._noisy[idx], self._artifact[idx]


def test_channel_wise_dataset_expands_channels_per_example():
    base = _FakeNpzCenterAdapter(n_examples=3, n_channels=2, samples=8)
    dataset = ChannelWiseSingleEpochArtifactDataset(base, demean_input=False, demean_target=False)
    assert len(dataset) == 6
    noisy, target = dataset[3]
    assert noisy.shape == (1, 8)
    assert target.shape == (1, 8)
    np.testing.assert_allclose(noisy[0], base._noisy[1, 1])
    np.testing.assert_allclose(target[0], base._artifact[1, 1])


def test_build_dataset_consumes_real_npz_bundle(tmp_path):
    dataset_path = tmp_path / "fake_niazy.npz"
    n_examples, context_epochs, n_channels, samples = 4, 7, 2, 16
    rng = np.random.default_rng(0)
    np.savez_compressed(
        dataset_path,
        noisy_context=rng.standard_normal((n_examples, context_epochs, n_channels, samples)).astype(np.float32),
        artifact_center=rng.standard_normal((n_examples, n_channels, samples)).astype(np.float32),
        sfreq=np.asarray([1000.0], dtype=np.float64),
    )
    dataset = build_dataset(path=str(dataset_path), demean_input=True, demean_target=True)
    assert len(dataset) == n_examples * n_channels
    noisy, target = dataset[0]
    assert noisy.shape == (1, samples)
    assert target.shape == (1, samples)


def test_denoise_mamba_correction_applies_artifact_subtraction(tmp_path):
    torch = pytest.importorskip("torch")
    mne = pytest.importorskip("mne")

    class ConstantArtifact(torch.nn.Module):
        def forward(self, x):
            return torch.full_like(x, 0.5)

    checkpoint = tmp_path / "constant_denoise_mamba.ts"
    scripted = torch.jit.trace(ConstantArtifact(), torch.zeros(1, 1, 16))
    scripted.save(str(checkpoint))

    n_samples = 64
    data = np.full((2, n_samples), 1.0, dtype=np.float64)
    info = mne.create_info(["C3", "C4"], sfreq=1000.0, ch_types="eeg")
    raw = mne.io.RawArray(data.copy(), info, verbose=False)
    metadata = ProcessingMetadata()
    context = ProcessingContext(raw=raw, metadata=metadata)

    result = context | DenoiseMambaCorrection(
        checkpoint_path=checkpoint,
        chunk_size_samples=16,
        chunk_overlap_samples=0,
        demean_input=False,
        remove_prediction_mean=False,
    )

    expected_noise = np.full_like(data, 0.5)
    np.testing.assert_allclose(result.get_estimated_noise(), expected_noise)
    np.testing.assert_allclose(result.get_raw()._data, data - expected_noise)
    runs = result.metadata.custom["deep_learning_runs"]
    assert runs[-1]["model"] == "DenoiseMambaAdapter"
    chunks = runs[-1]["prediction_metadata"]["chunks"]
    assert len(chunks) == n_samples // 16
    assert chunks[0]["prediction_metadata"]["chunk_size_samples"] == 16
