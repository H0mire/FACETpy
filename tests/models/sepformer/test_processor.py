from __future__ import annotations

import numpy as np
import pytest

from facet.core import ProcessingContext, ProcessingMetadata
from facet.models.sepformer import SepFormerArtifactCorrection
from facet.models.sepformer.training import (
    ChannelWiseContextArtifactDataset,
    SepFormerArtifactNet,
    build_loss,
    build_model,
)


def test_sepformer_factory_returns_module():
    torch = pytest.importorskip("torch")
    model = build_model(
        epoch_samples=64,
        context_epochs=7,
        encoder_channels=16,
        encoder_kernel=8,
        encoder_stride=4,
        chunk_size=16,
        n_blocks=1,
        intra_layers=1,
        inter_layers=1,
        intra_heads=4,
        inter_heads=4,
        d_ffn=32,
        dropout=0.0,
    )
    assert isinstance(model, torch.nn.Module)
    assert isinstance(model, SepFormerArtifactNet)


def test_sepformer_forward_output_shape():
    torch = pytest.importorskip("torch")
    model = build_model(
        epoch_samples=64,
        context_epochs=7,
        encoder_channels=16,
        encoder_kernel=8,
        encoder_stride=4,
        chunk_size=16,
        n_blocks=1,
        intra_layers=1,
        inter_layers=1,
        intra_heads=4,
        inter_heads=4,
        d_ffn=32,
        dropout=0.0,
    )
    x = torch.randn(2, 7, 1, 64)
    y = model(x)
    assert tuple(y.shape) == (2, 1, 64)


def test_sepformer_one_batch_backward_updates_gradients():
    torch = pytest.importorskip("torch")
    model = build_model(
        epoch_samples=64,
        context_epochs=7,
        encoder_channels=16,
        encoder_kernel=8,
        encoder_stride=4,
        chunk_size=16,
        n_blocks=1,
        intra_layers=1,
        inter_layers=1,
        intra_heads=4,
        inter_heads=4,
        d_ffn=32,
        dropout=0.0,
    )
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = build_loss("mse")
    before = [p.detach().clone() for p in model.parameters()]
    x = torch.randn(2, 7, 1, 64)
    target = torch.randn(2, 1, 64)
    loss = loss_fn(model(x), target)
    optim.zero_grad()
    loss.backward()
    optim.step()
    assert any(not torch.equal(p, q) for p, q in zip(model.parameters(), before))


def test_sepformer_torchscript_roundtrip_preserves_shape(tmp_path):
    torch = pytest.importorskip("torch")
    model = build_model(
        epoch_samples=64,
        context_epochs=7,
        encoder_channels=16,
        encoder_kernel=8,
        encoder_stride=4,
        chunk_size=16,
        n_blocks=1,
        intra_layers=1,
        inter_layers=1,
        intra_heads=4,
        inter_heads=4,
        d_ffn=32,
        dropout=0.0,
    )
    model.eval()
    scripted = torch.jit.trace(model, torch.zeros(1, 7, 1, 64))
    path = tmp_path / "smoke.ts"
    scripted.save(str(path))
    loaded = torch.jit.load(str(path), map_location="cpu")
    out = loaded(torch.randn(3, 7, 1, 64))
    assert tuple(out.shape) == (3, 1, 64)


class _ContextBaseDataset:
    sfreq = 4096.0

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        noisy = np.arange(7 * 3 * 8, dtype=np.float32).reshape(7, 3, 8) + idx
        target = noisy[3] * 0.5
        return noisy, target


def test_channel_wise_dataset_expands_channels():
    dataset = ChannelWiseContextArtifactDataset(
        _ContextBaseDataset(),
        context_epochs=7,
        demean_input=False,
        demean_target=False,
    )
    assert len(dataset) == 6  # 2 base examples × 3 channels
    noisy, target = dataset[1]
    assert noisy.shape == (7, 1, 8)
    assert target.shape == (1, 8)


def test_sepformer_correction_applies_center_epochs(tmp_path):
    torch = pytest.importorskip("torch")
    mne = pytest.importorskip("mne")

    class ConstantCenterArtifact(torch.nn.Module):
        def forward(self, x):
            return torch.ones((x.shape[0], 1, x.shape[-1]), dtype=x.dtype, device=x.device) * 0.25

    checkpoint = tmp_path / "constant_center_artifact.ts"
    scripted = torch.jit.trace(ConstantCenterArtifact(), torch.zeros(1, 7, 1, 8))
    scripted.save(str(checkpoint))

    data = np.ones((2, 80), dtype=np.float64)
    info = mne.create_info(["C3", "C4"], sfreq=1000.0, ch_types="eeg")
    raw = mne.io.RawArray(data.copy(), info, verbose=False)
    metadata = ProcessingMetadata(triggers=np.arange(0, 80, 8, dtype=np.int32))
    context = ProcessingContext(raw=raw, metadata=metadata)

    result = context | SepFormerArtifactCorrection(
        checkpoint_path=checkpoint,
        context_epochs=7,
        epoch_samples=8,
        demean_input=False,
        remove_prediction_mean=False,
    )

    expected_noise = np.zeros_like(data)
    # Same coverage as cascaded_context_dae: center epochs 3, 4, 5 → [24, 48).
    expected_noise[:, 24:48] = 0.25
    np.testing.assert_allclose(result.get_estimated_noise(), expected_noise)
    np.testing.assert_allclose(result.get_raw()._data, data - expected_noise)
    runs = result.metadata.custom["deep_learning_runs"]
    assert runs[-1]["model"] == "SepFormerArtifactAdapter"
    assert runs[-1]["prediction_metadata"]["context_epochs"] == 7
