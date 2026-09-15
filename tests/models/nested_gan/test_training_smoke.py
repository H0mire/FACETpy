from __future__ import annotations

import numpy as np
import pytest

from facet.models.nested_gan.training import (
    ChannelWiseContextArtifactDataset,
    NestedGANGenerator,
    NestedGANLoss,
    build_dataset,
    build_loss,
    build_model,
)


@pytest.mark.unit
def test_build_model_with_input_shape_returns_generator():
    torch = pytest.importorskip("torch")
    model = build_model(
        input_shape=(7, 1, 64),
        inner_channels=16,
        inner_blocks=1,
        inner_heads=2,
        outer_base_channels=8,
        n_fft=32,
        hop_length=8,
        win_length=32,
    )
    assert isinstance(model, NestedGANGenerator)
    x = torch.randn(2, 7, 1, 64)
    y = model(x)
    assert tuple(y.shape) == (2, 1, 64)


@pytest.mark.unit
def test_generator_forward_backward_updates_gradients():
    torch = pytest.importorskip("torch")
    model = build_model(
        input_shape=(7, 1, 64),
        inner_channels=16,
        inner_blocks=1,
        inner_heads=2,
        outer_base_channels=8,
        n_fft=32,
        hop_length=8,
        win_length=32,
    )
    loss_fn = build_loss(lambda_time=1.0, lambda_mrstft=0.5, fft_sizes=[16, 32], hop_fraction=0.25)
    x = torch.randn(2, 7, 1, 64, requires_grad=False)
    target = torch.randn(2, 1, 64)

    pred = model(x)
    loss = loss_fn(pred, target)
    assert torch.isfinite(loss)
    loss.backward()

    has_grad = [p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad]
    assert all(has_grad)
    assert any(p.grad.abs().sum() > 0 for p in model.parameters() if p.grad is not None)


@pytest.mark.unit
def test_nested_gan_loss_is_non_negative_scalar():
    torch = pytest.importorskip("torch")
    loss_fn = NestedGANLoss(lambda_time=1.0, lambda_mrstft=0.5, fft_sizes=(16, 32), hop_fraction=0.25)
    pred = torch.randn(3, 1, 64)
    target = pred + 0.01 * torch.randn_like(pred)
    value = loss_fn(pred, target)
    assert value.dim() == 0
    assert float(value) >= 0.0


@pytest.mark.unit
def test_outer_branch_receives_inner_corrected_center():
    torch = pytest.importorskip("torch")
    model = build_model(
        input_shape=(7, 1, 64),
        inner_channels=8,
        inner_blocks=1,
        inner_heads=2,
        outer_base_channels=4,
        n_fft=32,
        hop_length=8,
        win_length=32,
    )

    class CaptureOuter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, x):
            self.seen = x.detach().clone()
            return torch.zeros((x.shape[0], 1, x.shape[-1]), dtype=x.dtype, device=x.device)

    class ConstantInner(torch.nn.Module):
        target_samples = 64

        def forward(self, center):
            return torch.full_like(center, 0.5)

    model.inner = ConstantInner()
    model.outer = CaptureOuter()

    x = torch.ones(2, 7, 1, 64) * 2.0
    out = model(x)

    assert model.outer.seen is not None
    expected = x.squeeze(2).clone()
    expected[:, 3, :] = 2.0 - 0.5
    torch.testing.assert_close(model.outer.seen, expected)
    # Inner contributes 0.5 everywhere in the center slot; outer returns zeros.
    torch.testing.assert_close(out, torch.full((2, 1, 64), 0.5))


class _BaseDataset:
    sfreq = 5000.0

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        noisy = np.arange(7 * 3 * 8, dtype=np.float32).reshape(7, 3, 8) + idx
        target = noisy[3] * 0.5
        return noisy, target


@pytest.mark.unit
def test_channel_wise_dataset_expands_examples_and_demeans():
    dataset = ChannelWiseContextArtifactDataset(
        _BaseDataset(),
        context_epochs=7,
        demean_input=True,
        demean_target=True,
    )
    assert len(dataset) == 6
    noisy, target = dataset[2]
    assert noisy.shape == (7, 1, 8)
    assert target.shape == (1, 8)
    np.testing.assert_allclose(noisy.mean(axis=-1), 0.0, atol=1e-5)
    np.testing.assert_allclose(target.mean(axis=-1), 0.0, atol=1e-5)


@pytest.mark.unit
def test_build_dataset_loads_npz_bundle(tmp_path):
    path = tmp_path / "context.npz"
    rng = np.random.default_rng(0)
    np.savez(
        path,
        noisy_context=rng.standard_normal((4, 7, 3, 8)).astype(np.float32),
        artifact_center=rng.standard_normal((4, 3, 8)).astype(np.float32),
        sfreq=np.asarray([5000.0]),
    )

    dataset = build_dataset(path=str(path), context_epochs=7, demean_input=False, demean_target=False)
    assert dataset.input_shape == (7, 1, 8)
    assert dataset.target_shape == (1, 8)
    assert len(dataset) == 4 * 3
