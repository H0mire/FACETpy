"""Unit tests for the Demucs model, dataset wrapper, and adapter."""

from __future__ import annotations

import numpy as np
import pytest

from facet.models.demucs.training import (
    Demucs,
    FlatContextArtifactDataset,
    build_dataset,
    build_loss,
    build_model,
)


def test_build_model_forward_matches_input_shape():
    torch = pytest.importorskip("torch")

    model = build_model(input_shape=(1, 3584), depth=4, initial_channels=32)
    x = torch.randn(2, 1, 3584)
    y = model(x)

    assert tuple(y.shape) == (2, 1, 3584)


def test_build_model_smaller_depth_for_short_input():
    torch = pytest.importorskip("torch")

    model = build_model(input_shape=(1, 256), depth=2, initial_channels=16)
    x = torch.randn(1, 1, 256)
    y = model(x)

    assert tuple(y.shape) == (1, 1, 256)


def test_build_model_depth_collapse_raises():
    with pytest.raises(ValueError, match="collapses input length"):
        build_model(input_shape=(1, 64), depth=6, initial_channels=8)


def test_demucs_one_step_backward_updates_gradients():
    torch = pytest.importorskip("torch")

    model = Demucs(in_channels=1, out_channels=1, depth=2, initial_channels=16, lstm_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(4, 1, 256)
    target = torch.randn(4, 1, 256)

    optimizer.zero_grad()
    output = model(x)
    loss = build_loss("l1")(output, target)
    loss.backward()

    grads_present = [p for p in model.parameters() if p.grad is not None]
    assert grads_present, "no parameters received gradients"
    assert all(torch.isfinite(p.grad).all() for p in grads_present)


def test_build_loss_returns_expected_classes():
    torch = pytest.importorskip("torch")

    assert isinstance(build_loss("l1"), torch.nn.L1Loss)
    assert isinstance(build_loss("mse"), torch.nn.MSELoss)
    assert isinstance(build_loss("smooth_l1"), torch.nn.SmoothL1Loss)
    assert isinstance(build_loss("huber"), torch.nn.SmoothL1Loss)
    assert isinstance(build_loss("anything_else"), torch.nn.L1Loss)


def _write_synthetic_context_npz(path, n_examples=4, context_epochs=7, n_channels=3, samples=32, seed=0):
    rng = np.random.default_rng(seed)
    noisy = rng.standard_normal((n_examples, context_epochs, n_channels, samples)).astype(np.float32)
    artifact = noisy * 0.5
    np.savez(
        path,
        noisy_context=noisy,
        artifact_context=artifact,
        sfreq=np.array([2048.0]),
    )


def test_flat_context_artifact_dataset_shapes(tmp_path):
    path = tmp_path / "bundle.npz"
    _write_synthetic_context_npz(path, n_examples=2, context_epochs=7, n_channels=4, samples=16)

    dataset = FlatContextArtifactDataset(
        path,
        context_epochs=7,
        demean_input=False,
        demean_target=False,
    )

    assert len(dataset) == 2 * 4
    assert dataset.input_shape == (1, 7 * 16)
    assert dataset.target_shape == (1, 7 * 16)

    noisy, target = dataset[0]
    assert noisy.shape == (1, 7 * 16)
    assert target.shape == (1, 7 * 16)
    # First example, first channel — flattening should produce the 7×16 concatenation.
    expected = dataset._noisy[0, :, 0, :].reshape(1, -1)
    np.testing.assert_array_equal(noisy, expected)


def test_flat_context_artifact_dataset_demean(tmp_path):
    path = tmp_path / "bundle.npz"
    _write_synthetic_context_npz(path, n_examples=1, context_epochs=7, n_channels=1, samples=8)

    dataset = FlatContextArtifactDataset(
        path,
        context_epochs=7,
        demean_input=True,
        demean_target=True,
    )
    noisy, target = dataset[0]

    np.testing.assert_allclose(noisy.mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(target.mean(), 0.0, atol=1e-6)


def test_flat_context_dataset_split_disjoint(tmp_path):
    path = tmp_path / "bundle.npz"
    _write_synthetic_context_npz(path, n_examples=5, context_epochs=7, n_channels=2, samples=8)
    dataset = FlatContextArtifactDataset(path, context_epochs=7, demean_input=False, demean_target=False)

    train, val = dataset.train_val_split(val_ratio=0.4, seed=0)

    assert len(train) + len(val) == len(dataset)
    train_indices = set(train._indices)
    val_indices = set(val._indices)
    assert train_indices.isdisjoint(val_indices)


def test_build_dataset_round_trip(tmp_path):
    path = tmp_path / "bundle.npz"
    _write_synthetic_context_npz(path, n_examples=2, context_epochs=7, n_channels=2, samples=8)

    dataset = build_dataset(path=str(path), context_epochs=7, demean_input=False, demean_target=False)

    assert isinstance(dataset, FlatContextArtifactDataset)
    assert len(dataset) == 2 * 2


def test_demucs_correction_registered_name():
    from facet.core import get_processor

    cls = get_processor("demucs_correction")
    from facet.models.demucs import DemucsCorrection

    assert cls is DemucsCorrection
