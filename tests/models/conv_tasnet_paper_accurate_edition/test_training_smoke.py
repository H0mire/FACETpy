"""Cheap CPU smoke test for the paper-accurate Conv-TasNet edition.

Wires the model, dataset, and loss together end-to-end on tiny synthetic data.
Everything is shrunk so forward+backward over a few optimizer steps finishes in
well under a second on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from facet.models.conv_tasnet_paper_accurate_edition.training import (
    ChannelWiseSourceSeparationDataset,
    build_dataset,
    build_loss,
    build_model,
)


def _write_centers_npz(tmp_path, *, n_examples: int = 4, n_channels: int = 2, n_samples: int = 128):
    rng = np.random.default_rng(123)
    clean = rng.standard_normal((n_examples, n_channels, n_samples)).astype(np.float32)
    artifact = rng.standard_normal((n_examples, n_channels, n_samples)).astype(np.float32)
    bundle = {
        "noisy_center": (clean + artifact).astype(np.float32),  # noisy = clean + artifact
        "clean_center": clean,
        "artifact_center": artifact,
        "sfreq": np.asarray([200.0], dtype=np.float64),
    }
    path = tmp_path / "centers.npz"
    np.savez_compressed(path, **bundle)
    return path


def _tiny_model(torch):
    return build_model(
        encoder_filters=16,
        encoder_kernel=16,
        bottleneck_channels=8,
        hidden_channels=16,
        skip_channels=8,
        block_kernel=3,
        n_blocks=3,
        n_repeats=1,
        mask_activation="sigmoid",
        encoder_activation="linear",
        chunk_size=128,
    )


def test_dataset_attributes_and_split(tmp_path):
    pytest.importorskip("torch")
    npz_path = _write_centers_npz(tmp_path, n_examples=4, n_channels=2, n_samples=128)
    dataset = build_dataset(path=str(npz_path))

    assert isinstance(dataset, ChannelWiseSourceSeparationDataset)
    assert dataset.n_channels == 2
    assert dataset.chunk_size == 128
    assert dataset.epoch_samples == 128
    assert dataset.input_shape == (1, 128)
    assert dataset.target_shape == (2, 128)
    assert dataset.target_type == "artifact"
    assert dataset.trigger_aligned is True
    assert dataset.sfreq == pytest.approx(200.0)
    assert len(dataset) == 4 * 2
    assert dataset.n_chunks == len(dataset)

    mixture, sources = dataset[0]
    assert mixture.shape == (1, 128)
    assert sources.shape == (2, 128)

    train, val = dataset.train_val_split(val_ratio=0.25, seed=7)
    assert len(train) + len(val) == len(dataset)
    assert len(val) >= 1


def test_forward_shape(tmp_path):
    torch = pytest.importorskip("torch")
    model = _tiny_model(torch)
    out = model(torch.randn(2, 1, 128))
    assert out.shape == (2, 2, 128)


def test_one_step_training_decreases_loss(tmp_path):
    torch = pytest.importorskip("torch")
    npz_path = _write_centers_npz(tmp_path, n_examples=4, n_channels=2, n_samples=128)
    dataset = ChannelWiseSourceSeparationDataset(path=npz_path, demean_input=False, demean_target=False)

    mixture_batch = []
    target_batch = []
    for idx in range(len(dataset)):
        mixture, sources = dataset[idx]
        mixture_batch.append(mixture)
        target_batch.append(sources)
    mixture_tensor = torch.from_numpy(np.stack(mixture_batch, axis=0))
    target_tensor = torch.from_numpy(np.stack(target_batch, axis=0))

    model = _tiny_model(torch)
    loss_fn = build_loss("mse")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    initial_loss = loss_fn(model(mixture_tensor), target_tensor).item()
    for _ in range(10):
        optimizer.zero_grad()
        loss = loss_fn(model(mixture_tensor), target_tensor)
        loss.backward()
        optimizer.step()
    final_loss = loss_fn(model(mixture_tensor), target_tensor).item()
    assert final_loss < initial_loss


def test_consistency_loss_runs(tmp_path):
    torch = pytest.importorskip("torch")
    model = _tiny_model(torch)
    loss_fn = build_loss("consistency_mse", consistency_weight=0.5)
    mixture = torch.randn(2, 1, 128)
    target = torch.randn(2, 2, 128)
    value = loss_fn(model(mixture), target)
    assert value.ndim == 0
    value.backward()
