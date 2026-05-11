"""Smoke test that wires the model, dataset, and loss together end-to-end."""

from __future__ import annotations

import numpy as np
import pytest

from facet.models.conv_tasnet.training import (
    ChannelWiseSourceSeparationDataset,
    build_loss,
    build_model,
)


def _write_centers_npz(tmp_path, *, n_examples: int = 8, n_channels: int = 2, n_samples: int = 32):
    rng = np.random.default_rng(123)
    bundle = {
        "noisy_center": rng.standard_normal((n_examples, n_channels, n_samples)).astype(np.float32),
        "clean_center": rng.standard_normal((n_examples, n_channels, n_samples)).astype(np.float32),
        "artifact_center": rng.standard_normal((n_examples, n_channels, n_samples)).astype(np.float32),
        "sfreq": np.asarray([5000.0], dtype=np.float64),
    }
    path = tmp_path / "centers.npz"
    np.savez_compressed(path, **bundle)
    return path


def test_one_step_training_decreases_loss(tmp_path):
    torch = pytest.importorskip("torch")
    npz_path = _write_centers_npz(tmp_path, n_examples=8, n_channels=2, n_samples=32)
    dataset = ChannelWiseSourceSeparationDataset(path=npz_path, demean_input=False, demean_target=False)

    mixture_batch = []
    target_batch = []
    for idx in range(len(dataset)):
        mixture, sources = dataset[idx]
        mixture_batch.append(mixture)
        target_batch.append(sources)
    mixture_tensor = torch.from_numpy(np.stack(mixture_batch, axis=0))
    target_tensor = torch.from_numpy(np.stack(target_batch, axis=0))

    model = build_model(
        encoder_filters=8,
        encoder_kernel=4,
        bottleneck_channels=4,
        hidden_channels=8,
        block_kernel=3,
        n_blocks=3,
        n_repeats=1,
    )
    loss_fn = build_loss("mse")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    initial_loss = loss_fn(model(mixture_tensor), target_tensor).item()
    for _ in range(5):
        optimizer.zero_grad()
        loss = loss_fn(model(mixture_tensor), target_tensor)
        loss.backward()
        optimizer.step()
    final_loss = loss_fn(model(mixture_tensor), target_tensor).item()
    assert final_loss < initial_loss
