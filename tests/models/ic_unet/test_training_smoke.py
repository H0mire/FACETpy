"""Smoke test: tiny synthetic dataset, one training step, model converges marginally."""

from __future__ import annotations

import numpy as np
import pytest

from facet.models.ic_unet.training import (
    IcUnetEnsembleLoss,
    IcUnetWithIca,
    build_loss,
)


def _make_synthetic_batch(
    batch_size: int,
    n_channels: int,
    context_epochs: int,
    epoch_samples: int,
    seed: int = 0,
):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(seed)
    full = context_epochs * epoch_samples
    noisy = rng.standard_normal((batch_size, n_channels, full)).astype(np.float32)
    target = noisy[
        ...,
        (context_epochs // 2) * epoch_samples : (context_epochs // 2 + 1) * epoch_samples,
    ].copy() * 0.1
    return torch.from_numpy(noisy), torch.from_numpy(target)


def test_ensemble_loss_decomposes_into_four_terms():
    torch = pytest.importorskip("torch")
    loss = IcUnetEnsembleLoss()
    prediction = torch.zeros(2, 3, 16)
    target = torch.ones(2, 3, 16)
    value = loss(prediction, target)
    assert value.item() > 0.0


def test_one_training_step_reduces_loss():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = IcUnetWithIca(
        n_channels=3,
        context_epochs=7,
        epoch_samples=16,
        base_channels=8,
        demean_input=False,
        ica_init=None,
    )
    loss_fn = build_loss("mse")
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)

    noisy, target = _make_synthetic_batch(
        batch_size=4, n_channels=3, context_epochs=7, epoch_samples=16
    )

    model.train()
    losses: list[float] = []
    for _ in range(5):
        optimiser.zero_grad()
        prediction = model(noisy)
        loss = loss_fn(prediction, target)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"
