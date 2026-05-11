"""Smoke tests for D4PM training factories."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from facet.models.d4pm.training import (
    D4PMArtifactDataset,
    D4PMEpsilonLoss,
    D4PMTrainingModule,
    build_loss,
    build_model,
)


@pytest.mark.unit
def test_build_model_returns_module_with_expected_buffers():
    model = build_model(epoch_samples=128, num_steps=20, feats=16, d_model=32, d_ff=64, n_layers=1, embed_dim=32)
    assert isinstance(model, torch.nn.Module)
    assert isinstance(model, D4PMTrainingModule)
    assert model.num_steps == 20
    assert model.alphas_cumprod.shape == (20,)
    assert torch.all(model.sqrt_alphas_cumprod > 0)


@pytest.mark.unit
def test_forward_returns_packed_pred_and_true_noise():
    model = build_model(epoch_samples=128, num_steps=20, feats=16, d_model=32, d_ff=64, n_layers=1, embed_dim=32)
    packed = torch.randn(4, 2, 128)
    out = model(packed)
    assert out.shape == (4, 2, 128)


@pytest.mark.unit
def test_one_batch_backward_updates_gradients():
    model = build_model(epoch_samples=128, num_steps=20, feats=16, d_model=32, d_ff=64, n_layers=1, embed_dim=32)
    loss_fn = build_loss(kind="l1")
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    packed = torch.randn(4, 2, 128)
    target = torch.zeros(4, 1, 128)
    pred = model(packed)
    loss = loss_fn(pred, target)
    loss.backward()

    grads_present = [p.grad is not None and torch.any(p.grad != 0) for p in model.parameters()]
    assert any(grads_present)
    optim.step()


@pytest.mark.unit
def test_dataset_packs_pairs(tmp_path):
    npz_path = tmp_path / "fake.npz"
    rng = np.random.default_rng(0)
    noisy = rng.standard_normal((6, 4, 128)).astype(np.float32)
    artifact = rng.standard_normal((6, 4, 128)).astype(np.float32)
    np.savez(npz_path, noisy_center=noisy, artifact_center=artifact, sfreq=np.array([4096.0]))

    dataset = D4PMArtifactDataset(path=npz_path, demean_input=True, demean_target=True)
    assert len(dataset) == 6 * 4
    assert dataset.epoch_samples == 128
    assert dataset.n_channels == 4

    packed, dummy = dataset[0]
    assert packed.shape == (2, 128)
    assert dummy.shape == (1, 128)
    assert np.isclose(packed[0].mean(), 0.0, atol=1e-5)
    assert np.isclose(packed[1].mean(), 0.0, atol=1e-5)


@pytest.mark.unit
def test_loss_zero_when_predictions_match_truth():
    loss_fn = D4PMEpsilonLoss(kind="l1")
    pred_noise = torch.randn(2, 1, 128)
    packed = torch.cat([pred_noise, pred_noise], dim=1)
    loss = loss_fn(packed, torch.zeros(2, 1, 128))
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
