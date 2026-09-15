"""CPU smoke tests for the paper-accurate D4PM training factories.

Everything uses tiny dims so forward+backward finish in a few seconds on CPU.
Iterative inference sampling is NOT exercised (it is slow); the ancestral
sampler is covered structurally by the forward/backward path here.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from facet.models.d4pm_paper_accurate_edition.training import (
    D4PMArtifactDataset,
    D4PMEpsilonLoss,
    D4PMTrainingModule,
    build_dataset,
    build_loss,
    build_model,
)

_TINY = dict(epoch_samples=64, num_steps=8, feats=8, d_model=16, d_ff=32, n_heads=2, n_layers=1, embed_dim=16)


@pytest.mark.unit
def test_build_model_returns_module_with_expected_buffers():
    model = build_model(dual_branch=False, **_TINY)
    assert isinstance(model, torch.nn.Module)
    assert isinstance(model, D4PMTrainingModule)
    assert model.num_steps == 8
    assert model.alphas_cumprod.shape == (8,)
    assert model.sqrt_alphas_cumprod_prev.shape == (8,)
    # posterior buffers used by the ancestral sampler must exist
    assert model.posterior_mean_coef1.shape == (8,)
    assert model.posterior_mean_coef2.shape == (8,)
    assert torch.all(model.sqrt_alphas_cumprod > 0)


@pytest.mark.unit
def test_continuous_noise_level_within_interval():
    model = build_model(dual_branch=False, **_TINY)
    t = torch.tensor([3])
    u0 = torch.tensor([0.0])
    u1 = torch.tensor([1.0])
    # u=0 -> sqrt(abar_{t-1}); u=1 -> sqrt(abar_t). Since abar decreases with t,
    # the prev endpoint is numerically the larger of the two.
    end0, _ = model.continuous_noise_level(t, u0)
    end1, _ = model.continuous_noise_level(t, u1)
    assert torch.allclose(end0, model.sqrt_alphas_cumprod_prev[t])
    assert torch.allclose(end1, model.sqrt_alphas_cumprod[t])
    # midpoint lies inside the interval regardless of endpoint ordering
    mid, _ = model.continuous_noise_level(t, torch.tensor([0.5]))
    lo = torch.minimum(end0, end1)
    hi = torch.maximum(end0, end1)
    assert (mid >= lo).all() and (mid <= hi).all()


@pytest.mark.unit
def test_forward_shape_single_branch():
    model = build_model(dual_branch=False, **_TINY)
    packed = torch.randn(4, 2, 64)
    out = model(packed)
    assert out.shape == (4, 2, 64)


@pytest.mark.unit
def test_forward_shape_dual_branch():
    model = build_model(dual_branch=True, **_TINY)
    packed = torch.randn(4, 3, 64)
    out = model(packed)
    assert out.shape == (4, 4, 64)


@pytest.mark.unit
def test_optimizer_reduces_loss_single_branch():
    torch.manual_seed(0)
    model = build_model(dual_branch=False, **_TINY)
    loss_fn = build_loss(name="l1")
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    packed = torch.randn(4, 2, 64)
    target = torch.zeros(4, 1, 64)

    initial = None
    final = None
    for step in range(5):
        optim.zero_grad()
        pred = model(packed)
        loss = loss_fn(pred, target)
        loss.backward()
        optim.step()
        if step == 0:
            initial = float(loss.item())
        final = float(loss.item())
    assert final < initial


@pytest.mark.unit
def test_optimizer_reduces_loss_dual_branch():
    torch.manual_seed(0)
    model = build_model(dual_branch=True, **_TINY)
    loss_fn = build_loss(name="l1")
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    packed = torch.randn(4, 3, 64)
    target = torch.zeros(4, 1, 64)

    initial = None
    final = None
    for step in range(5):
        optim.zero_grad()
        pred = model(packed)
        loss = loss_fn(pred, target)
        loss.backward()
        optim.step()
        if step == 0:
            initial = float(loss.item())
        final = float(loss.item())
    assert final < initial


@pytest.mark.unit
def test_loss_zero_when_predictions_match_truth():
    loss_fn = D4PMEpsilonLoss(kind="l1")
    eps = torch.randn(2, 1, 64)
    single = torch.cat([eps, eps], dim=1)
    assert torch.isclose(loss_fn(single, torch.zeros(2, 1, 64)), torch.tensor(0.0), atol=1e-6)
    dual = torch.cat([eps, eps, eps, eps], dim=1)
    assert torch.isclose(loss_fn(dual, torch.zeros(2, 1, 64)), torch.tensor(0.0), atol=1e-6)


@pytest.mark.unit
def test_dataset_single_branch(tmp_path):
    npz_path = tmp_path / "fake.npz"
    rng = np.random.default_rng(0)
    noisy = rng.standard_normal((6, 3, 64)).astype(np.float32)
    artifact = rng.standard_normal((6, 3, 64)).astype(np.float32)
    clean = (noisy - artifact).astype(np.float32)
    np.savez(npz_path, noisy_center=noisy, clean_center=clean, artifact_center=artifact, sfreq=np.array([4096.0]))

    dataset = build_dataset(path=str(npz_path), dual_branch=False)
    assert isinstance(dataset, D4PMArtifactDataset)
    assert len(dataset) == 6 * 3
    assert dataset.n_channels == 3
    assert dataset.epoch_samples == 64
    assert dataset.input_shape == (2, 64)
    assert dataset.target_shape == (1, 64)
    assert dataset.target_type == "artifact"
    assert dataset.trigger_aligned is True
    assert dataset.sfreq == 4096.0
    assert dataset.n_chunks == len(dataset)

    packed, dummy = dataset[0]
    assert packed.shape == (2, 64)
    assert dummy.shape == (1, 64)
    assert np.isclose(packed[0].mean(), 0.0, atol=1e-5)

    train, val = dataset.train_val_split(val_ratio=0.2, seed=1)
    assert len(train) + len(val) == len(dataset)
    assert len(val) >= 1


@pytest.mark.unit
def test_dataset_dual_branch_requires_clean(tmp_path):
    npz_path = tmp_path / "fake_dual.npz"
    rng = np.random.default_rng(1)
    noisy = rng.standard_normal((5, 2, 64)).astype(np.float32)
    artifact = rng.standard_normal((5, 2, 64)).astype(np.float32)
    clean = (noisy - artifact).astype(np.float32)
    np.savez(npz_path, noisy_center=noisy, clean_center=clean, artifact_center=artifact, sfreq=np.array([256.0]))

    dataset = build_dataset(path=str(npz_path), dual_branch=True)
    assert dataset.input_shape == (3, 64)
    packed, _ = dataset[0]
    assert packed.shape == (3, 64)

    # Missing clean_center must raise for dual_branch.
    bad_path = tmp_path / "bad.npz"
    np.savez(bad_path, noisy_center=noisy, artifact_center=artifact, sfreq=np.array([256.0]))
    with pytest.raises(KeyError):
        D4PMArtifactDataset(path=bad_path, dual_branch=True)


@pytest.mark.unit
def test_processor_registers_unique_name():
    # Importing the processor module must register the UNIQUE name without
    # colliding with the original 'd4pm_correction'.
    from facet.core import get_processor

    from facet.models.d4pm_paper_accurate_edition.processor import D4PMPaperAccurateCorrection

    assert D4PMPaperAccurateCorrection.name == "d4pm_paper_accurate_correction"
    assert get_processor("d4pm_paper_accurate_correction") is D4PMPaperAccurateCorrection
