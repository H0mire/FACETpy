"""Cheap CPU smoke test for the paper-accurate DHCT-GAN v2 edition.

Exercises the paper-faithful machinery (LSGAN + feature-matching + three
discriminators + two-mask gating + 8-block LSA) at tiny dims so forward+backward
completes in a few seconds on CPU. No GPU, no real dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from facet.models.dhct_gan_v2_paper_accurate_edition.training import (
    build_dataset,
    build_loss,
    build_model,
)


def _make_fixture_npz(path: Path, *, context: int = 7) -> None:
    rng = np.random.default_rng(7)
    n_examples, n_channels, samples = 4, 2, 64
    noisy_context = rng.standard_normal((n_examples, context, n_channels, samples)).astype(np.float32)
    center = noisy_context[:, context // 2]
    clean = rng.standard_normal((n_examples, n_channels, samples)).astype(np.float32) * 0.05
    artifact = center - clean
    np.savez(
        path,
        noisy_context=noisy_context,
        noisy_center=center,
        clean_center=clean,
        artifact_center=artifact,
        sfreq=np.array([128.0]),
    )


@pytest.mark.unit
def test_build_dataset_contract(tmp_path: Path) -> None:
    npz = tmp_path / "fixture.npz"
    _make_fixture_npz(npz)
    ds = build_dataset(path=str(npz), context_epochs=7, demean_input=True, demean_target=True)

    # 4 examples * 2 channels = 8 windows.
    assert len(ds) == 8
    assert ds.input_shape == (7, 64)
    assert ds.target_shape == (3, 64)
    assert ds.n_channels == 2
    assert ds.chunk_size == 64
    assert ds.target_type == "artifact"
    assert ds.trigger_aligned is True
    assert ds.sfreq == pytest.approx(128.0)
    assert ds.epoch_samples == 64

    inp, tgt = ds[0]
    assert inp.shape == (7, 64)
    assert tgt.shape == (3, 64)

    train_ds, val_ds = ds.train_val_split(val_ratio=0.25, seed=0)
    assert len(train_ds) + len(val_ds) == len(ds)


@pytest.mark.unit
def test_forward_shape() -> None:
    model = build_model(
        input_shape=(7, 64),
        base_channels=8,
        depth=2,
        num_heads=2,
        local_blocks=4,
        lgtb_depth=1,
        epoch_samples=64,
    )
    model.eval()
    out = model(torch.randn(2, 7, 64))
    # forward returns the artifact (noisy_center - fused_clean): (B, 1, T)
    assert out.shape == (2, 1, 64)

    # The internal outputs expose all branch signals for the loss.
    outs = model._compute_outputs(torch.randn(2, 7, 64))
    for key in ("artifact", "artifact_from_fused", "clean", "fused_clean", "mask1", "mask2"):
        assert outs[key].shape == (2, 1, 64)


@pytest.mark.unit
def test_training_smoke_loss_decreases() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    model = build_model(
        input_shape=(7, 64),
        base_channels=8,
        depth=2,
        num_heads=2,
        local_blocks=4,
        lgtb_depth=1,
        epoch_samples=64,
    )
    loss_fn = build_loss(
        recon="mse",
        alpha_consistency=0.5,
        lambda_feat=0.1,
        lambda_adv=0.1,
        disc_channels=4,
        disc_depth=3,
        disc_lr=1e-4,
    )

    # Paper-recommended generator Adam betas (0.5, 0.9).
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.5, 0.9))

    batch = 4
    # A learnable signal: artifact ~ scaled noisy center so the network can fit.
    noisy = torch.randn(batch, 7, 64)
    center = noisy[:, 3:4, :]
    clean = 0.05 * torch.randn(batch, 1, 64)
    artifact = center - clean
    target = torch.cat([artifact, clean, center], dim=1)

    model.train()

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        pred = model(noisy)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    initial = step()
    last = initial
    for _ in range(5):
        last = step()

    assert np.isfinite(initial)
    assert np.isfinite(last)
    assert last < initial, f"expected loss to decrease, got initial={initial} final={last}"

    # Confirm the three discriminators actually stepped (finite parameters).
    for disc in (loss_fn.disc_clean, loss_fn.disc_noise, loss_fn.disc_fused):
        for p in disc.parameters():
            assert torch.isfinite(p).all()
        # At least one param has a gradient from the internal disc step.
        assert any(p.grad is not None for p in disc.parameters())
