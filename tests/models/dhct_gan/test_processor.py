"""Smoke tests for the DHCT-GAN processor and adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from facet.models.dhct_gan import DHCTGanAdapter, DHCTGanCorrection
from facet.models.dhct_gan.training import (
    DHCTGanGenerator,
    DHCTGanLoss,
    PatchGANDiscriminator,
    build_dataset,
    build_loss,
    build_model,
)


@pytest.fixture(scope="module")
def epoch_samples() -> int:
    return 512


@pytest.fixture(scope="module")
def generator(epoch_samples: int) -> DHCTGanGenerator:
    return build_model(epoch_samples=epoch_samples, base_channels=8, depth=3)


def test_build_model_returns_module(generator: DHCTGanGenerator) -> None:
    assert isinstance(generator, torch.nn.Module)
    assert isinstance(generator, DHCTGanGenerator)


def test_forward_shape(generator: DHCTGanGenerator, epoch_samples: int) -> None:
    x = torch.randn(2, 1, epoch_samples)
    y = generator(x)
    assert y.shape == (2, 1, epoch_samples)


def test_forward_dict_outputs(generator: DHCTGanGenerator, epoch_samples: int) -> None:
    x = torch.randn(2, 1, epoch_samples)
    out = generator._compute_outputs(x)
    for key in ("artifact", "clean", "fused_clean", "gate"):
        assert out[key].shape == (2, 1, epoch_samples), key


def test_backward_updates_gradients(generator: DHCTGanGenerator, epoch_samples: int) -> None:
    x = torch.randn(4, 1, epoch_samples)
    artifact = torch.randn(4, 1, epoch_samples) * 0.1
    clean = torch.randn(4, 1, epoch_samples) * 0.1
    target = torch.cat([artifact, clean, x], dim=1)
    loss_fn = build_loss(alpha_consistency=0.5, beta_adv=0.0)

    pred = generator(x)
    loss = loss_fn(pred, target)
    loss.backward()
    assert loss.item() == pytest.approx(loss.item())  # finite
    grad_count = sum(int(p.grad is not None and torch.any(p.grad != 0)) for p in generator.parameters())
    assert grad_count > 0, "no generator parameters received gradient"


def test_loss_runs_under_no_grad(generator: DHCTGanGenerator, epoch_samples: int) -> None:
    loss_fn = build_loss(beta_adv=0.1)
    x = torch.randn(2, 1, epoch_samples)
    target = torch.cat([torch.randn_like(x), torch.randn_like(x), x], dim=1)
    with torch.no_grad():
        pred = generator(x)
        value = loss_fn(pred, target)
    assert torch.isfinite(value)


def test_discriminator_shape(epoch_samples: int) -> None:
    disc = PatchGANDiscriminator(in_channels=1, base_channels=8, depth=3)
    out = disc(torch.randn(2, 1, epoch_samples))
    assert out.ndim == 3
    assert out.shape[0] == 2
    assert out.shape[1] == 1


def test_dataset_factory(tmp_path: Path) -> None:
    # Build a small synthetic NPZ matching the dataset contract.
    n_examples, n_channels, samples = 4, 3, 64
    rng = np.random.default_rng(0)
    artifact = rng.standard_normal((n_examples, n_channels, samples)).astype(np.float32)
    clean = rng.standard_normal((n_examples, n_channels, samples)).astype(np.float32) * 0.1
    noisy = clean + artifact

    npz = tmp_path / "fixture.npz"
    np.savez(
        npz,
        noisy_center=noisy,
        clean_center=clean,
        artifact_center=artifact,
        sfreq=np.array([512.0]),
    )

    ds = build_dataset(path=str(npz), demean=False)
    assert len(ds) == n_examples * n_channels
    noisy_window, target_stack = ds[0]
    assert noisy_window.shape == (1, samples)
    assert target_stack.shape == (3, samples)


def test_adapter_predict_with_synthetic_raw(tmp_path: Path) -> None:
    # Train a tiny model and trace it
    model = build_model(epoch_samples=128, base_channels=8, depth=3)
    example = torch.randn(1, 1, 128)
    scripted = torch.jit.trace(model.eval(), example)
    checkpoint = tmp_path / "dhct_gan.ts"
    scripted.save(str(checkpoint))

    # Build a tiny ProcessingContext
    import mne
    from facet.core import ProcessingContext

    sfreq = 512.0
    n_channels = 2
    n_samples = 1024
    data = np.random.default_rng(1).standard_normal((n_channels, n_samples)).astype(np.float32)
    info = mne.create_info([f"EEG{i:02d}" for i in range(n_channels)], sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    triggers = np.array([100, 300, 500, 700, 900], dtype=int)
    context = ProcessingContext(raw=raw)
    context = context.with_triggers(triggers)

    adapter = DHCTGanAdapter(
        checkpoint_path=str(checkpoint),
        epoch_samples=128,
        device="cpu",
        artifact_to_trigger_offset=0.0,
        demean_input=True,
        remove_prediction_mean=True,
    )
    adapter.validate_context(context)
    prediction = adapter.predict(context)
    assert prediction.artifact_data is not None
    assert prediction.artifact_data.shape == data.shape
    assert prediction.metadata["corrected_epochs"] >= 1
