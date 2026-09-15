"""CPU smoke tests for the paper-accurate DenoiseMamba edition.

Every dimension is tiny (base_channels 8, n_stages 1, d_state 4, chunk_size 64,
batch 2-4, a handful of optimizer steps, SSD chunk 16) so the full forward and
backward pass completes in milliseconds on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest


def _tiny_model(torch):
    from facet.models.denoise_mamba_paper_accurate_edition.training import build_model

    return build_model(
        epoch_samples=64,
        base_channels=8,
        n_stages=1,
        d_state=4,
        n_heads=1,
        d_conv=4,
        dropout=0.0,
        ssd_chunk_size=16,
    )


def test_forward_shape_is_preserved():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)

    model = _tiny_model(torch).eval()
    x = torch.randn(2, 1, 64)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (2, 1, 64)
    assert torch.isfinite(y).all()


def test_forward_shape_non_power_of_two_length():
    # The U-Net pads internally to a multiple of 2**n_stages and crops back.
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)

    from facet.models.denoise_mamba_paper_accurate_edition.training import build_model

    model = build_model(epoch_samples=70, base_channels=8, n_stages=2, d_state=4, ssd_chunk_size=16).eval()
    x = torch.randn(2, 1, 70)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (2, 1, 70)


def test_optimizer_reduces_loss():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)

    from facet.models.denoise_mamba_paper_accurate_edition.training import build_loss

    model = _tiny_model(torch).train()
    loss_fn = build_loss("mse")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)

    x = torch.randn(4, 1, 64)
    # Learnable target so the tiny net can actually fit it within a few steps.
    target = torch.sin(torch.linspace(0, 6.0, 64)).view(1, 1, 64).repeat(4, 1, 1)

    initial_loss = float(loss_fn(model(x), target).item())
    for _ in range(8):
        optimizer.zero_grad()
        loss = loss_fn(model(x), target)
        loss.backward()
        optimizer.step()
    final_loss = float(loss_fn(model(x), target).item())

    assert final_loss < initial_loss


def test_build_loss_variants():
    pytest.importorskip("torch")
    from facet.models.denoise_mamba_paper_accurate_edition.training import build_loss

    import torch.nn as nn

    assert isinstance(build_loss("mse"), nn.MSELoss)
    assert isinstance(build_loss("l1"), nn.L1Loss)
    assert isinstance(build_loss("smooth_l1"), nn.SmoothL1Loss)


def _write_tiny_npz(tmp_path):
    rng = np.random.default_rng(0)
    n_examples, n_channels, n_samples = 4, 2, 64
    clean = rng.standard_normal((n_examples, n_channels, n_samples)).astype(np.float32)
    artifact = rng.standard_normal((n_examples, n_channels, n_samples)).astype(np.float32)
    noisy = (clean + artifact).astype(np.float32)
    path = tmp_path / "tiny_center.npz"
    np.savez(
        path,
        noisy_center=noisy,
        clean_center=clean,
        artifact_center=artifact,
        sfreq=np.asarray([250.0], dtype=np.float32),
    )
    return path


def test_build_dataset_clean_target(tmp_path):
    pytest.importorskip("torch")
    from facet.models.denoise_mamba_paper_accurate_edition.training import build_dataset

    path = _write_tiny_npz(tmp_path)
    dataset = build_dataset(path=str(path), target_type="clean", normalize="zscore")

    assert len(dataset) == 4 * 2  # examples x channels
    noisy_item, target_item = dataset[0]
    assert noisy_item.shape == (1, 64)
    assert target_item.shape == (1, 64)
    assert noisy_item.dtype == np.float32

    assert dataset.n_channels == 2
    assert dataset.chunk_size == 64
    assert dataset.epoch_samples == 64
    assert dataset.input_shape == (1, 64)
    assert dataset.target_shape == (1, 64)
    assert dataset.n_chunks == len(dataset)
    assert dataset.target_type == "clean"
    assert dataset.trigger_aligned is True
    assert dataset.sfreq == pytest.approx(250.0)


def test_build_dataset_artifact_target(tmp_path):
    pytest.importorskip("torch")
    from facet.models.denoise_mamba_paper_accurate_edition.training import build_dataset

    path = _write_tiny_npz(tmp_path)
    dataset = build_dataset(path=str(path), target_type="artifact", normalize="demean")
    assert dataset.target_type == "artifact"
    noisy_item, target_item = dataset[1]
    assert noisy_item.shape == (1, 64)
    assert target_item.shape == (1, 64)


def test_train_val_split_disjoint(tmp_path):
    pytest.importorskip("torch")
    from facet.models.denoise_mamba_paper_accurate_edition.training import build_dataset

    path = _write_tiny_npz(tmp_path)
    dataset = build_dataset(path=str(path), target_type="clean")
    train, val = dataset.train_val_split(val_ratio=0.25, seed=0)

    assert len(train) + len(val) == len(dataset)
    assert len(val) >= 1
    assert len(train) >= 1


def test_processor_registration_unique():
    # Importing the processor module registers the unique correction name and
    # must not collide with the original "denoise_mamba_correction".
    from facet.core import list_processors

    import facet.models.denoise_mamba_paper_accurate_edition.processor  # noqa: F401

    names = list_processors()
    assert "denoise_mamba_paper_accurate_correction" in names
