"""CPU smoke test for the paper-accurate SepFormer edition.

Every architectural dimension is shrunk so forward + backward completes in a
few seconds on a CPU. Mirrors the original sepformer smoke style.
"""

from __future__ import annotations

import numpy as np
import pytest

from facet.models.sepformer_paper_accurate_edition.training import (
    build_dataset,
    build_loss,
    build_model,
)

# Tiny architecture used throughout the smoke (see smoke_plan / smoke YAML).
SMOKE_MODEL_KWARGS = dict(
    epoch_samples=32,
    context_epochs=3,
    encoder_channels=8,
    encoder_kernel=8,
    encoder_stride=4,
    chunk_size=8,
    n_blocks=1,
    intra_layers=1,
    inter_layers=1,
    intra_heads=2,
    inter_heads=2,
    d_ffn=16,
    dropout=0.0,
)


def _make_bundle(tmp_path, n_examples=6, n_channels=2, n_samples=32):
    rng = np.random.default_rng(0)
    clean = rng.standard_normal((n_examples, n_channels, n_samples)).astype(np.float32)
    artifact = rng.standard_normal((n_examples, n_channels, n_samples)).astype(np.float32)
    noisy = (clean + artifact).astype(np.float32)
    path = tmp_path / "proof_fit_smoke.npz"
    np.savez(
        path,
        noisy_center=noisy,
        clean_center=clean,
        artifact_center=artifact,
        sfreq=np.array([256.0], dtype=np.float64),
    )
    return path


def test_si_snr_loss_clips_at_30db():
    torch = pytest.importorskip("torch")
    loss = build_loss("si_snr")
    target = torch.randn(2, 1, 32)
    perfect = target.clone()
    # A perfect prediction would be +inf dB; with the 30 dB clip the negative
    # SI-SNR loss bottoms out near -30 (not -inf / very negative).
    value = float(loss(perfect, target))
    assert np.isfinite(value)
    assert -30.5 <= value <= -29.5


def test_si_snr_mse_loss_finite_on_random_pair():
    torch = pytest.importorskip("torch")
    loss = build_loss("si_snr_mse", mse_weight=0.1)
    prediction = torch.randn(2, 1, 32)
    target = torch.randn(2, 1, 32)
    assert np.isfinite(float(loss(prediction, target)))


def test_build_dataset_contract(tmp_path):
    pytest.importorskip("torch")
    path = _make_bundle(tmp_path)
    dataset = build_dataset(path=str(path), context_epochs=3)

    assert dataset.n_channels == 2
    assert dataset.epoch_samples == 32
    assert dataset.target_type == "artifact"
    assert dataset.trigger_aligned is True
    assert dataset.sfreq == pytest.approx(256.0)
    assert dataset.input_shape == (3, 1, 32)
    assert dataset.target_shape == (1, 32)
    # 6 examples, radius 1 -> 4 centres, x2 channels = 8 examples.
    assert len(dataset) == 8
    assert dataset.n_chunks == len(dataset)

    noisy, target = dataset[0]
    assert noisy.shape == (3, 1, 32)
    assert target.shape == (1, 32)
    assert noisy.dtype == np.float32 and target.dtype == np.float32

    train, val = dataset.train_val_split(val_ratio=0.25, seed=1)
    assert len(train) + len(val) == len(dataset)
    assert len(val) >= 1


def test_build_dataset_clean_target(tmp_path):
    pytest.importorskip("torch")
    path = _make_bundle(tmp_path)
    dataset = build_dataset(path=str(path), context_epochs=3, target_type="clean")
    assert dataset.target_type == "clean"
    _, target = dataset[0]
    assert target.shape == (1, 32)


def test_forward_shape(tmp_path):
    torch = pytest.importorskip("torch")
    model = build_model(input_shape=(3, 1, 32), **SMOKE_MODEL_KWARGS)
    model.eval()
    out = model(torch.randn(4, 3, 1, 32))
    assert tuple(out.shape) == (4, 1, 32)


def test_torchscript_trace_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    model = build_model(**SMOKE_MODEL_KWARGS)
    model.eval()
    scripted = torch.jit.trace(model, torch.zeros(1, 3, 1, 32))
    path = tmp_path / "sepformer_pa_smoke.ts"
    scripted.save(str(path))
    loaded = torch.jit.load(str(path), map_location="cpu")
    out = loaded(torch.randn(2, 3, 1, 32))
    assert tuple(out.shape) == (2, 1, 32)


def test_optimizer_reduces_loss():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = build_model(**SMOKE_MODEL_KWARGS)
    model.train()
    loss_fn = build_loss("mse")  # MSE anchors amplitude; clean overfit signal

    inputs = torch.randn(4, 3, 1, 32)
    target = torch.randn(4, 1, 32)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    initial_loss = float(loss_fn(model(inputs), target))
    for _ in range(5):
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), target)
        loss.backward()
        optimizer.step()
    final_loss = float(loss_fn(model(inputs), target))
    assert final_loss < initial_loss
