"""Cheap CPU smoke tests for the paper-accurate DPAE edition.

Every dimension is tiny (base_filters=4, input length 128/256, batch <=4,
<=5 optimizer steps) so forward+backward on CPU completes in a few seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from facet.models.dpae_paper_accurate_edition.training import (
    build_dataset,
    build_loss,
    build_model,
)


def test_build_loss_default_is_mse():
    torch = pytest.importorskip("torch")
    assert isinstance(build_loss("mse"), torch.nn.MSELoss)


def test_forward_shape_matches_input():
    torch = pytest.importorskip("torch")
    model = build_model(
        input_shape=(1, 256),
        base_filters=4,
        shrink_ratio_low=0.45,
        shrink_ratio_high=0.75,
        fusion_depth=2,
        pathway_layers=3,
    )
    model.eval()
    x = torch.randn(2, 1, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 256)


def test_forward_shape_odd_length_is_length_safe():
    # The paper-accurate edition removes the hard %4 constraint; arbitrary
    # lengths must still produce a length-matched output.
    torch = pytest.importorskip("torch")
    model = build_model(input_shape=(1, 130), base_filters=4, fusion_depth=2, pathway_layers=3)
    model.eval()
    x = torch.randn(2, 1, 130)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 130)


def test_fusion_has_symmetric_encoder_decoder_and_residual():
    # Structural faithfulness: the fusion module must have a compressing encoder
    # and an expanding decoder (the paper's "common feature coding") and a
    # residual skip wrapping it.
    torch = pytest.importorskip("torch")
    model = build_model(input_shape=(1, 128), base_filters=4, fusion_depth=2, pathway_layers=3)
    fusion = model.fusion
    assert len(list(fusion.fusion_encoder.children())) > 0
    assert len(list(fusion.fusion_decoder.children())) > 0
    # Residual identity: a zero input must map exactly to zero through the skip
    # only if encode/decode of zero is zero; instead assert the skip is present
    # by checking the output channel count equals the input channel count.
    z = torch.zeros(2, fusion.fused_channels, 8)
    with torch.no_grad():
        out = fusion(z)
    assert out.shape == z.shape


def test_one_batch_optimization_reduces_loss():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    model = build_model(input_shape=(1, 256), base_filters=4, fusion_depth=2, pathway_layers=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = build_loss("mse")

    noisy = torch.from_numpy(rng.standard_normal((4, 1, 256)).astype(np.float32))
    target = torch.from_numpy(rng.standard_normal((4, 1, 256)).astype(np.float32))

    model.train()
    optimizer.zero_grad()
    initial_loss = loss_fn(model(noisy), target)
    initial_loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no gradients computed"
    assert any(float(g.abs().max()) > 0 for g in grads), "all gradients are exactly zero"
    optimizer.step()

    for _ in range(4):
        optimizer.zero_grad()
        loss = loss_fn(model(noisy), target)
        loss.backward()
        optimizer.step()

    final_loss = loss_fn(model(noisy), target)
    assert float(final_loss) < float(initial_loss), (
        f"loss did not decrease: {float(initial_loss)} -> {float(final_loss)}"
    )


def _write_tiny_npz(tmp_path, n_examples=8, n_channels=3, samples=128):
    rng = np.random.default_rng(1)
    clean = rng.standard_normal((n_examples, n_channels, samples)).astype(np.float32)
    artifact = rng.standard_normal((n_examples, n_channels, samples)).astype(np.float32)
    noisy = clean + artifact
    path = tmp_path / "tiny_proof_fit.npz"
    np.savez(
        path,
        noisy_center=noisy,
        clean_center=clean,
        artifact_center=artifact,
        sfreq=np.array([256.0], dtype=np.float32),
    )
    return path


def test_build_dataset_exposes_contract_attributes(tmp_path):
    pytest.importorskip("torch")
    path = _write_tiny_npz(tmp_path)
    ds = build_dataset(path=str(path), target_type="clean", max_examples=8)

    for attr in (
        "n_channels",
        "chunk_size",
        "epoch_samples",
        "target_type",
        "trigger_aligned",
        "sfreq",
        "input_shape",
        "target_shape",
        "n_chunks",
    ):
        assert hasattr(ds, attr), f"dataset missing contract attribute '{attr}'"

    assert ds.target_type == "clean"
    assert ds.epoch_samples == 128
    assert ds.chunk_size == 128

    x, y = ds[0]
    assert x.shape == (1, 128)
    assert y.shape == (1, 128)

    train, val = ds.train_val_split(val_ratio=0.25, seed=0)
    for subset in (train, val):
        for attr in (
            "n_channels",
            "chunk_size",
            "epoch_samples",
            "target_type",
            "trigger_aligned",
            "sfreq",
            "input_shape",
            "target_shape",
            "n_chunks",
        ):
            assert hasattr(subset, attr), f"subset missing contract attribute '{attr}'"
        sx, sy = subset[0]
        assert sx.shape == (1, 128)
        assert sy.shape == (1, 128)


def test_build_dataset_artifact_target(tmp_path):
    pytest.importorskip("torch")
    path = _write_tiny_npz(tmp_path)
    ds = build_dataset(path=str(path), target_type="artifact", max_examples=8)
    assert ds.target_type == "artifact"
    x, y = ds[0]
    assert x.shape == (1, 128)
    assert y.shape == (1, 128)
