"""CPU smoke test for the paper-accurate IC-U-Net edition.

Tiny dims (n_channels=4, base_channels=4, depth=2, context 3*32=96) so
forward+backward finishes in well under a second on CPU. Verifies:

- ``build_model`` builds a sensor-level (no in-graph ICA) U-Net,
- a few AdamW steps reduce the loss and produce a non-zero gradient,
- ``build_loss('mse')`` is ``MSELoss`` and the ensemble loss runs (z-scored
  1-50 Hz PSD path) and is finite,
- the forward output shape is ``(batch, n_channels, epoch_samples)`` for the
  clean head, and the artifact head matches too,
- ``build_dataset`` reads a tiny synthetic NPZ and exposes the required
  attributes.
"""

from __future__ import annotations

import numpy as np
import pytest

from facet.models.ic_unet_paper_accurate_edition.training import (
    IcUNetEnsembleLoss,
    IcUNetPaperAccurate,
    build_dataset,
    build_loss,
    build_model,
)

N_CHANNELS = 4
CONTEXT_EPOCHS = 3
EPOCH_SAMPLES = 32
FULL_SAMPLES = CONTEXT_EPOCHS * EPOCH_SAMPLES  # 96, a multiple of 2**depth
BASE_CHANNELS = 4
DEPTH = 2
BATCH = 4


def _tiny_model(output_type: str = "clean"):
    return build_model(
        input_shape=(N_CHANNELS, FULL_SAMPLES),
        context_epochs=CONTEXT_EPOCHS,
        epoch_samples=EPOCH_SAMPLES,
        base_channels=BASE_CHANNELS,
        depth=DEPTH,
        kernel_size=7,
        target_type=output_type,
    )


def test_build_model_is_sensor_level_no_in_graph_ica():
    pytest.importorskip("torch")
    model = _tiny_model()
    assert isinstance(model, IcUNetPaperAccurate)
    # Paper-faithful default: no frozen ICA buffers in the graph.
    assert not model.use_frozen_ica
    assert not hasattr(model, "ica_W")
    assert model.output_type == "clean"


def test_forward_shape_clean_and_artifact_heads():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal((BATCH, N_CHANNELS, FULL_SAMPLES)).astype(np.float32))

    for output_type in ("clean", "artifact"):
        model = _tiny_model(output_type)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert tuple(out.shape) == (BATCH, N_CHANNELS, EPOCH_SAMPLES), output_type


def test_build_loss_aliases():
    torch = pytest.importorskip("torch")
    assert isinstance(build_loss("mse"), torch.nn.MSELoss)
    assert isinstance(build_loss("l1"), torch.nn.L1Loss)
    assert isinstance(build_loss("smooth_l1"), torch.nn.SmoothL1Loss)
    assert isinstance(build_loss("ensemble"), IcUNetEnsembleLoss)


def test_ensemble_loss_zscored_psd_path_is_finite():
    torch = pytest.importorskip("torch")
    loss = build_loss("ensemble", sfreq=256.0)
    rng = np.random.default_rng(1)
    prediction = torch.from_numpy(rng.standard_normal((BATCH, N_CHANNELS, EPOCH_SAMPLES)).astype(np.float32))
    target = torch.from_numpy(rng.standard_normal((BATCH, N_CHANNELS, EPOCH_SAMPLES)).astype(np.float32))
    value = loss(prediction, target)
    assert torch.isfinite(value)
    assert value.item() > 0.0


def test_ensemble_loss_equal_weight_normalised():
    # All-equal weights => divide by 4; identical preds give zero loss.
    torch = pytest.importorskip("torch")
    loss = IcUNetEnsembleLoss(sfreq=256.0)
    assert loss._weight_sum == pytest.approx(4.0)
    x = torch.zeros(2, N_CHANNELS, EPOCH_SAMPLES)
    assert loss(x, x).item() == pytest.approx(0.0, abs=1e-6)


def test_a_few_optimizer_steps_reduce_loss():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = _tiny_model("clean")
    loss_fn = build_loss("mse")
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)

    rng = np.random.default_rng(0)
    noisy = torch.from_numpy(rng.standard_normal((BATCH, N_CHANNELS, FULL_SAMPLES)).astype(np.float32))
    center = (CONTEXT_EPOCHS // 2) * EPOCH_SAMPLES
    # A learnable clean target: a smoothed, scaled-down version of the center epoch.
    target = noisy[..., center : center + EPOCH_SAMPLES].clone() * 0.1

    model.train()
    losses: list[float] = []
    grad_seen = False
    for _ in range(5):
        optimiser.zero_grad()
        prediction = model(noisy)
        loss = loss_fn(prediction, target)
        loss.backward()
        if not grad_seen:
            grad_seen = any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())
        optimiser.step()
        losses.append(loss.item())

    assert grad_seen, "no parameter received a non-zero gradient"
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def test_build_dataset_from_tiny_npz(tmp_path):
    pytest.importorskip("torch")
    n_examples = 8
    rng = np.random.default_rng(2)
    clean = rng.standard_normal((n_examples, CONTEXT_EPOCHS, N_CHANNELS, EPOCH_SAMPLES)).astype(np.float32)
    artifact = rng.standard_normal((n_examples, CONTEXT_EPOCHS, N_CHANNELS, EPOCH_SAMPLES)).astype(np.float32)
    noisy = clean + artifact
    center = CONTEXT_EPOCHS // 2

    npz_path = tmp_path / "tiny_context.npz"
    np.savez_compressed(
        npz_path,
        noisy_context=noisy,
        clean_context=clean,
        artifact_context=artifact,
        noisy_center=noisy[:, center],
        clean_center=clean[:, center],
        artifact_center=artifact[:, center],
        sfreq=np.asarray([256.0], dtype=np.float64),
    )

    ds = build_dataset(path=str(npz_path), target_type="clean", normalize="zscore")
    assert len(ds) == n_examples
    assert ds.n_channels == N_CHANNELS
    assert ds.chunk_size == EPOCH_SAMPLES
    assert ds.epoch_samples == EPOCH_SAMPLES
    assert ds.input_shape == (N_CHANNELS, FULL_SAMPLES)
    assert ds.target_shape == (N_CHANNELS, EPOCH_SAMPLES)
    assert ds.n_chunks == n_examples
    assert ds.target_type == "clean"
    assert ds.trigger_aligned is True
    assert ds.sfreq == pytest.approx(256.0)

    x, y = ds[0]
    assert x.shape == (N_CHANNELS, FULL_SAMPLES)
    assert y.shape == (N_CHANNELS, EPOCH_SAMPLES)
    assert x.dtype == np.float32 and y.dtype == np.float32
    # z-scored input: per-channel std ~ 1.
    assert np.allclose(x.std(axis=-1), 1.0, atol=1e-2)

    train, val = ds.train_val_split(val_ratio=0.25, seed=0)
    assert len(train) > 0 and len(val) > 0
    assert len(train) + len(val) == n_examples
