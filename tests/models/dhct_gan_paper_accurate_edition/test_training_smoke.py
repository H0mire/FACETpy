"""Cheap CPU smoke test for the paper-accurate DHCT-GAN edition.

Builds tiny tensors in-memory, runs a few optimizer steps over the generator
(LSGAN discriminator steps fire internally when grad is enabled), and asserts the
final loss is below the initial loss. Also checks the forward shape, the eval
no_grad path, and that the generator exports via ``torch.jit.trace``. Target ~3 s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from facet.models.dhct_gan_paper_accurate_edition.training import (
    DHCTGanGeneratorPA,
    build_dataset,
    build_loss,
    build_model,
)

# Tiny dims so a forward+backward on CPU completes in a few seconds.
_SAMPLES = 64
_BATCH = 4
_GEN_KWARGS = dict(base_channels=4, depth=2, num_heads=2, n_local_blocks=4, n_lgtb=1)
_LOSS_KWARGS = dict(lambda_feat=1.0, lambda_adv=0.1, disc_channels=4, disc_depth=2, disc_lr=1e-3)


def _make_fixture_npz(path: Path) -> None:
    rng = np.random.default_rng(7)
    n_examples, n_channels, samples = 6, 3, _SAMPLES
    artifact = rng.standard_normal((n_examples, n_channels, samples)).astype(np.float32)
    clean = rng.standard_normal((n_examples, n_channels, samples)).astype(np.float32) * 0.05
    noisy = clean + artifact
    np.savez(
        path,
        noisy_center=noisy,
        clean_center=clean,
        artifact_center=artifact,
        sfreq=np.array([512.0], dtype=np.float32),
    )


@pytest.mark.unit
def test_forward_shape() -> None:
    model = build_model(epoch_samples=_SAMPLES, **_GEN_KWARGS)
    x = torch.randn(_BATCH, 1, _SAMPLES)
    out = model(x)
    assert out.shape == (_BATCH, 1, _SAMPLES)
    assert torch.isfinite(out).all()


@pytest.mark.unit
def test_generator_exposes_all_branches() -> None:
    model: DHCTGanGeneratorPA = build_model(epoch_samples=_SAMPLES, **_GEN_KWARGS)
    x = torch.randn(_BATCH, 1, _SAMPLES)
    outputs = model._compute_outputs(x)
    for key in ("artifact", "clean", "fused_clean", "mask1", "mask2"):
        assert outputs[key].shape == (_BATCH, 1, _SAMPLES), key


@pytest.mark.unit
def test_dataset_contract(tmp_path: Path) -> None:
    npz = tmp_path / "fixture.npz"
    _make_fixture_npz(npz)
    dataset = build_dataset(path=str(npz), demean=True)
    assert dataset.n_channels == 3
    assert dataset.chunk_size == _SAMPLES
    assert dataset.input_shape == (1, _SAMPLES)
    assert dataset.target_shape == (3, _SAMPLES)
    assert dataset.target_type == "artifact"
    assert dataset.trigger_aligned is True
    x, y = dataset[0]
    assert x.shape == (1, _SAMPLES)
    assert y.shape == (3, _SAMPLES)
    train_ds, val_ds = dataset.train_val_split(val_ratio=0.25, seed=0)
    assert len(train_ds) + len(val_ds) == len(dataset)
    assert train_ds.n_channels == 3


@pytest.mark.unit
def test_training_decreases_loss(tmp_path: Path) -> None:
    npz = tmp_path / "fixture.npz"
    _make_fixture_npz(npz)
    dataset = build_dataset(path=str(npz), demean=True)

    model = build_model(epoch_samples=_SAMPLES, **_GEN_KWARGS)
    loss_fn = build_loss(**_LOSS_KWARGS)

    # Single CLI-style optimizer over generator params only (mirrors the wrapper).
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # One fixed batch so the loss trend is clean.
    xs, ys = [], []
    for i in range(_BATCH):
        x, y = dataset[i]
        xs.append(x)
        ys.append(y)
    x = torch.as_tensor(np.stack(xs), dtype=torch.float32)
    target = torch.as_tensor(np.stack(ys), dtype=torch.float32)

    model.train()
    losses: list[float] = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)  # artifact head
        loss = loss_fn(pred, target)  # internal LSGAN disc steps run here
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


@pytest.mark.unit
def test_eval_no_grad_path(tmp_path: Path) -> None:
    npz = tmp_path / "fixture.npz"
    _make_fixture_npz(npz)
    dataset = build_dataset(path=str(npz), demean=True)
    model = build_model(epoch_samples=_SAMPLES, **_GEN_KWARGS)
    loss_fn = build_loss(**_LOSS_KWARGS)

    x, y = dataset[0]
    x = torch.as_tensor(x[None], dtype=torch.float32)
    target = torch.as_tensor(y[None], dtype=torch.float32)

    # Warm up the loss device/optimizer bootstrap under grad first.
    model.train()
    _ = loss_fn(model(x), target)

    model.eval()
    with torch.no_grad():
        value = loss_fn(model(x), target)
    assert np.isfinite(float(value))
    assert value.ndim == 0


@pytest.mark.unit
def test_generator_traceable() -> None:
    model = build_model(epoch_samples=_SAMPLES, **_GEN_KWARGS)
    model.eval()
    example = torch.randn(1, 1, _SAMPLES)
    traced = torch.jit.trace(model, example)
    out = traced(torch.randn(2, 1, _SAMPLES))
    assert out.shape == (2, 1, _SAMPLES)
