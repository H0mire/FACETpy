"""Smoke test for the paper-accurate Demucs ``facet-train`` factory chain.

Builds a tiny synthetic context NPZ, exercises ``build_dataset`` /
``build_model`` / ``build_loss`` through their public signatures, runs a few
Adam steps asserting the loss decreases, and verifies the length-agnostic
forward (the regression that fixes the original crash) plus the 2x resampling
trick. Every dimension is tiny so the whole file runs in a couple of seconds on
CPU.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest


def _write_synthetic_context_npz(path: Path, n_examples=4, context_epochs=7, n_channels=2, samples=32):
    rng = np.random.default_rng(0)
    noisy = rng.standard_normal((n_examples, context_epochs, n_channels, samples)).astype(np.float32)
    # Make the target a learnable function of the input so the loss can drop.
    artifact = noisy * 0.3 + 0.01 * rng.standard_normal(noisy.shape).astype(np.float32)
    np.savez(
        path,
        noisy_context=noisy,
        artifact_context=artifact,
        sfreq=np.array([2048.0]),
    )


def test_demucs_paper_accurate_factory_chain_and_training(tmp_path):
    torch = pytest.importorskip("torch")
    from facet.models.demucs_paper_accurate_edition.training import (
        build_dataset,
        build_loss,
        build_model,
    )

    bundle = tmp_path / "bundle.npz"
    _write_synthetic_context_npz(bundle, n_examples=4, context_epochs=7, n_channels=2, samples=32)

    # (1) Factory chain.
    dataset = build_dataset(path=str(bundle), context_epochs=7, max_examples=8)
    assert dataset.input_shape == (1, 7 * 32)
    assert dataset.target_shape == (1, 7 * 32)
    assert dataset.target_type == "artifact"
    assert dataset.trigger_aligned is True
    assert dataset.n_channels == 2
    assert dataset.epoch_samples == 32
    assert dataset.chunk_size == 7 * 32
    assert dataset.n_chunks == len(dataset)
    assert dataset.sfreq == pytest.approx(2048.0)
    train, val = dataset.train_val_split(val_ratio=0.25, seed=1)
    assert len(train) > 0 and len(val) > 0

    model = build_model(
        input_shape=dataset.input_shape,
        depth=2,
        initial_channels=8,
        lstm_layers=1,
        resample=1,
    )
    loss_fn = build_loss("l1")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Build a small batch.
    xs, ys = [], []
    for i in range(min(4, len(dataset))):
        noisy, target = dataset[i]
        xs.append(noisy)
        ys.append(target)
    x = torch.as_tensor(np.stack(xs), dtype=torch.float32)
    y_target = torch.as_tensor(np.stack(ys), dtype=torch.float32)

    # (2) Train assertion: loss decreases.
    model.train()
    with torch.no_grad():
        initial_loss = loss_fn(model(x), y_target).item()
    for _ in range(5):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y_target)
        loss.backward()
        optimizer.step()
    final_loss = loss.item()
    assert torch.isfinite(loss)
    assert final_loss < initial_loss

    # (3) Forward-shape, stride-divisible length (224 / 4^2 = 14, integer).
    model.eval()
    with torch.no_grad():
        out_div = model(x)
    assert tuple(out_div.shape) == tuple(x.shape)


def test_demucs_paper_accurate_length_agnostic_forward():
    """Regression: non-divisible length must NOT crash and must preserve length.

    The original edition raises a skip-size RuntimeError here; the valid_length
    padding + center-trim fix makes the output shape equal the input shape.
    """
    torch = pytest.importorskip("torch")
    from facet.models.demucs_paper_accurate_edition.training import build_model

    model = build_model(input_shape=(1, 7 * 30), depth=2, initial_channels=8, lstm_layers=1, resample=1)
    model.eval()

    # 210 = 7 * 30 is NOT a multiple of stride^depth (4^2 = 16).
    assert (7 * 30) % (4**2) != 0
    x = torch.randn(2, 1, 7 * 30)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == tuple(x.shape)


def test_demucs_paper_accurate_resampling_trick_shape():
    """The 2x resampling trick must restore the original waveform length."""
    torch = pytest.importorskip("torch")
    from facet.models.demucs_paper_accurate_edition.training import build_model

    model = build_model(input_shape=(1, 7 * 32), depth=2, initial_channels=8, lstm_layers=1, resample=2)
    assert model.resampler is not None
    model.eval()
    x = torch.randn(2, 1, 7 * 32)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == tuple(x.shape)


def test_demucs_paper_accurate_torchscript_trace_roundtrip(tmp_path):
    """Trace + reload + forward-shape check (mirrors what facet-train export does)."""
    torch = pytest.importorskip("torch")
    from facet.models.demucs_paper_accurate_edition.training import build_model

    model = build_model(input_shape=(1, 7 * 32), depth=2, initial_channels=8, lstm_layers=1, resample=1)
    model.eval()
    example = torch.randn(1, 1, 7 * 32)
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        traced = torch.jit.trace(model, example)
        expected = model(example)
    export_path = tmp_path / "demucs_paper_accurate.ts"
    traced.save(str(export_path))
    assert export_path.exists() and export_path.stat().st_size > 0

    reloaded = torch.jit.load(str(export_path))
    with torch.no_grad():
        reloaded_y = reloaded(example)
    assert tuple(reloaded_y.shape) == tuple(expected.shape)


def test_demucs_paper_accurate_processor_registered():
    """The correction must register under the unique paper-accurate name."""
    from facet.core import get_processor
    import facet.models.demucs_paper_accurate_edition.processor  # noqa: F401

    cls = get_processor("demucs_paper_accurate_correction")
    assert cls is not None
    assert cls.name == "demucs_paper_accurate_correction"
