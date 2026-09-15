"""Smoke test for the Demucs ``facet-train`` factory chain.

Builds a tiny synthetic context NPZ, calls ``build_model`` / ``build_dataset``
through their public signatures, runs one forward-backward pass on CPU, and
exports the traced model to TorchScript. Mirrors what ``facet-train fit`` does
end-to-end without spinning up the full training loop.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest


def _write_synthetic_context_npz(path: Path, n_examples=4, context_epochs=7, n_channels=2, samples=32):
    rng = np.random.default_rng(0)
    noisy = rng.standard_normal((n_examples, context_epochs, n_channels, samples)).astype(np.float32)
    artifact = noisy * 0.3
    np.savez(
        path,
        noisy_context=noisy,
        artifact_context=artifact,
        sfreq=np.array([2048.0]),
    )


def test_demucs_factories_one_step_and_export(tmp_path):
    torch = pytest.importorskip("torch")
    from facet.models.demucs.training import build_dataset, build_loss, build_model

    bundle = tmp_path / "bundle.npz"
    _write_synthetic_context_npz(bundle, n_examples=4, context_epochs=7, n_channels=2, samples=32)

    dataset = build_dataset(path=str(bundle), context_epochs=7)
    assert dataset.input_shape == (1, 7 * 32)

    model = build_model(input_shape=dataset.input_shape, depth=2, initial_channels=8, lstm_layers=1)
    loss_fn = build_loss("l1")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    noisy, target = dataset[0]
    x = torch.as_tensor(noisy, dtype=torch.float32).unsqueeze(0)
    y_target = torch.as_tensor(target, dtype=torch.float32).unsqueeze(0)

    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y_target)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)

    export_path = tmp_path / "demucs.ts"
    model.eval()
    example = x.detach().clone().requires_grad_(False)
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        traced = torch.jit.trace(model, example)
    traced.save(str(export_path))
    assert export_path.exists() and export_path.stat().st_size > 0

    reloaded = torch.jit.load(str(export_path))
    with torch.no_grad():
        reloaded_y = reloaded(example)
    assert tuple(reloaded_y.shape) == tuple(y_pred.shape)
