"""End-to-end smoke test: dataset → wrapper → optimizer step → trace export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from facet.models.vit_spectrogram.training import (
    ChannelWiseSpectrogramDataset,
    build_dataset,
    build_loss,
    build_model,
)
from facet.training.wrapper import PyTorchModelWrapper


class _StubBase:
    """Replaces NPZContextArtifactDataset with a small in-memory bundle."""

    sfreq = 2048.0

    def __init__(self, n_examples: int = 4, n_channels: int = 2, samples: int = 512):
        rng = np.random.default_rng(0)
        self._noisy = rng.standard_normal((n_examples, 7, n_channels, samples)).astype(np.float32) * 1e-3
        self._target = rng.standard_normal((n_examples, n_channels, samples)).astype(np.float32) * 1e-4

    def __len__(self):
        return self._noisy.shape[0]

    def __getitem__(self, idx):
        return self._noisy[idx].copy(), self._target[idx].copy()


@pytest.mark.unit
def test_dataset_factory_round_trip_via_in_memory_npz(tmp_path: Path):
    rng = np.random.default_rng(0)
    n_examples, n_channels, samples = 4, 2, 512
    bundle = {
        "noisy_context": rng.standard_normal((n_examples, 7, n_channels, samples)).astype(np.float32),
        "clean_center": rng.standard_normal((n_examples, n_channels, samples)).astype(np.float32),
        "sfreq": np.asarray([2048.0]),
    }
    path = tmp_path / "tiny.npz"
    np.savez_compressed(path, **bundle)
    dataset = build_dataset(path=str(path), context_epochs=7, demean_input=False, demean_target=False)
    assert isinstance(dataset, ChannelWiseSpectrogramDataset)
    assert len(dataset) == n_examples * n_channels
    noisy, target = dataset[0]
    assert noisy.shape == (7, 1, samples)
    assert target.shape == (1, samples)
    assert dataset.input_shape == (7, 1, samples)
    assert dataset.target_shape == (1, samples)


@pytest.mark.unit
def test_pytorch_wrapper_runs_train_and_eval_step():
    torch = pytest.importorskip("torch")

    model = build_model(input_shape=(7, 1, 512), depth=2, embed_dim=96, n_heads=4, mlp_ratio=2.0)
    loss_fn = build_loss("mse")
    wrapper = PyTorchModelWrapper(model=model, loss_fn=loss_fn, device="cpu", learning_rate=3e-4)

    rng = np.random.default_rng(0)
    noisy = rng.standard_normal((2, 7, 1, 512)).astype(np.float32) * 1e-3
    target = rng.standard_normal((2, 1, 512)).astype(np.float32) * 1e-4

    eval_metrics = wrapper.eval_step(noisy, target)
    train_metrics = wrapper.train_step(noisy, target)
    assert "loss" in eval_metrics
    assert "loss" in train_metrics
    assert eval_metrics["loss"] >= 0.0
    assert train_metrics["loss"] >= 0.0


@pytest.mark.unit
def test_full_torchscript_export_round_trip(tmp_path: Path):
    torch = pytest.importorskip("torch")

    model = build_model(input_shape=(7, 1, 512), depth=2, embed_dim=96, n_heads=4, mlp_ratio=2.0).eval()
    example = torch.randn(1, 7, 1, 512)
    scripted = torch.jit.trace(model, example)
    artifact_path = tmp_path / "model.ts"
    scripted.save(str(artifact_path))
    assert artifact_path.exists()

    reloaded = torch.jit.load(str(artifact_path))
    reloaded.eval()
    with torch.no_grad():
        a = model(example)
        b = reloaded(example)
    assert tuple(a.shape) == tuple(b.shape) == (1, 1, 512)
    assert torch.allclose(a, b, atol=1e-5)
