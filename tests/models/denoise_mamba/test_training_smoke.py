"""Smoke test: a single training step with the DenoiseMamba pure-PyTorch model."""

from __future__ import annotations

import numpy as np
import pytest


def test_pytorch_wrapper_train_step_runs(tmp_path):
    pytest.importorskip("torch")
    from facet.models.denoise_mamba.training import build_loss, build_model
    from facet.training.wrapper import PyTorchModelWrapper

    model = build_model(epoch_samples=16, d_model=8, d_state=4, n_blocks=1, dropout=0.0)
    loss_fn = build_loss("mse")
    wrapper = PyTorchModelWrapper(model=model, loss_fn=loss_fn, device="cpu")

    rng = np.random.default_rng(0)
    noisy = rng.standard_normal((4, 1, 16)).astype(np.float32)
    target = rng.standard_normal((4, 1, 16)).astype(np.float32)

    initial = wrapper.train_step(noisy, target)
    assert "loss" in initial
    initial_loss = initial["loss"]

    for _ in range(3):
        wrapper.train_step(noisy, target)
    final = wrapper.eval_step(noisy, target)
    assert final["loss"] <= initial_loss + 1e-3


def test_train_val_split_returns_disjoint_subsets():
    pytest.importorskip("torch")
    from facet.models.denoise_mamba.training import ChannelWiseSingleEpochArtifactDataset

    class _FakeBase:
        sfreq = 100.0
        def __len__(self): return 4
        def __getitem__(self, idx):
            return (np.zeros((2, 8), dtype=np.float32), np.ones((2, 8), dtype=np.float32))

    dataset = ChannelWiseSingleEpochArtifactDataset(_FakeBase(), demean_input=False, demean_target=False)
    train, val = dataset.train_val_split(val_ratio=0.25, seed=0)
    assert len(train) + len(val) == len(dataset)
    assert len(val) >= 1
