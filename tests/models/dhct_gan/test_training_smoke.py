"""Smoke training test for DHCT-GAN — runs a tiny in-memory training loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from facet.models.dhct_gan.training import build_dataset, build_loss, build_model
from facet.training.trainer import Trainer
from facet.training.config import TrainingConfig
from facet.training.wrapper import PyTorchModelWrapper


def _make_fixture_npz(path: Path) -> None:
    rng = np.random.default_rng(7)
    n_examples, n_channels, samples = 6, 4, 128
    artifact = rng.standard_normal((n_examples, n_channels, samples)).astype(np.float32)
    clean = rng.standard_normal((n_examples, n_channels, samples)).astype(np.float32) * 0.05
    noisy = clean + artifact
    np.savez(
        path,
        noisy_center=noisy,
        clean_center=clean,
        artifact_center=artifact,
        sfreq=np.array([512.0]),
    )


@pytest.mark.unit
def test_two_epoch_smoke_loop(tmp_path: Path) -> None:
    npz = tmp_path / "fixture.npz"
    _make_fixture_npz(npz)

    dataset = build_dataset(path=str(npz), demean=True)
    train_ds, val_ds = dataset.train_val_split(val_ratio=0.25, seed=0)

    model = build_model(epoch_samples=128, base_channels=8, depth=3)
    loss_fn = build_loss(alpha_consistency=0.5, beta_adv=0.05)

    wrapper = PyTorchModelWrapper(
        model=model,
        loss_fn=loss_fn,
        device="cpu",
        learning_rate=1e-3,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
    )

    config = TrainingConfig(
        model_name="DHCTGanSmoke",
        chunk_size=128,
        target_type="artifact",
        trigger_aligned=True,
        val_ratio=0.25,
        max_epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
        seed=0,
        output_dir=str(tmp_path / "training_output"),
    )
    config.logging.rich_live = False
    config.logging.progress_bar = False
    config.logging.log_file = None
    config.logging.loss_plot_file = None
    config.early_stopping = None

    trainer = Trainer(
        wrapper=wrapper,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
    )
    result = trainer.fit()
    assert result.success
    assert result.total_epochs == 2
    assert np.isfinite(result.best_metric)
