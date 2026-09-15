"""
PyTorch training example for the framework-agnostic FACETpy training module.

This example:

1. creates a small synthetic EEG artifact dataset,
2. wraps a simple 1D convolutional model in ``PyTorchModelWrapper``,
3. trains it with the Rich-enabled ``Trainer``,
4. prints the resulting run directory and best validation loss.

Requires the optional PyTorch extra:

    pip install facetpy[pytorch]

or with Poetry:

    poetry install -E pytorch
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mne
import numpy as np
import torch

from facet.core import ProcessingContext, ProcessingMetadata
from facet.training import (
    EarlyStoppingConfig,
    EEGArtifactDataset,
    LoggingConfig,
    PyTorchModelWrapper,
    Trainer,
    TrainingConfig,
)


class ArtifactNet(torch.nn.Module):
    """Small channel-wise artifact predictor for demonstration purposes."""

    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(n_channels, 16, kernel_size=9, padding=4),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 16, kernel_size=9, padding=4),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, n_channels, kernel_size=9, padding=4),
        )

    def forward(self, x):
        return self.net(x)


def build_context() -> ProcessingContext:
    """Create a synthetic context with clean and artifact-contaminated EEG."""
    sfreq = 250.0
    n_samples = 2500
    rng = np.random.default_rng(42)
    times = np.arange(n_samples) / sfreq

    clean = np.vstack(
        [
            np.sin(2 * np.pi * 8.0 * times),
            np.sin(2 * np.pi * 10.0 * times + 0.3),
            np.sin(2 * np.pi * 12.5 * times + 0.7),
            np.sin(2 * np.pi * 18.0 * times + 1.1),
        ]
    ).astype(np.float32)
    clean += 0.02 * rng.standard_normal(clean.shape).astype(np.float32)

    artifact = np.zeros_like(clean)
    trigger_starts = np.arange(0, n_samples, 250, dtype=int)[:10]
    for start in trigger_starts:
        artifact[:, start : start + 50] += 0.35

    noisy = clean + artifact
    info = mne.create_info(
        ch_names=["EEG001", "EEG002", "EEG003", "EEG004"],
        sfreq=sfreq,
        ch_types=["eeg"] * 4,
    )
    metadata = ProcessingMetadata()
    metadata.triggers = trigger_starts
    metadata.artifact_length = 50
    return ProcessingContext(
        raw=mne.io.RawArray(noisy, info, verbose=False),
        raw_original=mne.io.RawArray(clean, info, verbose=False),
        metadata=metadata,
    )


def main() -> None:
    context = build_context()
    dataset = EEGArtifactDataset(
        context,
        chunk_size=250,
        target_type="artifact",
        trigger_aligned=True,
    )
    train_ds, val_ds = dataset.train_val_split(val_ratio=0.2, seed=7)

    model = ArtifactNet(n_channels=dataset.n_channels)
    wrapper = PyTorchModelWrapper(
        model=model,
        loss_fn=torch.nn.L1Loss(),
        device="cpu",
        learning_rate=1e-3,
        weight_decay=1e-4,
    )

    with tempfile.TemporaryDirectory(prefix="facetpy-train-example-") as tmp_dir:
        config = TrainingConfig(
            model_name="ArtifactNetDemo",
            max_epochs=5,
            batch_size=4,
            output_dir=tmp_dir,
            target_type="artifact",
            logging=LoggingConfig(rich_live=False, log_file="metrics.jsonl"),
            early_stopping=EarlyStoppingConfig(
                monitor="loss",
                patience=2,
                min_delta=1e-6,
            ),
        )
        result = Trainer(wrapper, train_ds, val_ds, config).fit()

        print(f"Run directory: {Path(result.run_dir).resolve()}")
        print(f"Epochs completed: {result.total_epochs}")
        print(f"Best epoch: {result.best_epoch}")
        print(f"Best metric: {result.best_metric:.6f}")


if __name__ == "__main__":
    main()
