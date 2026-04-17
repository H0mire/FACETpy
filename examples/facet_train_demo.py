"""
Demo factories for ``facet-train fit --config examples/facet_train_demo.yaml``.

This module provides the minimum pieces required by the training CLI:

- ``build_contexts``: returns one or more supervised ``ProcessingContext`` objects
- ``build_model``: returns a trainable PyTorch model
- ``build_loss``: returns a PyTorch loss callable

Run from the repository root:

    poetry run facet-train fit --config examples/facet_train_demo.yaml

Requires the PyTorch extra:

    poetry install -E pytorch
"""

from __future__ import annotations

import mne
import numpy as np
import torch

from facet.core import ProcessingContext, ProcessingMetadata


class ArtifactNet(torch.nn.Module):
    """Small 1D CNN artifact predictor used by the CLI demo config."""

    def __init__(self, n_channels: int, hidden_channels: int = 16) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(n_channels, hidden_channels, kernel_size=9, padding=4),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size=9, padding=4),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_channels, n_channels, kernel_size=9, padding=4),
        )

    def forward(self, x):
        return self.net(x)


def build_contexts(
    n_recordings: int = 3,
    n_channels: int = 4,
    n_samples: int = 2500,
    sfreq: float = 250.0,
) -> list[ProcessingContext]:
    """Create small supervised synthetic recordings for the training CLI demo."""
    contexts: list[ProcessingContext] = []
    base_times = np.arange(n_samples) / sfreq

    for recording_idx in range(n_recordings):
        rng = np.random.default_rng(42 + recording_idx)

        clean = np.vstack(
            [
                np.sin(2 * np.pi * 8.0 * base_times + 0.2 * ch_idx)
                + 0.3 * np.sin(2 * np.pi * (12.0 + ch_idx) * base_times)
                for ch_idx in range(n_channels)
            ]
        ).astype(np.float32)
        clean += 0.02 * rng.standard_normal(clean.shape).astype(np.float32)

        artifact = np.zeros_like(clean)
        trigger_starts = np.arange(0, n_samples, 250, dtype=int)[:10]
        for start in trigger_starts:
            stop = min(start + 50, n_samples)
            artifact[:, start:stop] += 0.35 + 0.05 * recording_idx

        noisy = clean + artifact

        info = mne.create_info(
            ch_names=[f"EEG{idx + 1:03d}" for idx in range(n_channels)],
            sfreq=sfreq,
            ch_types=["eeg"] * n_channels,
        )
        metadata = ProcessingMetadata()
        metadata.triggers = trigger_starts
        metadata.artifact_length = 50

        contexts.append(
            ProcessingContext(
                raw=mne.io.RawArray(noisy, info, verbose=False),
                raw_original=mne.io.RawArray(clean, info, verbose=False),
                metadata=metadata,
            )
        )

    return contexts


def build_model(
    n_channels: int,
    chunk_size: int,
    hidden_channels: int = 16,
    **_: object,
) -> ArtifactNet:
    """Factory used by the CLI to instantiate the demonstration model."""
    del chunk_size  # Included for CLI compatibility; not needed by this simple CNN.
    return ArtifactNet(n_channels=n_channels, hidden_channels=hidden_channels)


def build_loss(name: str = "l1"):
    """Return a simple PyTorch loss module for the CLI demo."""
    normalized = name.strip().lower()
    if normalized == "mse":
        return torch.nn.MSELoss()
    return torch.nn.L1Loss()
