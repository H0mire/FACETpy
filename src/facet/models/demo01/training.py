"""Factories for training a 7-epoch context artifact model with ``facet-train``.

The dataset is expected to come from:
``examples/build_synthetic_spike_artifact_context_dataset.py``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from facet.training import NPZContextArtifactDataset


class SevenEpochContextArtifactNet(torch.nn.Module):
    """Small fully convolutional model for center-epoch artifact prediction."""

    def __init__(
        self,
        context_epochs: int,
        n_channels: int,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        in_channels = context_epochs * n_channels
        self.context_epochs = context_epochs
        self.n_channels = n_channels
        self.net = torch.nn.Sequential(
            torch.nn.ReflectionPad1d(4),
            torch.nn.Conv1d(in_channels, hidden_channels, kernel_size=9),
            torch.nn.GELU(),
            torch.nn.ReflectionPad1d(4),
            torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size=9),
            torch.nn.GELU(),
            torch.nn.ReflectionPad1d(4),
            torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size=9),
            torch.nn.GELU(),
            torch.nn.ReflectionPad1d(4),
            torch.nn.Conv1d(hidden_channels, n_channels, kernel_size=9, bias=False),
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected input shape (batch, context_epochs, channels, samples)")
        batch, context_epochs, n_channels, samples = x.shape
        return self.net(x.reshape(batch, context_epochs * n_channels, samples))


def build_dataset(
    path: str = "./output/synthetic_spike_artifact_context_from_generator/synthetic_spike_artifact_context_dataset.npz",
    max_examples: int | None = None,
    demean_input: bool = True,
    demean_target: bool = True,
    **_: object,
) -> NPZContextArtifactDataset:
    """Load the synthetic 7-epoch context bundle."""
    return NPZContextArtifactDataset(
        path=Path(path),
        input_key="noisy_context",
        target_key="artifact_center",
        max_examples=max_examples,
        demean_input=demean_input,
        demean_target=demean_target,
    )


def build_model(
    input_shape: tuple[int, int, int],
    hidden_channels: int = 32,
    **_: object,
) -> SevenEpochContextArtifactNet:
    """Instantiate the context model from the dataset input shape."""
    context_epochs, n_channels, _epoch_samples = input_shape
    return SevenEpochContextArtifactNet(
        context_epochs=context_epochs,
        n_channels=n_channels,
        hidden_channels=hidden_channels,
    )


def build_loss(name: str = "l1"):
    """Return a PyTorch loss for artifact-center prediction."""
    normalized = name.strip().lower()
    if normalized == "mse":
        return torch.nn.MSELoss()
    return torch.nn.L1Loss()
