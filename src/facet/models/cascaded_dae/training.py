"""Training factories for Cascaded DAE: channel-wise cascaded denoising autoencoder.

The implementation is based on the older ``feature/deeplearning`` PyTorch
prototype, but adapted to the current FACETpy training CLI. Each training item
is a single-channel window with shape ``(1, samples)``. This keeps the exported
model independent of the number of channels in a later dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from facet.training.dataset import EEGArtifactDataset, NPZContextArtifactDataset


class DenoisingAutoencoder(torch.nn.Module):
    """Fully connected denoising autoencoder used as one cascade stage."""

    def __init__(
        self,
        input_size: int,
        hidden_units: tuple[int, int, int] = (128, 32, 128),
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_units = tuple(int(v) for v in hidden_units)
        self.dropout_rate = float(dropout_rate)

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_units[0]),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(self.dropout_rate),
            torch.nn.Linear(self.hidden_units[0], self.hidden_units[1]),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(self.dropout_rate),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_units[1], self.hidden_units[2]),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(self.dropout_rate),
            torch.nn.Linear(self.hidden_units[2], self.input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        flat = x.reshape(batch_size, -1)
        decoded = self.decoder(self.encoder(flat))
        return decoded.reshape_as(x)


class CascadedDenoisingAutoencoder(torch.nn.Module):
    """Two-stage residual artifact predictor inspired by the legacy DAE prototype.

    The model predicts the artifact signal, not the clean EEG. FACETpy subtracts
    the predicted artifact via ``DeepLearningCorrection`` during inference.
    Stage 1 estimates an initial artifact. Stage 2 sees the signal after that
    estimate has been subtracted and predicts the remaining residual artifact.
    """

    def __init__(
        self,
        input_size: int,
        hidden_units: tuple[int, int, int] = (128, 32, 128),
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.stage1 = DenoisingAutoencoder(
            input_size=self.input_size,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
        )
        self.stage2 = DenoisingAutoencoder(
            input_size=self.input_size,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stage1_artifact = self.stage1(x)
        residual_signal = x - stage1_artifact
        residual_artifact = self.stage2(residual_signal)
        return stage1_artifact + residual_artifact


class ChannelWiseArtifactDataset:
    """View that turns chunk datasets into per-channel artifact examples."""

    def __init__(
        self,
        base_dataset: Any,
        *,
        demean_input: bool = True,
        demean_target: bool = True,
        max_examples: int | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.demean_input = bool(demean_input)
        self.demean_target = bool(demean_target)

        n_base = len(base_dataset)
        if n_base == 0:
            raise ValueError("base dataset must contain at least one example")
        first_noisy, first_target = base_dataset[0]
        if first_noisy.ndim == 3:
            self._context_dataset = True
            self.n_channels = int(first_noisy.shape[1])
            self.chunk_size = int(first_noisy.shape[2])
        elif first_noisy.ndim == 2:
            self._context_dataset = False
            self.n_channels = int(first_noisy.shape[0])
            self.chunk_size = int(first_noisy.shape[1])
        else:
            raise ValueError("base dataset input must have shape (channels, samples) or (context, channels, samples)")
        if first_target.ndim != 2:
            raise ValueError("base dataset target must have shape (channels, samples)")

        total = n_base * self.n_channels
        self._length = total if max_examples is None else max(0, min(int(max_examples), total))
        self.target_type = "artifact"
        self.trigger_aligned = bool(getattr(base_dataset, "trigger_aligned", False))
        self.sfreq = float(getattr(base_dataset, "sfreq", float("nan")))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        base_idx = int(idx) // self.n_channels
        channel_idx = int(idx) % self.n_channels
        noisy, target = self.base_dataset[base_idx]
        if self._context_dataset:
            center = noisy.shape[0] // 2
            noisy_channel = noisy[center, channel_idx]
        else:
            noisy_channel = noisy[channel_idx]
        target_channel = target[channel_idx]

        noisy_out = noisy_channel[np.newaxis, :].astype(np.float32, copy=True)
        target_out = target_channel[np.newaxis, :].astype(np.float32, copy=True)
        if self.demean_input:
            noisy_out -= noisy_out.mean(axis=-1, keepdims=True)
        if self.demean_target:
            target_out -= target_out.mean(axis=-1, keepdims=True)
        return noisy_out, target_out

    @property
    def input_shape(self) -> tuple[int, int]:
        return (1, self.chunk_size)

    @property
    def target_shape(self) -> tuple[int, int]:
        return (1, self.chunk_size)

    @property
    def n_chunks(self) -> int:
        return len(self)

    def train_val_split(self, val_ratio: float = 0.2, seed: int = 42):
        n = len(self)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n).tolist()
        n_val = max(1, int(n * val_ratio))
        val_idx = set(indices[:n_val])
        train_idx = [i for i in range(n) if i not in val_idx]
        val_idx_list = [i for i in range(n) if i in val_idx]
        return _SubsetDataset(self, train_idx), _SubsetDataset(self, val_idx_list)


class _SubsetDataset:
    def __init__(self, parent: ChannelWiseArtifactDataset, indices: list[int]) -> None:
        self._parent = parent
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._parent[self._indices[idx]]


def _parse_hidden_units(hidden_units: list[int] | tuple[int, int, int] | None) -> tuple[int, int, int]:
    values = tuple(hidden_units or (128, 32, 128))
    if len(values) != 3:
        raise ValueError("hidden_units must contain exactly three values")
    return (int(values[0]), int(values[1]), int(values[2]))


def build_model(
    input_shape: tuple[int, int] | tuple[int, int, int] | None = None,
    chunk_size: int | None = None,
    hidden_units: list[int] | tuple[int, int, int] | None = None,
    dropout_rate: float = 0.2,
    **_: object,
) -> CascadedDenoisingAutoencoder:
    """Build the channel-wise cascaded DAE for ``facet-train``."""
    if input_shape is not None:
        input_size = int(np.prod(input_shape))
    elif chunk_size is not None:
        input_size = int(chunk_size)
    else:
        raise ValueError("build_model requires input_shape or chunk_size")
    return CascadedDenoisingAutoencoder(
        input_size=input_size,
        hidden_units=_parse_hidden_units(hidden_units),
        dropout_rate=dropout_rate,
    )


def build_loss(name: str = "mse"):
    normalized = name.strip().lower()
    if normalized == "l1":
        return torch.nn.L1Loss()
    if normalized in {"smooth_l1", "huber"}:
        return torch.nn.SmoothL1Loss()
    return torch.nn.MSELoss()


def build_dataset(
    path: str | None = None,
    context_path: str | None = None,
    max_examples: int | None = None,
    demean_input: bool = True,
    demean_target: bool = True,
    **_: object,
) -> ChannelWiseArtifactDataset:
    """Load a prepared artifact bundle and expose per-channel examples.

    ``path`` may point to the context dataset generated for Demo01. The center
    noisy epoch is paired with the center artifact target.
    """
    dataset_path = Path(path or context_path or "").expanduser()
    if not str(dataset_path):
        raise ValueError("build_dataset requires path or context_path")
    base = NPZContextArtifactDataset(
        path=dataset_path,
        input_key="noisy_context",
        target_key="artifact_center",
        demean_input=False,
        demean_target=False,
    )
    return ChannelWiseArtifactDataset(
        base,
        demean_input=demean_input,
        demean_target=demean_target,
        max_examples=max_examples,
    )


def build_dataset_from_contexts(
    contexts,
    chunk_size: int = 292,
    target_type: str = "artifact",
    trigger_aligned: bool = True,
    overlap: float = 0.0,
    eeg_only: bool = True,
    max_examples: int | None = None,
    demean_input: bool = True,
    demean_target: bool = True,
    **_: object,
) -> ChannelWiseArtifactDataset:
    """Build per-channel examples from FACETpy ``ProcessingContext`` objects."""
    base = EEGArtifactDataset(
        contexts=contexts,
        chunk_size=chunk_size,
        target_type=target_type,
        trigger_aligned=trigger_aligned,
        overlap=overlap,
        eeg_only=eeg_only,
    )
    return ChannelWiseArtifactDataset(
        base,
        demean_input=demean_input,
        demean_target=demean_target,
        max_examples=max_examples,
    )
