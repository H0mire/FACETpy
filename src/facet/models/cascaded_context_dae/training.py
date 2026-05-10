"""Training factories for the 7-epoch cascaded context DAE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from facet.training.dataset import NPZContextArtifactDataset


class ContextDenoisingAutoencoder(torch.nn.Module):
    """Fully connected DAE stage mapping context epochs to the center artifact."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_units: tuple[int, int, int] = (512, 128, 512),
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
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
            torch.nn.Linear(self.hidden_units[2], self.output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        flat = x.reshape(batch_size, -1)
        return self.decoder(self.encoder(flat)).reshape(batch_size, 1, self.output_size)


class CascadedContextDenoisingAutoencoder(torch.nn.Module):
    """Two-stage residual context DAE that predicts the center-epoch artifact."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_units: tuple[int, int, int] = (512, 128, 512),
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.stage1 = ContextDenoisingAutoencoder(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
        )
        self.stage2 = ContextDenoisingAutoencoder(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stage1_artifact = self.stage1(x)
        residual_context = x.clone()
        center_idx = residual_context.shape[1] // 2
        residual_context[:, center_idx : center_idx + 1, :, :] = (
            residual_context[:, center_idx : center_idx + 1, :, :] - stage1_artifact[:, None, :, :]
        )
        residual_artifact = self.stage2(residual_context)
        return stage1_artifact + residual_artifact


class ChannelWiseContextArtifactDataset:
    """Expose `(context_epochs, 1, samples)` -> `(1, samples)` examples."""

    def __init__(
        self,
        base_dataset: Any,
        *,
        context_epochs: int = 7,
        demean_input: bool = True,
        demean_target: bool = True,
        max_examples: int | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.context_epochs = int(context_epochs)
        self.demean_input = bool(demean_input)
        self.demean_target = bool(demean_target)
        if self.context_epochs < 1 or self.context_epochs % 2 == 0:
            raise ValueError("context_epochs must be a positive odd integer")

        n_base = len(base_dataset)
        if n_base == 0:
            raise ValueError("base dataset must contain at least one example")
        first_noisy, first_target = base_dataset[0]
        if first_noisy.ndim != 3:
            raise ValueError("base dataset input must have shape (context_epochs, channels, samples)")
        if first_target.ndim != 2:
            raise ValueError("base dataset target must have shape (channels, samples)")
        if first_noisy.shape[0] != self.context_epochs:
            raise ValueError(f"expected {self.context_epochs} context epochs, got {first_noisy.shape[0]}")
        self.n_channels = int(first_noisy.shape[1])
        self.epoch_samples = int(first_noisy.shape[2])
        self.chunk_size = self.epoch_samples
        self.target_type = "artifact"
        self.trigger_aligned = True
        self.sfreq = float(getattr(base_dataset, "sfreq", float("nan")))

        total = n_base * self.n_channels
        self._length = total if max_examples is None else max(0, min(int(max_examples), total))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        base_idx = int(idx) // self.n_channels
        channel_idx = int(idx) % self.n_channels
        noisy_context, target = self.base_dataset[base_idx]
        noisy_out = noisy_context[:, channel_idx : channel_idx + 1, :].astype(np.float32, copy=True)
        target_out = target[channel_idx : channel_idx + 1, :].astype(np.float32, copy=True)
        if self.demean_input:
            noisy_out -= noisy_out.mean(axis=-1, keepdims=True)
        if self.demean_target:
            target_out -= target_out.mean(axis=-1, keepdims=True)
        return noisy_out, target_out

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.context_epochs, 1, self.epoch_samples)

    @property
    def target_shape(self) -> tuple[int, int]:
        return (1, self.epoch_samples)

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
    def __init__(self, parent: ChannelWiseContextArtifactDataset, indices: list[int]) -> None:
        self._parent = parent
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._parent[self._indices[idx]]


def _parse_hidden_units(hidden_units: list[int] | tuple[int, int, int] | None) -> tuple[int, int, int]:
    values = tuple(hidden_units or (512, 128, 512))
    if len(values) != 3:
        raise ValueError("hidden_units must contain exactly three values")
    return (int(values[0]), int(values[1]), int(values[2]))


def build_model(
    input_shape: tuple[int, int, int] | None = None,
    target_shape: tuple[int, int] | None = None,
    epoch_samples: int | None = None,
    hidden_units: list[int] | tuple[int, int, int] | None = None,
    dropout_rate: float = 0.2,
    **_: object,
) -> CascadedContextDenoisingAutoencoder:
    if input_shape is None:
        if epoch_samples is None:
            raise ValueError("build_model requires input_shape or epoch_samples")
        input_shape = (7, 1, int(epoch_samples))
    input_size = int(np.prod(input_shape))
    output_size = int(np.prod(target_shape)) if target_shape is not None else int(input_shape[-1])
    return CascadedContextDenoisingAutoencoder(
        input_size=input_size,
        output_size=output_size,
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
    context_epochs: int = 7,
    max_examples: int | None = None,
    demean_input: bool = True,
    demean_target: bool = True,
    **_: object,
) -> ChannelWiseContextArtifactDataset:
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
    return ChannelWiseContextArtifactDataset(
        base,
        context_epochs=context_epochs,
        demean_input=demean_input,
        demean_target=demean_target,
        max_examples=max_examples,
    )
