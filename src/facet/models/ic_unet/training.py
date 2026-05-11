"""Training factories for the IC-U-Net gradient-artifact model.

This file defines:

- ``IcUnet1D``         — the multichannel 1-D U-Net core, ported from
                         ``roseDwayane/AIEEG`` (Chuang et al. 2022).
- ``IcUnetWithIca``    — wraps ``IcUnet1D`` with frozen ICA and inverse-ICA
                         linear layers plus the FACETpy ``artifact_center``
                         output head.
- ``NiazyContextIcDataset`` — adapts the standard Niazy proof-fit NPZ
                         (``noisy_context`` of shape ``(7, 30, 512)``) into the
                         ``(30, 7*512)`` multichannel time series the model
                         consumes.
- ``build_model``      — facet-train factory returning ``IcUnetWithIca``.
- ``build_loss``       — facet-train factory returning MSE, MAE, or the
                         IC-U-Net ensemble loss.
- ``build_dataset``    — facet-train factory returning ``NiazyContextIcDataset``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from facet.training.dataset import NPZContextArtifactDataset


# ---------------------------------------------------------------------------
# U-Net core (multichannel 1-D)
# ---------------------------------------------------------------------------


class _DoubleConv1d(nn.Module):
    """Two stacked Conv1d-BN-act blocks; the canonical U-Net building block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        negative_slope: float = 0.1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv = _DoubleConv1d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.conv = _DoubleConv1d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            x = nn.functional.pad(x, (0, skip.shape[-1] - x.shape[-1]))
        return self.conv(torch.cat([skip, x], dim=1))


class IcUnet1D(nn.Module):
    """1-D U-Net with the IC-U-Net channel and kernel ladder.

    Channels follow the reference repo ``roseDwayane/AIEEG``:
    ``in_channels -> 64 -> 128 -> 256 -> 512`` and back. Kernel sizes per
    level are ``7, 7, 5, 3`` to match the published configuration.
    """

    def __init__(
        self,
        in_channels: int = 30,
        out_channels: int = 30,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.inc = _DoubleConv1d(in_channels, c1, kernel_size=7)
        self.down1 = _Down(c1, c2, kernel_size=7)
        self.down2 = _Down(c2, c3, kernel_size=5)
        self.down3 = _Down(c3, c4, kernel_size=3)
        self.up1 = _Up(c4 + c3, c3, kernel_size=3)
        self.up2 = _Up(c3 + c2, c2, kernel_size=3)
        self.up3 = _Up(c2 + c1, c1, kernel_size=3)
        self.outc = nn.Conv1d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)


# ---------------------------------------------------------------------------
# IC-U-Net = ICA + U-Net + inverse ICA + center extraction
# ---------------------------------------------------------------------------


class IcUnetWithIca(nn.Module):
    """Full IC-U-Net inference module for the Niazy proof-fit context.

    Pipeline:

    1. Reshape input ``(B, 30, 7*512)`` (already produced by the dataset).
    2. Per-channel demean (optional).
    3. Forward ICA: ``ic = W @ x`` along the channel dim.
    4. ``IcUnet1D`` denoises in IC space.
    5. Inverse ICA: ``clean = W_pinv @ ic_clean``.
    6. Extract center 512 samples (the ``artifact_center`` slot).
    7. Return predicted artifact ``noisy_center - clean_center``.
    """

    def __init__(
        self,
        n_channels: int,
        context_epochs: int,
        epoch_samples: int,
        base_channels: int = 64,
        demean_input: bool = True,
        ica_init: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        self.context_epochs = int(context_epochs)
        self.epoch_samples = int(epoch_samples)
        self.full_samples = self.context_epochs * self.epoch_samples
        self.center_index = self.context_epochs // 2
        self.center_start = self.center_index * self.epoch_samples
        self.center_stop = self.center_start + self.epoch_samples
        self.demean_input = bool(demean_input)

        self.unet = IcUnet1D(
            in_channels=self.n_channels,
            out_channels=self.n_channels,
            base_channels=base_channels,
        )

        if ica_init is None:
            ica_init = np.eye(self.n_channels, dtype=np.float32)
        else:
            ica_init = np.asarray(ica_init, dtype=np.float32)
            if ica_init.shape != (self.n_channels, self.n_channels):
                raise ValueError(
                    f"ica_init must have shape ({self.n_channels}, {self.n_channels}), "
                    f"got {ica_init.shape}"
                )
        ica_inv = np.linalg.pinv(ica_init).astype(np.float32)

        self.register_buffer("ica_W", torch.from_numpy(ica_init))
        self.register_buffer("ica_W_pinv", torch.from_numpy(ica_inv))

    def _channel_demean(self, x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=-1, keepdim=True)

    def _apply_ica(self, x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ij,bjt->bit", matrix, x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.demean_input:
            x = self._channel_demean(x)

        ic = self._apply_ica(x, self.ica_W)
        ic_clean = self.unet(ic)
        clean_full = self._apply_ica(ic_clean, self.ica_W_pinv)

        noisy_center = x[..., self.center_start : self.center_stop]
        clean_center = clean_full[..., self.center_start : self.center_stop]
        return noisy_center - clean_center


# ---------------------------------------------------------------------------
# Loss factories
# ---------------------------------------------------------------------------


class IcUnetEnsembleLoss(nn.Module):
    """Amplitude + velocity + acceleration + frequency MSE ensemble.

    Approximates the four-term ensemble loss from Chuang et al. 2022.
    """

    def __init__(
        self,
        amplitude_weight: float = 1.0,
        velocity_weight: float = 1.0,
        acceleration_weight: float = 1.0,
        frequency_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.amplitude_weight = float(amplitude_weight)
        self.velocity_weight = float(velocity_weight)
        self.acceleration_weight = float(acceleration_weight)
        self.frequency_weight = float(frequency_weight)
        self._mse = nn.MSELoss()

    @staticmethod
    def _diff(x: torch.Tensor) -> torch.Tensor:
        return x[..., 1:] - x[..., :-1]

    @staticmethod
    def _spectrum(x: torch.Tensor) -> torch.Tensor:
        return torch.abs(torch.fft.rfft(x, dim=-1))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.amplitude_weight * self._mse(prediction, target)
        if self.velocity_weight > 0:
            loss = loss + self.velocity_weight * self._mse(
                self._diff(prediction), self._diff(target)
            )
        if self.acceleration_weight > 0:
            loss = loss + self.acceleration_weight * self._mse(
                self._diff(self._diff(prediction)), self._diff(self._diff(target))
            )
        if self.frequency_weight > 0:
            loss = loss + self.frequency_weight * self._mse(
                self._spectrum(prediction), self._spectrum(target)
            )
        return loss


def build_loss(name: str = "mse", **kwargs: Any) -> nn.Module:
    """facet-train loss factory."""
    normalized = name.strip().lower()
    if normalized in {"mse", "amplitude"}:
        return nn.MSELoss()
    if normalized in {"l1", "mae"}:
        return nn.L1Loss()
    if normalized in {"smooth_l1", "huber"}:
        return nn.SmoothL1Loss()
    if normalized in {"ensemble", "ic_unet_ensemble"}:
        return IcUnetEnsembleLoss(
            amplitude_weight=float(kwargs.get("amplitude_weight", 1.0)),
            velocity_weight=float(kwargs.get("velocity_weight", 1.0)),
            acceleration_weight=float(kwargs.get("acceleration_weight", 1.0)),
            frequency_weight=float(kwargs.get("frequency_weight", 0.5)),
        )
    raise ValueError(f"Unknown loss '{name}'")


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------


class NiazyContextIcDataset:
    """Dataset that flattens the 7-epoch context into a single time series.

    Wraps :class:`facet.training.dataset.NPZContextArtifactDataset`. The base
    dataset returns ``(noisy_context, artifact_center)`` with shapes
    ``(7, 30, 512)`` and ``(30, 512)`` respectively. This wrapper reshapes the
    input to ``(30, 7*512=3584)`` so a multichannel 1-D U-Net can consume the
    full context as one long signal.
    """

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
        if first_noisy.ndim != 3:
            raise ValueError(
                "base dataset input must have shape (context_epochs, channels, samples)"
            )
        if first_target.ndim != 2:
            raise ValueError("base dataset target must have shape (channels, samples)")

        self.context_epochs = int(first_noisy.shape[0])
        self.n_channels = int(first_noisy.shape[1])
        self.epoch_samples = int(first_noisy.shape[2])
        self.full_samples = self.context_epochs * self.epoch_samples
        self.chunk_size = self.epoch_samples
        self.target_type = "artifact"
        self.trigger_aligned = True
        self.sfreq = float(getattr(base_dataset, "sfreq", float("nan")))

        self._length = n_base if max_examples is None else max(0, min(int(max_examples), n_base))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        noisy_context, target = self.base_dataset[idx]
        noisy_flat = noisy_context.transpose(1, 0, 2).reshape(self.n_channels, self.full_samples).astype(np.float32, copy=True)
        target_out = target.astype(np.float32, copy=True)
        if self.demean_input:
            noisy_flat -= noisy_flat.mean(axis=-1, keepdims=True)
        if self.demean_target:
            target_out -= target_out.mean(axis=-1, keepdims=True)
        return noisy_flat, target_out

    @property
    def input_shape(self) -> tuple[int, int]:
        return (self.n_channels, self.full_samples)

    @property
    def target_shape(self) -> tuple[int, int]:
        return (self.n_channels, self.epoch_samples)

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
    def __init__(self, parent: NiazyContextIcDataset, indices: list[int]) -> None:
        self._parent = parent
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._parent[self._indices[idx]]


# ---------------------------------------------------------------------------
# ICA fitting helper
# ---------------------------------------------------------------------------


def _fit_ica_matrix(
    dataset_path: str | Path,
    n_channels: int,
    n_samples: int = 100_000,
    random_state: int = 0,
) -> np.ndarray:
    """Fit a frozen ICA unmixing matrix on a flat sample of the dataset.

    Returns
    -------
    np.ndarray
        Shape ``(n_channels, n_channels)``. If FastICA fails to converge, an
        identity matrix is returned so training can still proceed.
    """
    try:
        from sklearn.decomposition import FastICA
    except ImportError:
        return np.eye(n_channels, dtype=np.float32)

    path = Path(dataset_path).expanduser()
    if not path.exists():
        return np.eye(n_channels, dtype=np.float32)

    with np.load(path, allow_pickle=True) as bundle:
        noisy = np.asarray(bundle["noisy_context"], dtype=np.float32)

    if noisy.ndim != 4 or noisy.shape[2] != n_channels:
        return np.eye(n_channels, dtype=np.float32)

    flat = noisy.transpose(0, 2, 1, 3).reshape(-1, n_channels, noisy.shape[1] * noisy.shape[3])
    flat = flat.transpose(0, 2, 1).reshape(-1, n_channels)
    rng = np.random.default_rng(random_state)
    if flat.shape[0] > n_samples:
        idx = rng.choice(flat.shape[0], size=n_samples, replace=False)
        flat = flat[idx]

    ica = FastICA(
        n_components=n_channels,
        whiten="unit-variance",
        max_iter=500,
        tol=1e-4,
        random_state=random_state,
    )
    try:
        ica.fit(flat)
    except Exception:
        return np.eye(n_channels, dtype=np.float32)
    return ica.components_.astype(np.float32)


# ---------------------------------------------------------------------------
# facet-train factories
# ---------------------------------------------------------------------------


def build_model(
    input_shape: tuple[int, int] | tuple[int, int, int] | None = None,
    target_shape: tuple[int, int] | None = None,
    context_epochs: int | None = None,
    epoch_samples: int | None = None,
    n_channels: int | None = None,
    base_channels: int = 64,
    demean_input: bool = True,
    dataset_path: str | None = None,
    fit_ica: bool = True,
    ica_random_state: int = 0,
    **_: object,
) -> IcUnetWithIca:
    """facet-train model factory."""
    if input_shape is None and (n_channels is None or epoch_samples is None or context_epochs is None):
        raise ValueError(
            "build_model requires input_shape or (n_channels, context_epochs, epoch_samples)"
        )

    if input_shape is not None and len(input_shape) == 2:
        resolved_n_channels = int(input_shape[0])
        full_samples = int(input_shape[1])
        if context_epochs and epoch_samples:
            if context_epochs * epoch_samples != full_samples:
                raise ValueError(
                    f"context_epochs * epoch_samples ({context_epochs}*{epoch_samples}) does not "
                    f"match input_shape[1]={full_samples}"
                )
            resolved_context_epochs = int(context_epochs)
            resolved_epoch_samples = int(epoch_samples)
        else:
            resolved_context_epochs = int(context_epochs) if context_epochs else 7
            if full_samples % resolved_context_epochs != 0:
                raise ValueError(
                    f"Cannot split full_samples={full_samples} into {resolved_context_epochs} epochs"
                )
            resolved_epoch_samples = full_samples // resolved_context_epochs
    else:
        resolved_n_channels = int(n_channels)
        resolved_context_epochs = int(context_epochs)
        resolved_epoch_samples = int(epoch_samples)

    ica_init = None
    if fit_ica and dataset_path is not None:
        ica_init = _fit_ica_matrix(
            dataset_path=dataset_path,
            n_channels=resolved_n_channels,
            random_state=ica_random_state,
        )

    return IcUnetWithIca(
        n_channels=resolved_n_channels,
        context_epochs=resolved_context_epochs,
        epoch_samples=resolved_epoch_samples,
        base_channels=int(base_channels),
        demean_input=bool(demean_input),
        ica_init=ica_init,
    )


def build_dataset(
    path: str | None = None,
    context_path: str | None = None,
    max_examples: int | None = None,
    demean_input: bool = True,
    demean_target: bool = True,
    **_: object,
) -> NiazyContextIcDataset:
    """facet-train dataset factory."""
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
    return NiazyContextIcDataset(
        base,
        demean_input=demean_input,
        demean_target=demean_target,
        max_examples=max_examples,
    )
