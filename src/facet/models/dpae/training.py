"""Training factories for the Dual-Pathway Autoencoder (DPAE).

The architecture follows Xiong et al. 2023 (Frontiers in Neuroscience,
"A general dual-pathway network for EEG denoising"). We use the 1D-CNN
variant. Each training item is a single-channel ``(1, samples)`` window so the
exported checkpoint stays independent of the EEG channel count of the target
dataset, mirroring the ``cascaded_dae`` convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from facet.training.dataset import EEGArtifactDataset, NPZContextArtifactDataset


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


class _LocalPathway(nn.Module):
    """Small-kernel dilated 1D conv stack for fine temporal detail."""

    def __init__(self, base_filters: int, latent_filters: int) -> None:
        super().__init__()
        f = base_filters
        self.layers = nn.Sequential(
            nn.Conv1d(1, f, kernel_size=3, padding=1, dilation=1),
            nn.SELU(inplace=True),
            nn.Conv1d(f, f, kernel_size=3, padding=2, dilation=2),
            nn.SELU(inplace=True),
            nn.Conv1d(f, f * 2, kernel_size=3, padding=4, dilation=4),
            nn.SELU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(f * 2, f * 2, kernel_size=3, padding=8, dilation=8),
            nn.SELU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(f * 2, latent_filters, kernel_size=3, padding=1),
            nn.SELU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _GlobalPathway(nn.Module):
    """Large-kernel pooled 1D conv stack for slow trends and gradient envelope."""

    def __init__(self, base_filters: int, latent_filters: int) -> None:
        super().__init__()
        f = base_filters
        self.layers = nn.Sequential(
            nn.Conv1d(1, f, kernel_size=15, padding=7),
            nn.SELU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(f, f * 2, kernel_size=11, padding=5),
            nn.SELU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(f * 2, latent_filters, kernel_size=7, padding=3),
            nn.SELU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _Decoder(nn.Module):
    """Mirror decoder upsampling the fused latent back to the input length."""

    def __init__(self, fused_filters: int, base_filters: int) -> None:
        super().__init__()
        f = base_filters
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(fused_filters, f * 2, kernel_size=4, stride=2, padding=1),
            nn.SELU(inplace=True),
            nn.ConvTranspose1d(f * 2, f, kernel_size=4, stride=2, padding=1),
            nn.SELU(inplace=True),
            nn.Conv1d(f, f, kernel_size=3, padding=1),
            nn.SELU(inplace=True),
            nn.Conv1d(f, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DualPathwayAutoencoder(nn.Module):
    """1D-CNN dual-pathway autoencoder predicting the gradient artifact waveform.

    Forward contract:
        input  : (batch, 1, samples)  noisy single-channel epoch
        output : (batch, 1, samples)  predicted artifact (subtracted by FACETpy)

    Both pathways consume the input in parallel; their bottleneck features are
    concatenated along the channel axis, fused by a 1x1 conv with batch
    normalisation, and decoded back to the input length. A learned per-channel
    residual scalar adds a fraction of the input back into the prediction so the
    network can begin training from a near-identity starting point.
    """

    def __init__(
        self,
        input_size: int,
        base_filters: int = 32,
        latent_filters: int = 128,
        residual_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.base_filters = int(base_filters)
        self.latent_filters = int(latent_filters)

        self.local = _LocalPathway(self.base_filters, self.latent_filters)
        self.global_ = _GlobalPathway(self.base_filters, self.latent_filters)

        fused = self.latent_filters * 2
        self.fusion = nn.Sequential(
            nn.BatchNorm1d(fused),
            nn.Conv1d(fused, fused, kernel_size=1),
            nn.SELU(inplace=True),
        )
        self.decoder = _Decoder(fused, self.base_filters)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[1] == 1 and x.shape[2] == 1:
            # Accept (batch, 1, 1, samples) tensors from upstream for robustness.
            x = x.squeeze(1)
        local_latent = self.local(x)
        global_latent = self.global_(x)
        fused = torch.cat([local_latent, global_latent], dim=1)
        fused = self.fusion(fused)
        decoded = self.decoder(fused)
        return decoded + self.residual_scale * x


# ---------------------------------------------------------------------------
# Channel-wise dataset wrapper (mirrors cascaded_dae)
# ---------------------------------------------------------------------------


class ChannelWiseArtifactDataset:
    """Expose ``(channels, samples)`` or context bundles as ``(1, samples)`` items."""

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


# ---------------------------------------------------------------------------
# facet-train factories
# ---------------------------------------------------------------------------


def build_model(
    input_shape: tuple[int, int] | tuple[int, int, int] | None = None,
    chunk_size: int | None = None,
    base_filters: int = 32,
    latent_filters: int = 128,
    residual_init: float = 0.0,
    **_: object,
) -> DualPathwayAutoencoder:
    """Build the DPAE 1D-CNN model for facet-train."""
    if input_shape is not None:
        input_size = int(input_shape[-1])
    elif chunk_size is not None:
        input_size = int(chunk_size)
    else:
        raise ValueError("build_model requires input_shape or chunk_size")
    if input_size % 4 != 0:
        raise ValueError(f"input_size must be divisible by 4 for the dual-pathway encoder, got {input_size}")
    return DualPathwayAutoencoder(
        input_size=input_size,
        base_filters=int(base_filters),
        latent_filters=int(latent_filters),
        residual_init=float(residual_init),
    )


def build_loss(name: str = "mse"):
    normalized = name.strip().lower()
    if normalized == "l1":
        return nn.L1Loss()
    if normalized in {"smooth_l1", "huber"}:
        return nn.SmoothL1Loss()
    return nn.MSELoss()


def build_dataset(
    path: str | None = None,
    context_path: str | None = None,
    max_examples: int | None = None,
    demean_input: bool = True,
    demean_target: bool = True,
    **_: object,
) -> ChannelWiseArtifactDataset:
    """Load a Niazy proof-fit context bundle and expose per-channel center epochs."""
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
    chunk_size: int = 512,
    target_type: str = "artifact",
    trigger_aligned: bool = True,
    overlap: float = 0.0,
    eeg_only: bool = True,
    max_examples: int | None = None,
    demean_input: bool = True,
    demean_target: bool = True,
    **_: object,
) -> ChannelWiseArtifactDataset:
    """Build per-channel examples from FACETpy ProcessingContext objects."""
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
