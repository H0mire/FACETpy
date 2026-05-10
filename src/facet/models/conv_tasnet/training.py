"""Training factories for the Conv-TasNet gradient-artifact source separator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


class _GlobalLayerNorm(torch.nn.Module):
    """Global layer norm over channel and time dimensions.

    Equivalent to gLN in the Conv-TasNet paper. Normalises features by the
    mean and variance taken jointly over the channel axis and the time
    axis, and applies a learned scale and bias per channel.
    """

    def __init__(self, n_features: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(1, n_features, 1))
        self.beta = torch.nn.Parameter(torch.zeros(1, n_features, 1))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class _TemporalBlock(torch.nn.Module):
    """One dilated TCN block (1×1 expand → DConv → 1×1 contract)."""

    def __init__(
        self,
        bottleneck_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.expand = torch.nn.Conv1d(bottleneck_channels, hidden_channels, kernel_size=1)
        self.expand_act = torch.nn.PReLU()
        self.expand_norm = _GlobalLayerNorm(hidden_channels)
        self.dconv = torch.nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            groups=hidden_channels,
        )
        self.dconv_act = torch.nn.PReLU()
        self.dconv_norm = _GlobalLayerNorm(hidden_channels)
        self.residual = torch.nn.Conv1d(hidden_channels, bottleneck_channels, kernel_size=1)
        self.skip = torch.nn.Conv1d(hidden_channels, bottleneck_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.expand_norm(self.expand_act(self.expand(x)))
        h = self.dconv_norm(self.dconv_act(self.dconv(h)))
        return x + self.residual(h), self.skip(h)


class ConvTasNetSeparator(torch.nn.Module):
    """1D Conv-TasNet adapted to single-channel artifact source separation.

    Input shape: ``(batch, 1, samples)``. Output shape:
    ``(batch, n_sources, samples)``. With ``n_sources=2`` the first source
    is the clean EEG estimate and the second is the gradient artifact
    estimate.
    """

    def __init__(
        self,
        *,
        n_sources: int = 2,
        encoder_filters: int = 256,
        encoder_kernel: int = 16,
        bottleneck_channels: int = 128,
        hidden_channels: int = 256,
        block_kernel: int = 3,
        n_blocks: int = 8,
        n_repeats: int = 2,
        mask_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        if encoder_kernel < 2 or encoder_kernel % 2 != 0:
            raise ValueError("encoder_kernel must be an even integer >= 2")
        self.n_sources = int(n_sources)
        self.encoder_filters = int(encoder_filters)
        self.encoder_kernel = int(encoder_kernel)
        self.bottleneck_channels = int(bottleneck_channels)
        self.hidden_channels = int(hidden_channels)
        self.block_kernel = int(block_kernel)
        self.n_blocks = int(n_blocks)
        self.n_repeats = int(n_repeats)
        self.mask_activation = mask_activation

        self.encoder_stride = self.encoder_kernel // 2
        self.encoder = torch.nn.Conv1d(
            1,
            self.encoder_filters,
            kernel_size=self.encoder_kernel,
            stride=self.encoder_stride,
            bias=False,
        )
        self.encoder_act = torch.nn.ReLU()
        self.pre_norm = _GlobalLayerNorm(self.encoder_filters)
        self.bottleneck = torch.nn.Conv1d(self.encoder_filters, self.bottleneck_channels, kernel_size=1)
        self.tcn_blocks = torch.nn.ModuleList(
            [
                _TemporalBlock(
                    bottleneck_channels=self.bottleneck_channels,
                    hidden_channels=self.hidden_channels,
                    kernel_size=self.block_kernel,
                    dilation=2 ** block_idx,
                )
                for _ in range(self.n_repeats)
                for block_idx in range(self.n_blocks)
            ]
        )
        self.mask_act = torch.nn.PReLU()
        self.mask_conv = torch.nn.Conv1d(
            self.bottleneck_channels,
            self.n_sources * self.encoder_filters,
            kernel_size=1,
        )
        self.decoder = torch.nn.ConvTranspose1d(
            self.encoder_filters,
            1,
            kernel_size=self.encoder_kernel,
            stride=self.encoder_stride,
            bias=False,
        )

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        if mixture.dim() != 3 or mixture.shape[1] != 1:
            raise ValueError(
                f"ConvTasNetSeparator expects shape (batch, 1, samples), got {tuple(mixture.shape)}"
            )
        n_samples = mixture.shape[-1]
        latent = self.encoder_act(self.encoder(mixture))
        bottleneck = self.bottleneck(self.pre_norm(latent))

        skip_sum = torch.zeros_like(bottleneck)
        h = bottleneck
        for block in self.tcn_blocks:
            h, skip = block(h)
            skip_sum = skip_sum + skip

        mask_logits = self.mask_conv(self.mask_act(skip_sum))
        masks = self._apply_mask_activation(mask_logits)
        masks = masks.view(
            mixture.shape[0], self.n_sources, self.encoder_filters, latent.shape[-1]
        )

        sources: list[torch.Tensor] = []
        for src_idx in range(self.n_sources):
            masked = latent * masks[:, src_idx]
            decoded = self.decoder(masked)
            sources.append(decoded[..., :n_samples])
        return torch.cat(sources, dim=1)

    def _apply_mask_activation(self, mask_logits: torch.Tensor) -> torch.Tensor:
        normalized = self.mask_activation.strip().lower()
        if normalized == "sigmoid":
            return torch.sigmoid(mask_logits)
        if normalized == "relu":
            return torch.relu(mask_logits)
        if normalized == "softmax":
            batch, _, frames = mask_logits.shape
            reshaped = mask_logits.view(batch, self.n_sources, self.encoder_filters, frames)
            return torch.softmax(reshaped, dim=1).view(batch, -1, frames)
        raise ValueError(
            f"Unsupported mask_activation '{self.mask_activation}'. "
            "Expected one of: sigmoid, relu, softmax."
        )


# ---------------------------------------------------------------------------
# Channel-wise dataset wrapper
# ---------------------------------------------------------------------------


class ChannelWiseSourceSeparationDataset:
    """Yield ``(noisy, sources)`` per channel from an NPZ context bundle.

    Each item returns ``(mixture, sources)`` where ``mixture`` has shape
    ``(1, samples)`` and ``sources`` has shape ``(2, samples)`` with index
    ``0`` = clean EEG and index ``1`` = gradient artifact.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_examples: int | None = None,
        demean_input: bool = True,
        demean_target: bool = True,
    ) -> None:
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        with np.load(self.path, allow_pickle=True) as bundle:
            noisy_center = bundle["noisy_center"].astype(np.float32, copy=False)
            clean_center = bundle["clean_center"].astype(np.float32, copy=False)
            artifact_center = bundle["artifact_center"].astype(np.float32, copy=False)
            self.sfreq = float(bundle["sfreq"][0]) if "sfreq" in bundle else float("nan")

        if noisy_center.shape != clean_center.shape or noisy_center.shape != artifact_center.shape:
            raise ValueError("noisy_center, clean_center, and artifact_center must have identical shapes")
        if noisy_center.ndim != 3:
            raise ValueError("noisy_center must have shape (examples, channels, samples)")

        self.noisy_center = noisy_center
        self.clean_center = clean_center
        self.artifact_center = artifact_center
        self.n_examples = int(noisy_center.shape[0])
        self.n_channels = int(noisy_center.shape[1])
        self.epoch_samples = int(noisy_center.shape[2])
        self.chunk_size = self.epoch_samples
        self.target_type = "artifact"
        self.trigger_aligned = True
        self.demean_input = bool(demean_input)
        self.demean_target = bool(demean_target)

        total = self.n_examples * self.n_channels
        self._length = total if max_examples is None else max(0, min(int(max_examples), total))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        example_idx = int(idx) // self.n_channels
        channel_idx = int(idx) % self.n_channels

        mixture = self.noisy_center[example_idx, channel_idx : channel_idx + 1, :].astype(np.float32, copy=True)
        clean = self.clean_center[example_idx, channel_idx, :].astype(np.float32, copy=True)
        artifact = self.artifact_center[example_idx, channel_idx, :].astype(np.float32, copy=True)

        if self.demean_input:
            mixture -= mixture.mean(axis=-1, keepdims=True)
        if self.demean_target:
            clean -= clean.mean()
            artifact -= artifact.mean()

        sources = np.stack([clean, artifact], axis=0).astype(np.float32, copy=False)
        return mixture, sources

    @property
    def input_shape(self) -> tuple[int, int]:
        return (1, self.epoch_samples)

    @property
    def target_shape(self) -> tuple[int, int]:
        return (2, self.epoch_samples)

    @property
    def n_chunks(self) -> int:
        return len(self)

    def train_val_split(self, val_ratio: float = 0.2, seed: int = 42):
        n = len(self)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n).tolist()
        n_val = max(1, int(n * val_ratio))
        val_set = set(indices[:n_val])
        train_idx = [i for i in range(n) if i not in val_set]
        val_idx = [i for i in range(n) if i in val_set]
        return _SubsetDataset(self, train_idx), _SubsetDataset(self, val_idx)


class _SubsetDataset:
    def __init__(self, parent: ChannelWiseSourceSeparationDataset, indices: list[int]) -> None:
        self._parent = parent
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._parent[self._indices[idx]]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class _NegSISDR(torch.nn.Module):
    """Negative SI-SDR averaged over the source axis (no permutation)."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Negative SI-SDR requires matching shapes, got {tuple(prediction.shape)} vs {tuple(target.shape)}"
            )
        prediction = prediction - prediction.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        scale = (prediction * target).sum(dim=-1, keepdim=True) / (
            target.pow(2).sum(dim=-1, keepdim=True) + self.eps
        )
        projection = scale * target
        noise = prediction - projection
        sdr = 10.0 * torch.log10(
            (projection.pow(2).sum(dim=-1) + self.eps)
            / (noise.pow(2).sum(dim=-1) + self.eps)
        )
        return -sdr.mean()


class _WeightedSourceMSE(torch.nn.Module):
    """MSE per source with separate weights for clean and artifact."""

    def __init__(self, clean_weight: float = 1.0, artifact_weight: float = 1.0) -> None:
        super().__init__()
        self.clean_weight = float(clean_weight)
        self.artifact_weight = float(artifact_weight)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        clean_mse = torch.nn.functional.mse_loss(prediction[:, 0], target[:, 0])
        artifact_mse = torch.nn.functional.mse_loss(prediction[:, 1], target[:, 1])
        return self.clean_weight * clean_mse + self.artifact_weight * artifact_mse


# ---------------------------------------------------------------------------
# Factories used by facet-train
# ---------------------------------------------------------------------------


def build_model(
    *,
    n_sources: int = 2,
    encoder_filters: int = 256,
    encoder_kernel: int = 16,
    bottleneck_channels: int = 128,
    hidden_channels: int = 256,
    block_kernel: int = 3,
    n_blocks: int = 8,
    n_repeats: int = 2,
    mask_activation: str = "sigmoid",
    **_: object,
) -> ConvTasNetSeparator:
    return ConvTasNetSeparator(
        n_sources=n_sources,
        encoder_filters=encoder_filters,
        encoder_kernel=encoder_kernel,
        bottleneck_channels=bottleneck_channels,
        hidden_channels=hidden_channels,
        block_kernel=block_kernel,
        n_blocks=n_blocks,
        n_repeats=n_repeats,
        mask_activation=mask_activation,
    )


def build_loss(
    name: str = "mse",
    *,
    clean_weight: float = 1.0,
    artifact_weight: float = 1.0,
) -> torch.nn.Module:
    normalized = name.strip().lower()
    if normalized == "mse":
        return torch.nn.MSELoss()
    if normalized in {"l1", "mae"}:
        return torch.nn.L1Loss()
    if normalized in {"weighted_mse", "source_mse"}:
        return _WeightedSourceMSE(clean_weight=clean_weight, artifact_weight=artifact_weight)
    if normalized in {"si_sdr_neg", "neg_si_sdr", "si_sdr"}:
        return _NegSISDR()
    raise ValueError(
        f"Unsupported loss name '{name}'. Use one of: mse, l1, weighted_mse, si_sdr_neg."
    )


def build_dataset(
    path: str | None = None,
    context_path: str | None = None,
    max_examples: int | None = None,
    demean_input: bool = True,
    demean_target: bool = True,
    **_: object,
) -> ChannelWiseSourceSeparationDataset:
    dataset_path = path or context_path
    if not dataset_path:
        raise ValueError("build_dataset requires path or context_path")
    return ChannelWiseSourceSeparationDataset(
        path=dataset_path,
        max_examples=max_examples,
        demean_input=demean_input,
        demean_target=demean_target,
    )


__all__ = [
    "ConvTasNetSeparator",
    "ChannelWiseSourceSeparationDataset",
    "build_model",
    "build_loss",
    "build_dataset",
]
