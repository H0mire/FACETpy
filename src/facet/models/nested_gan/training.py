"""Training factories for the Nested-GAN model on the Niazy proof-fit dataset.

The exported model is the generator only. Training uses a generator-only
recipe with a multi-resolution STFT loss that captures the same spectral
fidelity the published Nested-GAN's multi-resolution discriminators are
trained to enforce. See documentation/research_notes.md for the deliberate
scope reduction relative to the published paper.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from facet.training.dataset import NPZContextArtifactDataset


CENTER_INDEX = 3  # 7-epoch context, zero-based index of the center epoch


# ---------------------------------------------------------------------------
# Inner branch: complex STFT Restormer block
# ---------------------------------------------------------------------------


class _LayerNorm2d(torch.nn.Module):
    """Channel-wise layer norm for ``(B, C, H, W)`` tensors."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(channels))
        self.bias = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + 1e-6)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class _MDTA(torch.nn.Module):
    """Multi-DConv head transposed attention from Restormer (Zamir 2022).

    Attention is computed across the channel dimension instead of spatial
    positions, which keeps complexity linear in the spatial size.
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.temperature = torch.nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = torch.nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = torch.nn.Conv2d(
            channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False
        )
        self.project_out = torch.nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.reshape(b, c, h, w)
        return self.project_out(out)


class _GDFN(torch.nn.Module):
    """Gated DConv feed-forward network from Restormer (Zamir 2022)."""

    def __init__(self, channels: int, expansion: float = 2.0) -> None:
        super().__init__()
        hidden = max(1, int(channels * expansion))
        self.project_in = torch.nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=False)
        self.dwconv = torch.nn.Conv2d(
            hidden * 2, hidden * 2, kernel_size=3, padding=1, groups=hidden * 2, bias=False
        )
        self.project_out = torch.nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class _RestormerBlock(torch.nn.Module):
    """One Restormer block: MDTA → GDFN with layer-norm residuals."""

    def __init__(self, channels: int, num_heads: int = 4, expansion: float = 2.0) -> None:
        super().__init__()
        self.norm1 = _LayerNorm2d(channels)
        self.attn = _MDTA(channels, num_heads=num_heads)
        self.norm2 = _LayerNorm2d(channels)
        self.ffn = _GDFN(channels, expansion=expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class InnerSpectrogramRestormer(torch.nn.Module):
    """Inner branch: light-weighted complex-valued Restormer over the center-epoch STFT.

    Operates on the complex STFT of the center epoch. Real/imaginary parts
    are treated as two input channels. The output is the predicted complex
    spectrogram of the artifact, inverse-STFT'd to time domain.
    """

    def __init__(
        self,
        *,
        n_fft: int = 64,
        hop_length: int = 16,
        win_length: int | None = None,
        channels: int = 48,
        num_blocks: int = 4,
        num_heads: int = 4,
        expansion: float = 2.0,
        target_samples: int = 512,
    ) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length or n_fft)
        self.target_samples = int(target_samples)
        self.register_buffer(
            "stft_window",
            torch.hann_window(self.win_length, periodic=True),
            persistent=False,
        )

        self.input_proj = torch.nn.Conv2d(2, channels, kernel_size=3, padding=1, bias=False)
        self.blocks = torch.nn.Sequential(
            *[
                _RestormerBlock(channels=channels, num_heads=num_heads, expansion=expansion)
                for _ in range(int(num_blocks))
            ]
        )
        self.output_proj = torch.nn.Conv2d(channels, 2, kernel_size=3, padding=1, bias=False)

    def _stft(self, signal: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.stft_window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )
        real = spec.real.unsqueeze(1)
        imag = spec.imag.unsqueeze(1)
        return torch.cat([real, imag], dim=1)

    def _istft(self, spec: torch.Tensor) -> torch.Tensor:
        complex_spec = torch.complex(spec[:, 0], spec[:, 1])
        return torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.stft_window,
            center=True,
            normalized=False,
            length=self.target_samples,
            return_complex=False,
        )

    def forward(self, center_epoch: torch.Tensor) -> torch.Tensor:
        spec = self._stft(center_epoch)
        z = self.input_proj(spec)
        z = self.blocks(z)
        artifact_spec = self.output_proj(z)
        return self._istft(artifact_spec)


# ---------------------------------------------------------------------------
# Outer branch: 1D U-Net refiner over 7-epoch context
# ---------------------------------------------------------------------------


def _conv_block(in_channels: int, out_channels: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
        torch.nn.GELU(),
        torch.nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
        torch.nn.GELU(),
    )


class OuterTimeRefiner(torch.nn.Module):
    """Outer branch: 1D residual U-Net over the 7-epoch context.

    Input is the context stack with the center epoch replaced by the inner
    branch's time-domain output. Output is a refined center-epoch artifact.
    """

    def __init__(
        self,
        *,
        context_epochs: int = 7,
        base_channels: int = 32,
        target_samples: int = 512,
    ) -> None:
        super().__init__()
        if context_epochs < 1:
            raise ValueError("context_epochs must be >= 1")
        self.context_epochs = int(context_epochs)
        self.target_samples = int(target_samples)

        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        self.enc1 = _conv_block(self.context_epochs, c1)
        self.enc2 = _conv_block(c1, c2)
        self.enc3 = _conv_block(c2, c3)
        self.bottleneck = _conv_block(c3, c4)

        self.up3 = torch.nn.ConvTranspose1d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = _conv_block(c4, c3)
        self.up2 = torch.nn.ConvTranspose1d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = _conv_block(c3, c2)
        self.up1 = torch.nn.ConvTranspose1d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = _conv_block(c2, c1)

        self.head = torch.nn.Conv1d(c1, 1, kernel_size=1)

    def forward(self, context_stack: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(context_stack)
        e2 = self.enc2(F.avg_pool1d(e1, kernel_size=2))
        e3 = self.enc3(F.avg_pool1d(e2, kernel_size=2))
        b = self.bottleneck(F.avg_pool1d(e3, kernel_size=2))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ---------------------------------------------------------------------------
# Nested-GAN generator (the only network we train and export)
# ---------------------------------------------------------------------------


class NestedGANGenerator(torch.nn.Module):
    """Inner spectrogram Restormer cascaded into an outer time-domain refiner.

    Input shape:  ``(batch, context_epochs, 1, target_samples)`` (channel-wise).
    Output shape: ``(batch, 1, target_samples)`` predicted center-epoch artifact.
    """

    def __init__(
        self,
        *,
        context_epochs: int = 7,
        target_samples: int = 512,
        inner_channels: int = 48,
        inner_blocks: int = 4,
        inner_heads: int = 4,
        inner_expansion: float = 2.0,
        outer_base_channels: int = 32,
        n_fft: int = 64,
        hop_length: int = 16,
        win_length: int | None = None,
    ) -> None:
        super().__init__()
        if context_epochs < 1 or context_epochs % 2 == 0:
            raise ValueError("context_epochs must be a positive odd integer")
        self.context_epochs = int(context_epochs)
        self.center_index = self.context_epochs // 2
        self.target_samples = int(target_samples)

        self.inner = InnerSpectrogramRestormer(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            channels=inner_channels,
            num_blocks=inner_blocks,
            num_heads=inner_heads,
            expansion=inner_expansion,
            target_samples=self.target_samples,
        )
        self.outer = OuterTimeRefiner(
            context_epochs=self.context_epochs,
            base_channels=outer_base_channels,
            target_samples=self.target_samples,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected (B, C_epochs, 1, T) input, got shape {tuple(x.shape)}")
        batch, n_epochs, n_in_channels, n_samples = x.shape
        if n_epochs != self.context_epochs:
            raise ValueError(f"expected {self.context_epochs} context epochs, got {n_epochs}")
        if n_in_channels != 1:
            raise ValueError(f"expected per-channel input with 1 channel, got {n_in_channels}")
        if n_samples != self.target_samples:
            raise ValueError(f"expected {self.target_samples} samples per epoch, got {n_samples}")

        context_2d = x.squeeze(2)
        center_epoch = context_2d[:, self.center_index, :]

        inner_artifact = self.inner(center_epoch)

        refined_context = context_2d.clone()
        refined_context[:, self.center_index, :] = (
            context_2d[:, self.center_index, :] - inner_artifact
        )

        residual = self.outer(refined_context)
        return inner_artifact.unsqueeze(1) + residual


# ---------------------------------------------------------------------------
# Loss: L1 in time plus multi-resolution log-magnitude STFT loss
# ---------------------------------------------------------------------------


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Sum of L1 errors on log-magnitude STFT at several window sizes."""

    def __init__(
        self,
        *,
        fft_sizes: tuple[int, ...] = (32, 64, 128, 256),
        hop_fraction: float = 0.25,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not fft_sizes:
            raise ValueError("fft_sizes must contain at least one window size")
        self.fft_sizes = tuple(int(n) for n in fft_sizes)
        self.hop_fraction = float(hop_fraction)
        self.eps = float(eps)
        for n_fft in self.fft_sizes:
            self.register_buffer(
                f"window_{n_fft}",
                torch.hann_window(n_fft, periodic=True),
                persistent=False,
            )

    def _log_mag(self, signal: torch.Tensor, n_fft: int) -> torch.Tensor:
        hop = max(1, int(n_fft * self.hop_fraction))
        window = getattr(self, f"window_{n_fft}")
        spec = torch.stft(
            signal,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )
        return torch.log(spec.abs() + self.eps)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = prediction.reshape(-1, prediction.shape[-1])
        targ = target.reshape(-1, target.shape[-1])
        total = pred.new_zeros(())
        for n_fft in self.fft_sizes:
            total = total + F.l1_loss(self._log_mag(pred, n_fft), self._log_mag(targ, n_fft))
        return total / float(len(self.fft_sizes))


class NestedGANLoss(torch.nn.Module):
    """L1 time-domain loss plus multi-resolution STFT magnitude loss."""

    def __init__(
        self,
        *,
        lambda_time: float = 1.0,
        lambda_mrstft: float = 0.5,
        fft_sizes: tuple[int, ...] = (32, 64, 128, 256),
        hop_fraction: float = 0.25,
    ) -> None:
        super().__init__()
        if lambda_time < 0 or lambda_mrstft < 0:
            raise ValueError("loss weights must be non-negative")
        self.lambda_time = float(lambda_time)
        self.lambda_mrstft = float(lambda_mrstft)
        self.mrstft = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_fraction=hop_fraction,
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        time_loss = F.l1_loss(prediction, target)
        spec_loss = self.mrstft(prediction, target)
        return self.lambda_time * time_loss + self.lambda_mrstft * spec_loss


# ---------------------------------------------------------------------------
# Per-channel context dataset (mirrors cascaded_context_dae)
# ---------------------------------------------------------------------------


class ChannelWiseContextArtifactDataset:
    """Expose ``(context_epochs, 1, samples)`` -> ``(1, samples)`` examples."""

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


# ---------------------------------------------------------------------------
# Factories consumed by facet-train via the YAML config
# ---------------------------------------------------------------------------


def build_model(
    input_shape: tuple[int, int, int] | None = None,
    epoch_samples: int | None = None,
    context_epochs: int | None = None,
    inner_channels: int = 48,
    inner_blocks: int = 4,
    inner_heads: int = 4,
    inner_expansion: float = 2.0,
    outer_base_channels: int = 32,
    n_fft: int = 64,
    hop_length: int = 16,
    win_length: int | None = None,
    **_: object,
) -> NestedGANGenerator:
    if input_shape is not None:
        ctx, _ch, samp = input_shape
    else:
        if context_epochs is None or epoch_samples is None:
            raise ValueError("build_model requires input_shape or context_epochs + epoch_samples")
        ctx, samp = int(context_epochs), int(epoch_samples)

    if win_length is None:
        win_length = n_fft

    target_samples = int(samp)
    if math.gcd(target_samples, hop_length) != hop_length:
        raise ValueError(
            f"target_samples={target_samples} must be a multiple of hop_length={hop_length} for clean iSTFT"
        )

    return NestedGANGenerator(
        context_epochs=int(ctx),
        target_samples=target_samples,
        inner_channels=int(inner_channels),
        inner_blocks=int(inner_blocks),
        inner_heads=int(inner_heads),
        inner_expansion=float(inner_expansion),
        outer_base_channels=int(outer_base_channels),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=int(win_length),
    )


def build_loss(
    lambda_time: float = 1.0,
    lambda_mrstft: float = 0.5,
    fft_sizes: list[int] | tuple[int, ...] | None = None,
    hop_fraction: float = 0.25,
    **_: object,
) -> NestedGANLoss:
    sizes = tuple(int(v) for v in (fft_sizes or (32, 64, 128, 256)))
    return NestedGANLoss(
        lambda_time=float(lambda_time),
        lambda_mrstft=float(lambda_mrstft),
        fft_sizes=sizes,
        hop_fraction=float(hop_fraction),
    )


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
