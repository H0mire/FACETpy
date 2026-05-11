"""Training factories for the ViT Spectrogram Inpainter (`vit_spectrogram`).

This module exposes three factories consumed by ``facet-train fit``:

- :func:`build_model` constructs :class:`ViTSpectrogramInpainter`.
- :func:`build_loss` returns the optimization loss.
- :func:`build_dataset` materialises a channel-wise context dataset whose
  targets are the *clean* center epoch (suitable for the inpainting paradigm).

The model takes a per-channel 7-epoch context and returns the reconstructed
clean center epoch. Artifact subtraction is performed by the adapter at
inference time.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from facet.training.dataset import NPZContextArtifactDataset


# ---------------------------------------------------------------------------
# Dataset wrapper — exposes per-channel context examples with clean targets
# ---------------------------------------------------------------------------


class ChannelWiseSpectrogramDataset:
    """Expose ``(context_epochs, 1, samples) -> (1, samples)`` per-channel examples.

    Wraps an :class:`NPZContextArtifactDataset` configured with the
    ``clean_center`` target so each item gives the model both the noisy
    multi-epoch context input and the clean center-epoch target needed by the
    inpainting loss.
    """

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
        self.target_type = "clean"
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
    def __init__(self, parent: ChannelWiseSpectrogramDataset, indices: list[int]) -> None:
        self._parent = parent
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._parent[self._indices[idx]]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class _SelfAttention(torch.nn.Module):
    """Multi-head self-attention with an explicit Q/K/V projection.

    Implemented by hand so that ``torch.jit.trace`` produces a stable graph;
    :class:`torch.nn.MultiheadAttention` dispatches between native and
    Python paths based on tensor properties, which makes its traced graph
    differ across re-runs even though the numerical output is identical.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"embed_dim ({dim}) must be divisible by n_heads ({n_heads})")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = torch.nn.Linear(dim, dim * 3)
        self.attn_drop = torch.nn.Dropout(dropout)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        qkv = self.qkv(x).reshape(batch, seq, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch, seq, self.dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class _TransformerBlock(torch.nn.Module):
    """Pre-norm transformer block with self-attention and MLP."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attn = _SelfAttention(dim, n_heads, dropout)
        self.norm2 = torch.nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTSpectrogramInpainter(torch.nn.Module):
    """Vision-Transformer-based spectrogram inpainter for GA correction.

    Forward contract (matches the dataset's ``(input_shape, target_shape)``):

    - Input: ``(B, context_epochs, 1, epoch_samples)`` noisy per-channel
      EEG context.
    - Output: ``(B, 1, epoch_samples)`` reconstructed clean center epoch.

    Internally the model concatenates the seven epochs into a single
    per-channel time series, computes its STFT, treats the magnitude as a
    2-D image, applies a structural mask covering the center-epoch time
    region, predicts the clean log-magnitude through a small ViT encoder,
    and reconstructs the time-domain signal via iSTFT using the input's
    original phase. The center-epoch slice of that reconstruction is
    returned.

    Parameters
    ----------
    context_epochs : int
        Number of trigger-defined epochs in the input context (must be odd).
    epoch_samples : int
        Samples per epoch in the dataset.
    n_fft : int
        STFT window/FFT size in samples.
    hop_length : int
        STFT hop in samples. Must divide ``epoch_samples``.
    freq_bins : int
        Number of frequency rows kept (must be divisible by ``patch_freq``).
    time_frames : int
        Number of time columns kept (must be divisible by ``patch_time``).
    patch_freq, patch_time : int
        Spectrogram patch sizes along the frequency and time axes.
    embed_dim : int
        Transformer token width.
    depth : int
        Number of transformer encoder layers.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Hidden-to-embedding ratio inside the MLP block.
    dropout : float
        Dropout probability applied in attention and MLP.
    mask_margin_patches : int
        Extra patches included on each side of the center-epoch time region
        in the structural mask, to absorb STFT smearing at the boundary.
    """

    def __init__(
        self,
        context_epochs: int = 7,
        epoch_samples: int = 512,
        n_fft: int = 64,
        hop_length: int = 16,
        freq_bins: int = 32,
        time_frames: int = 224,
        patch_freq: int = 4,
        patch_time: int = 16,
        embed_dim: int = 192,
        depth: int = 6,
        n_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        mask_margin_patches: int = 1,
    ) -> None:
        super().__init__()
        if context_epochs % 2 == 0:
            raise ValueError("context_epochs must be odd")
        if freq_bins % patch_freq != 0:
            raise ValueError("freq_bins must be divisible by patch_freq")
        if time_frames % patch_time != 0:
            raise ValueError("time_frames must be divisible by patch_time")

        self.context_epochs = int(context_epochs)
        self.epoch_samples = int(epoch_samples)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.freq_bins = int(freq_bins)
        self.time_frames = int(time_frames)
        self.patch_freq = int(patch_freq)
        self.patch_time = int(patch_time)
        self.embed_dim = int(embed_dim)
        self.mask_margin_patches = int(mask_margin_patches)

        self.total_samples = self.context_epochs * self.epoch_samples
        self.center_epoch_idx = self.context_epochs // 2
        self.center_start_sample = self.center_epoch_idx * self.epoch_samples
        self.center_stop_sample = self.center_start_sample + self.epoch_samples

        self.n_freq_patches = self.freq_bins // self.patch_freq
        self.n_time_patches = self.time_frames // self.patch_time
        self.n_patches = self.n_freq_patches * self.n_time_patches
        self.patch_pixels = self.patch_freq * self.patch_time

        center_frame_start = self.center_start_sample // self.hop_length
        center_frame_stop = math.ceil(self.center_stop_sample / self.hop_length)
        time_patch_start = max(0, center_frame_start // self.patch_time - self.mask_margin_patches)
        time_patch_stop = min(
            self.n_time_patches,
            math.ceil(center_frame_stop / self.patch_time) + self.mask_margin_patches,
        )
        mask_buf = torch.zeros(self.n_patches, dtype=torch.bool)
        for t_idx in range(time_patch_start, time_patch_stop):
            for f_idx in range(self.n_freq_patches):
                patch_id = f_idx * self.n_time_patches + t_idx
                mask_buf[patch_id] = True
        self.register_buffer("patch_mask", mask_buf, persistent=False)

        self.register_buffer(
            "stft_window",
            torch.hann_window(self.n_fft, periodic=True),
            persistent=False,
        )

        self.patch_embed = torch.nn.Linear(self.patch_pixels, self.embed_dim)
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.freq_pos = torch.nn.Parameter(torch.zeros(1, self.n_freq_patches, 1, self.embed_dim))
        self.time_pos = torch.nn.Parameter(torch.zeros(1, 1, self.n_time_patches, self.embed_dim))
        self.norm_in = torch.nn.LayerNorm(self.embed_dim)
        self.blocks = torch.nn.ModuleList(
            [
                _TransformerBlock(self.embed_dim, int(n_heads), float(mlp_ratio), float(dropout))
                for _ in range(int(depth))
            ]
        )
        self.norm_out = torch.nn.LayerNorm(self.embed_dim)
        self.decoder_head = torch.nn.Linear(self.embed_dim, self.patch_pixels)

        torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        torch.nn.init.trunc_normal_(self.freq_pos, std=0.02)
        torch.nn.init.trunc_normal_(self.time_pos, std=0.02)

    @property
    def n_masked_patches(self) -> int:
        return int(self.patch_mask.sum().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        signal = x.reshape(batch, self.total_samples)

        Z = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.stft_window,
            center=True,
            return_complex=True,
        )
        full_freq_bins = Z.shape[-2]
        full_time_frames = Z.shape[-1]
        Z_cropped = Z[:, : self.freq_bins, : self.time_frames]
        magnitude = Z_cropped.abs()
        phase = torch.angle(Z_cropped)
        log_mag = torch.log1p(magnitude)

        patches = self._patchify(log_mag)
        tokens = self.patch_embed(patches)
        tokens = self._apply_mask(tokens)
        tokens = tokens + self._positional_embedding()
        tokens = self.norm_in(tokens)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm_out(tokens)
        predicted_patches = self.decoder_head(tokens)
        pred_log_mag = self._unpatchify(predicted_patches)
        pred_magnitude = torch.expm1(pred_log_mag).clamp(min=0.0)

        complex_spec = torch.polar(pred_magnitude, phase)
        complex_spec = torch.nn.functional.pad(
            complex_spec,
            (0, full_time_frames - self.time_frames, 0, full_freq_bins - self.freq_bins),
        )

        time_signal = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.stft_window,
            center=True,
            length=self.total_samples,
        )

        center = time_signal[:, self.center_start_sample : self.center_stop_sample]
        return center.unsqueeze(1)

    def _patchify(self, log_mag: torch.Tensor) -> torch.Tensor:
        batch = log_mag.shape[0]
        reshaped = log_mag.reshape(
            batch,
            self.n_freq_patches,
            self.patch_freq,
            self.n_time_patches,
            self.patch_time,
        )
        reshaped = reshaped.permute(0, 1, 3, 2, 4).contiguous()
        return reshaped.reshape(batch, self.n_patches, self.patch_pixels)

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch = patches.shape[0]
        reshaped = patches.reshape(
            batch,
            self.n_freq_patches,
            self.n_time_patches,
            self.patch_freq,
            self.patch_time,
        )
        reshaped = reshaped.permute(0, 1, 3, 2, 4).contiguous()
        return reshaped.reshape(batch, self.freq_bins, self.time_frames)

    def _apply_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = self.patch_mask.view(1, -1, 1)
        return torch.where(mask, self.mask_token.expand_as(tokens), tokens)

    def _positional_embedding(self) -> torch.Tensor:
        pos = self.freq_pos.expand(1, self.n_freq_patches, self.n_time_patches, self.embed_dim) + self.time_pos.expand(
            1, self.n_freq_patches, self.n_time_patches, self.embed_dim
        )
        return pos.reshape(1, self.n_patches, self.embed_dim)


# ---------------------------------------------------------------------------
# Factories used by the training CLI
# ---------------------------------------------------------------------------


def build_model(
    input_shape: tuple[int, int, int] | None = None,
    target_shape: tuple[int, int] | None = None,
    epoch_samples: int | None = None,
    context_epochs: int | None = None,
    n_fft: int = 64,
    hop_length: int = 16,
    freq_bins: int = 32,
    time_frames: int = 224,
    patch_freq: int = 4,
    patch_time: int = 16,
    embed_dim: int = 192,
    depth: int = 6,
    n_heads: int = 6,
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
    mask_margin_patches: int = 1,
    **_: object,
) -> ViTSpectrogramInpainter:
    if input_shape is not None:
        resolved_context_epochs = int(input_shape[0])
        resolved_epoch_samples = int(input_shape[-1])
    else:
        if context_epochs is None or epoch_samples is None:
            raise ValueError("build_model requires input_shape, or both context_epochs and epoch_samples")
        resolved_context_epochs = int(context_epochs)
        resolved_epoch_samples = int(epoch_samples)

    return ViTSpectrogramInpainter(
        context_epochs=resolved_context_epochs,
        epoch_samples=resolved_epoch_samples,
        n_fft=n_fft,
        hop_length=hop_length,
        freq_bins=freq_bins,
        time_frames=time_frames,
        patch_freq=patch_freq,
        patch_time=patch_time,
        embed_dim=embed_dim,
        depth=depth,
        n_heads=n_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        mask_margin_patches=mask_margin_patches,
    )


def build_loss(name: str = "mse", **_: object) -> torch.nn.Module:
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
) -> ChannelWiseSpectrogramDataset:
    dataset_path = Path(path or context_path or "").expanduser()
    if not str(dataset_path):
        raise ValueError("build_dataset requires path or context_path")
    base = NPZContextArtifactDataset(
        path=dataset_path,
        input_key="noisy_context",
        target_key="clean_center",
        demean_input=False,
        demean_target=False,
    )
    return ChannelWiseSpectrogramDataset(
        base,
        context_epochs=context_epochs,
        demean_input=demean_input,
        demean_target=demean_target,
        max_examples=max_examples,
    )
