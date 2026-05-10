"""Training factories for the channel-wise SepFormer artifact predictor.

The model is a compact dual-path Transformer (SepFormer) adapted for fMRI
gradient artifact removal on the seven-epoch Niazy proof-fit context.

See ``documentation/research_notes.md`` for the architectural rationale
and hyperparameter choices.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from facet.training.dataset import NPZContextArtifactDataset

# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------


class _SinusoidalPositionalEncoding(torch.nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017).

    Computed on the fly so it works for any sequence length up to
    ``max_len``.
    """

    def __init__(self, d_model: int, max_len: int = 8192) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


class _ChannelLayerNorm(torch.nn.Module):
    """LayerNorm-equivalent operating along dim 1 of a 4-D ``(B, C, S, K)`` tensor.

    Avoids ``nn.GroupNorm`` which calls a private ``_verify_batch_size``
    helper that mishandles traced tensor shapes in PyTorch ≥ 2.10.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize across the channel axis. Works for 3-D ``(B, C, T)``
        # and 4-D ``(B, C, S, K)`` tensors.
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        shape = [1, -1] + [1] * (x.dim() - 2)
        weight = self.weight.view(*shape)
        bias = self.bias.view(*shape)
        return x_hat * weight + bias


class _MultiHeadSelfAttention(torch.nn.Module):
    """Hand-written multi-head self-attention layer.

    Replaces ``nn.MultiheadAttention`` so that the trace graph remains
    structurally identical across re-traces. ``nn.MultiheadAttention``
    instantiates internal sub-modules during forward which makes
    ``torch.jit.trace(check_trace=True)`` complain about mangled module
    names.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = self.d_model // self.n_heads
        self.scale = self.d_head ** -0.5
        self.qkv = torch.nn.Linear(self.d_model, 3 * self.d_model)
        self.out_proj = torch.nn.Linear(self.d_model, self.d_model)
        self.attn_dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, c)
        return self.out_proj(out)


class _SBTransformerLayer(torch.nn.Module):
    """Pre-norm Transformer encoder layer used by SepFormer."""

    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.attn = _MultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = torch.nn.Dropout(dropout)

        self.norm2 = torch.nn.LayerNorm(d_model)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ffn),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_ffn, d_model),
        )
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.dropout1(self.attn(h))
        h = self.norm2(x)
        x = x + self.dropout2(self.ffn(h))
        return x


class _SBTransformerStack(torch.nn.Module):
    """A stack of ``num_layers`` pre-norm Transformer encoder layers with a
    shared sinusoidal positional encoding at the input."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        dropout: float,
        max_len: int,
        use_positional: bool,
    ) -> None:
        super().__init__()
        self.use_positional = bool(use_positional)
        self.pos_enc = _SinusoidalPositionalEncoding(d_model, max_len=max_len) if self.use_positional else None
        self.layers = torch.nn.ModuleList(
            [_SBTransformerLayer(d_model, n_heads, d_ffn, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        return x


class _DualPathBlock(torch.nn.Module):
    """One SepFormer dual-path block: intra-chunk then inter-chunk attention.

    Input/output tensor shape: ``(B, d_model, n_chunks, chunk_size)``.
    """

    def __init__(
        self,
        d_model: int,
        intra_layers: int,
        inter_layers: int,
        intra_heads: int,
        inter_heads: int,
        d_ffn: int,
        dropout: float,
        max_intra_len: int,
        max_inter_len: int,
        skip_around_intra: bool = True,
    ) -> None:
        super().__init__()
        self.intra = _SBTransformerStack(
            num_layers=intra_layers,
            d_model=d_model,
            n_heads=intra_heads,
            d_ffn=d_ffn,
            dropout=dropout,
            max_len=max_intra_len,
            use_positional=True,
        )
        self.intra_norm = _ChannelLayerNorm(d_model)
        self.intra_linear = torch.nn.Linear(d_model, d_model)

        self.inter = _SBTransformerStack(
            num_layers=inter_layers,
            d_model=d_model,
            n_heads=inter_heads,
            d_ffn=d_ffn,
            dropout=dropout,
            max_len=max_inter_len,
            use_positional=True,
        )
        self.inter_norm = _ChannelLayerNorm(d_model)
        self.inter_linear = torch.nn.Linear(d_model, d_model)

        self.skip_around_intra = bool(skip_around_intra)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, S, K) with C=d_model, S=n_chunks, K=chunk_size
        b, c, s, k = x.shape

        intra_in = x.permute(0, 2, 3, 1).reshape(b * s, k, c)
        intra_out = self.intra(intra_in)
        intra_out = self.intra_linear(intra_out)
        intra_out = intra_out.reshape(b, s, k, c).permute(0, 3, 1, 2).contiguous()
        intra_out = self.intra_norm(intra_out)
        if self.skip_around_intra:
            intra_out = intra_out + x

        inter_in = intra_out.permute(0, 3, 2, 1).reshape(b * k, s, c)
        inter_out = self.inter(inter_in)
        inter_out = self.inter_linear(inter_out)
        inter_out = inter_out.reshape(b, k, s, c).permute(0, 3, 2, 1).contiguous()
        inter_out = self.inter_norm(inter_out)
        return inter_out + intra_out


# ---------------------------------------------------------------------------
# Top-level SepFormer model
# ---------------------------------------------------------------------------


class SepFormerArtifactNet(torch.nn.Module):
    """Compact SepFormer that predicts the centre-epoch artifact.

    The model takes a 7-epoch channel-wise context of shape ``(B, 7, 1, S)``
    where ``S`` is ``epoch_samples`` (default 512). The seven epochs are
    flattened to a contiguous waveform of length ``7·S``, encoded by a
    strided 1D convolution, processed by ``N`` dual-path Transformer
    blocks, masked, decoded, and finally the centre epoch is sliced out
    as the artifact prediction of shape ``(B, 1, S)``.
    """

    def __init__(
        self,
        epoch_samples: int = 512,
        context_epochs: int = 7,
        encoder_channels: int = 128,
        encoder_kernel: int = 16,
        encoder_stride: int = 8,
        chunk_size: int = 64,
        n_blocks: int = 2,
        intra_layers: int = 4,
        inter_layers: int = 4,
        intra_heads: int = 4,
        inter_heads: int = 4,
        d_ffn: int = 256,
        dropout: float = 0.1,
        skip_around_intra: bool = True,
        mask_activation: str = "relu",
    ) -> None:
        super().__init__()
        if context_epochs < 1 or context_epochs % 2 == 0:
            raise ValueError("context_epochs must be a positive odd integer")
        if encoder_kernel % 2 != 0 or encoder_kernel <= 0:
            raise ValueError("encoder_kernel must be a positive even integer")
        if encoder_stride <= 0 or encoder_stride >= encoder_kernel:
            raise ValueError("encoder_stride must satisfy 0 < stride < encoder_kernel")
        if chunk_size <= 1:
            raise ValueError("chunk_size must be > 1")
        if encoder_channels % max(intra_heads, inter_heads) != 0:
            raise ValueError("encoder_channels must be divisible by every attention head count")

        self.epoch_samples = int(epoch_samples)
        self.context_epochs = int(context_epochs)
        self.center_index = self.context_epochs // 2
        self.encoder_channels = int(encoder_channels)
        self.encoder_kernel = int(encoder_kernel)
        self.encoder_stride = int(encoder_stride)
        self.chunk_size = int(chunk_size)
        self.chunk_hop = self.chunk_size // 2
        self.mask_activation = mask_activation.lower()

        total_samples = self.context_epochs * self.epoch_samples
        self.encoder = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.encoder_channels,
            kernel_size=self.encoder_kernel,
            stride=self.encoder_stride,
            padding=0,
            bias=False,
        )
        self.encoder_activation = torch.nn.ReLU()

        feature_length = (total_samples - self.encoder_kernel) // self.encoder_stride + 1
        self.feature_length = feature_length
        # Precompute static padding so the chunker can be expressed without
        # any tensor-shape conditionals (keeps torch.jit.trace happy).
        if feature_length < self.chunk_size:
            self.pre_chunk_pad = self.chunk_size - feature_length
            feature_length = self.chunk_size
        else:
            self.pre_chunk_pad = 0
        gap_needed = (self.chunk_hop - (feature_length - self.chunk_size) % self.chunk_hop) % self.chunk_hop
        self.chunk_gap = int(gap_needed)
        padded_length = feature_length + self.chunk_gap
        self.n_chunks = (padded_length - self.chunk_size) // self.chunk_hop + 1
        self.padded_feature_length = padded_length
        max_inter_len = max(8, self.n_chunks + 4)

        self.pre_mask_norm = _ChannelLayerNorm(self.encoder_channels)
        self.pre_mask_linear = torch.nn.Conv1d(self.encoder_channels, self.encoder_channels, kernel_size=1)

        self.blocks = torch.nn.ModuleList(
            [
                _DualPathBlock(
                    d_model=self.encoder_channels,
                    intra_layers=intra_layers,
                    inter_layers=inter_layers,
                    intra_heads=intra_heads,
                    inter_heads=inter_heads,
                    d_ffn=d_ffn,
                    dropout=dropout,
                    max_intra_len=self.chunk_size,
                    max_inter_len=max_inter_len,
                    skip_around_intra=skip_around_intra,
                )
                for _ in range(n_blocks)
            ]
        )

        self.mask_head = torch.nn.Conv1d(self.encoder_channels, self.encoder_channels, kernel_size=1)
        self.decoder = torch.nn.ConvTranspose1d(
            in_channels=self.encoder_channels,
            out_channels=1,
            kernel_size=self.encoder_kernel,
            stride=self.encoder_stride,
            padding=0,
            bias=False,
        )

        # Decoder output length is deterministic from the encoder geometry.
        self.target_length = self.context_epochs * self.epoch_samples
        self.decoded_length = (feature_length - 1) * self.encoder_stride + self.encoder_kernel
        self.decoded_pad_right = max(0, self.target_length - self.decoded_length)
        self.decoded_trim_right = max(0, self.decoded_length - self.target_length)

    def _chunk(self, x: torch.Tensor) -> torch.Tensor:
        # Static padding only: pre_chunk_pad + chunk_gap are precomputed at
        # construction time from epoch_samples / context_epochs / encoder
        # geometry, so the chunker contains no shape-conditional Python.
        total_pad = self.pre_chunk_pad + self.chunk_gap
        if total_pad > 0:
            x = torch.nn.functional.pad(x, (0, total_pad))
        unfolded = x.unfold(dimension=-1, size=self.chunk_size, step=self.chunk_hop)
        return unfolded.contiguous()

    def _overlap_add(self, chunks: torch.Tensor) -> torch.Tensor:
        # Use stored Python ints (not chunks.shape values, which become
        # traced tensors and break F.fold's kernel_size validation).
        b = chunks.shape[0]
        c = self.encoder_channels
        k = self.chunk_size
        s = self.n_chunks
        hop = self.chunk_hop
        total = self.padded_feature_length
        flat = chunks.permute(0, 1, 3, 2).reshape(b * c, k, s)
        out = torch.nn.functional.fold(
            flat,
            output_size=(1, total),
            kernel_size=(1, k),
            stride=(1, hop),
        )
        norm = torch.nn.functional.fold(
            torch.ones_like(flat),
            output_size=(1, total),
            kernel_size=(1, k),
            stride=(1, hop),
        )
        out = out.view(b, c, total) / norm.view(b, c, total).clamp_min(1.0)
        return out[..., : self.feature_length]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input contract: ``(B, context_epochs, 1, epoch_samples)``. No
        # shape-time guard here because torch.jit.trace traces the body
        # below as a single straight-line graph with all sizes already
        # determined by construction-time hyperparameters.
        b = x.shape[0]
        waveform = x.reshape(b, 1, self.context_epochs * self.epoch_samples)

        features = self.encoder_activation(self.encoder(waveform))
        features_to_mask = self.pre_mask_linear(self.pre_mask_norm(features))

        chunks = self._chunk(features_to_mask)
        for block in self.blocks:
            chunks = block(chunks)

        mask_features = self._overlap_add(chunks)
        mask = self.mask_head(mask_features)
        if self.mask_activation == "relu":
            mask = torch.relu(mask)
        elif self.mask_activation == "sigmoid":
            mask = torch.sigmoid(mask)

        masked = features * mask
        decoded = self.decoder(masked)
        # Both pad and trim amounts are precomputed Python ints, so neither
        # arithmetic operation reads a traced tensor shape.
        if self.decoded_pad_right > 0:
            decoded = torch.nn.functional.pad(decoded, (0, self.decoded_pad_right))
        if self.decoded_trim_right > 0:
            decoded = decoded[..., : self.target_length]

        decoded = decoded.reshape(b, self.context_epochs, 1, self.epoch_samples)
        return decoded[:, self.center_index, :, :]


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


class _NegSISNRLoss(torch.nn.Module):
    """Scale-invariant SI-SNR loss (negative dB so that lower is better)."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = prediction.reshape(prediction.shape[0], -1)
        tgt = target.reshape(target.shape[0], -1)

        pred = pred - pred.mean(dim=-1, keepdim=True)
        tgt = tgt - tgt.mean(dim=-1, keepdim=True)

        s_target = (
            (pred * tgt).sum(dim=-1, keepdim=True)
            / (tgt.pow(2).sum(dim=-1, keepdim=True) + self.eps)
        ) * tgt
        e_noise = pred - s_target
        ratio = s_target.pow(2).sum(dim=-1) / (e_noise.pow(2).sum(dim=-1) + self.eps)
        si_snr = 10.0 * torch.log10(ratio + self.eps)
        return -si_snr.mean()


class _SISNRPlusMSELoss(torch.nn.Module):
    """SI-SNR with a small MSE term to anchor amplitude scale."""

    def __init__(self, mse_weight: float = 0.1, eps: float = 1e-8) -> None:
        super().__init__()
        self.si_snr = _NegSISNRLoss(eps=eps)
        self.mse = torch.nn.MSELoss()
        self.mse_weight = float(mse_weight)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.si_snr(prediction, target) + self.mse_weight * self.mse(prediction, target)


def build_loss(name: str = "mse", **kwargs: Any) -> torch.nn.Module:
    """Loss factory referenced by the training YAML."""
    normalized = name.strip().lower()
    if normalized == "mse":
        return torch.nn.MSELoss()
    if normalized == "l1":
        return torch.nn.L1Loss()
    if normalized in {"smooth_l1", "huber"}:
        return torch.nn.SmoothL1Loss()
    if normalized in {"si_snr", "sisnr"}:
        return _NegSISNRLoss(eps=float(kwargs.get("eps", 1e-8)))
    if normalized in {"si_snr_mse", "sisnr_mse"}:
        return _SISNRPlusMSELoss(
            mse_weight=float(kwargs.get("mse_weight", 0.1)),
            eps=float(kwargs.get("eps", 1e-8)),
        )
    raise ValueError(f"Unsupported loss '{name}'")


# ---------------------------------------------------------------------------
# Channel-wise context dataset (same shape contract as cascaded_context_dae)
# ---------------------------------------------------------------------------


class _SubsetDataset:
    def __init__(self, parent: "ChannelWiseContextArtifactDataset", indices: list[int]) -> None:
        self._parent = parent
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._parent[self._indices[idx]]


class ChannelWiseContextArtifactDataset:
    """Expose ``(7, 1, S) -> (1, S)`` channel-wise examples for SepFormer.

    Re-implements the same interface as
    ``cascaded_context_dae.training.ChannelWiseContextArtifactDataset`` so
    the SepFormer model package stays self-contained (no cross-model
    imports). The underlying NPZ base dataset is the shared
    :class:`NPZContextArtifactDataset`.
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


# ---------------------------------------------------------------------------
# CLI factories
# ---------------------------------------------------------------------------


def build_model(
    input_shape: tuple[int, int, int] | None = None,
    target_shape: tuple[int, int] | None = None,
    epoch_samples: int | None = None,
    context_epochs: int | None = None,
    encoder_channels: int = 128,
    encoder_kernel: int = 16,
    encoder_stride: int = 8,
    chunk_size: int = 64,
    n_blocks: int = 2,
    intra_layers: int = 4,
    inter_layers: int = 4,
    intra_heads: int = 4,
    inter_heads: int = 4,
    d_ffn: int = 256,
    dropout: float = 0.1,
    skip_around_intra: bool = True,
    mask_activation: str = "relu",
    **_: object,
) -> SepFormerArtifactNet:
    resolved_context = context_epochs
    resolved_samples = epoch_samples
    if input_shape is not None:
        if len(input_shape) != 3:
            raise ValueError("input_shape must be (context_epochs, 1, samples)")
        resolved_context = resolved_context or int(input_shape[0])
        resolved_samples = resolved_samples or int(input_shape[-1])
    if resolved_context is None:
        resolved_context = 7
    if resolved_samples is None:
        raise ValueError("build_model requires epoch_samples or input_shape")
    return SepFormerArtifactNet(
        epoch_samples=int(resolved_samples),
        context_epochs=int(resolved_context),
        encoder_channels=int(encoder_channels),
        encoder_kernel=int(encoder_kernel),
        encoder_stride=int(encoder_stride),
        chunk_size=int(chunk_size),
        n_blocks=int(n_blocks),
        intra_layers=int(intra_layers),
        inter_layers=int(inter_layers),
        intra_heads=int(intra_heads),
        inter_heads=int(inter_heads),
        d_ffn=int(d_ffn),
        dropout=float(dropout),
        skip_around_intra=bool(skip_around_intra),
        mask_activation=str(mask_activation),
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
