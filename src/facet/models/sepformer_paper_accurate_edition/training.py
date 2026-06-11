"""Training factories for the *paper-accurate* SepFormer artifact predictor.

This is the paper-faithful edition of the SepFormer model
(Subakan et al., ICASSP 2021, arXiv:2010.13154, "Attention is All You Need
in Speech Separation"; canonical transformer math from Vaswani et al., 2017).
It does NOT modify the original ``facet.models.sepformer`` package.

What is more faithful here than in the original (see
``documentation/paper_accuracy_review.md`` for the full discrepancy table):

* The masking network now reproduces the *exact* Fig. 2 ordering with the
  three bracketing learned stages: ``LayerNorm + Linear`` BEFORE chunking,
  ``PReLU + Linear`` after the dual-path stack (before overlap-add), and a
  ``FeedForward + ReLU`` mask generator after overlap-add.
* The Intra/Inter transformers now apply the whole-stack residual
  ``f(z) = g^K(z + e) + z`` (Eq. 6): positional encoding is added once at
  the input, K pre-norm layers run, and the *pre-PE* input is added back
  across the whole stack.
* Feature-space chunking is documented and the default chunk size is tied to
  the feature length (the paper's ``C=250`` is speech-scale and inappropriate
  for the ~hundreds-of-frames EEG feature sequence). The misleading
  "7 epochs = 7 chunks" framing of the original research notes is corrected:
  chunking happens in *feature* space, not epoch space.
* The default loss is the paper's scale-invariant SI-SNR with the 30 dB
  clipping ``clamp(si_snr, max=30)`` before negation. PIT is documented as
  inapplicable (a single target, no source-permutation ambiguity).

Deliberate, documented deviations (kept for EEG-fMRI suitability / CPU cost):

* Compact capacity (``d_model=128``, ``d_ffn=512``, 4-8 heads, ``K=8/4``)
  instead of the paper's ``d_model=256``, ``d_ffn=1024``, 8 heads, ``K=8/8``
  (~26M params). Justified by the ~hundreds-of-example proof-fit set and CPU
  feasibility; the paper values are exposed as a documented option.
* Single multiplicative ReLU mask (``Ns=1``). Artifact removal is the
  degenerate single-source case of separation; PIT and Dynamic Mixing are
  dropped (speech-specific). See the review document.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Core building blocks (ported from the original; trace-safe by construction)
# ---------------------------------------------------------------------------


class _SinusoidalPositionalEncoding(torch.nn.Module):
    """Canonical sinusoidal positional encoding (Vaswani et al., 2017, Eq.).

    ``PE(pos, 2i) = sin(pos / 10000^(2i/d))`` and the cosine for odd indices.
    Computed up to ``max_len`` and sliced on the fly so it works for any
    sequence length.
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
    """LayerNorm operating along dim 1 (the feature/channel axis).

    Works for 3-D ``(B, C, T)`` and 4-D ``(B, C, S, K)`` tensors. Avoids
    ``nn.GroupNorm`` which calls a private ``_verify_batch_size`` helper that
    mishandles traced tensor shapes in PyTorch >= 2.10.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        shape = [1, -1] + [1] * (x.dim() - 2)
        weight = self.weight.view(*shape)
        bias = self.bias.view(*shape)
        return x_hat * weight + bias


class _MultiHeadSelfAttention(torch.nn.Module):
    """Hand-written multi-head self-attention (Vaswani scaled dot-product).

    Mathematically identical to ``nn.MultiheadAttention`` (1/sqrt(d_head)
    scaling, softmax over keys, standard head split/concat) but implemented
    explicitly so the trace graph stays structurally identical across
    re-traces. ``nn.MultiheadAttention`` instantiates internal sub-modules
    during forward which makes ``torch.jit.trace(check_trace=True)`` complain
    about mangled module names.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = self.d_model // self.n_heads
        self.scale = self.d_head**-0.5
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
    """Pre-norm Transformer encoder layer used by SepFormer.

    SepFormer modifies the original Vaswani POST-norm sublayer
    ``LayerNorm(x + Sublayer(x))`` to PRE-norm
    ``x + Sublayer(LayerNorm(x))`` (paper Sec. 2.3). The feed-forward inner
    activation is ReLU (paper default); GELU is exposed as a documented switch.
    """

    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float, ffn_activation: str = "relu") -> None:
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.attn = _MultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = torch.nn.Dropout(dropout)

        act = torch.nn.GELU() if str(ffn_activation).lower() == "gelu" else torch.nn.ReLU()
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ffn),
            act,
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
    """A stack of ``num_layers`` pre-norm layers with the whole-stack residual.

    Implements the paper's Eq. 6: ``f(z) = g^K(z + e) + z``. The positional
    encoding ``e`` is added once at the input, the K pre-norm layers (``g^K``)
    run, and the *pre-PE* input ``z`` is added back across the entire stack to
    improve gradient backpropagation. The original FACETpy SepFormer omitted
    this outer residual.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        dropout: float,
        max_len: int,
        use_positional: bool,
        whole_stack_residual: bool = True,
        ffn_activation: str = "relu",
    ) -> None:
        super().__init__()
        self.use_positional = bool(use_positional)
        self.whole_stack_residual = bool(whole_stack_residual)
        self.pos_enc = _SinusoidalPositionalEncoding(d_model, max_len=max_len) if self.use_positional else None
        self.layers = torch.nn.ModuleList(
            [_SBTransformerLayer(d_model, n_heads, d_ffn, dropout, ffn_activation=ffn_activation) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        if self.whole_stack_residual:
            x = x + residual
        return x


class _DualPathBlock(torch.nn.Module):
    """One SepFormer dual-path block: intra-chunk then inter-chunk transformer.

    Input/output tensor shape: ``(B, d_model, n_chunks, chunk_size)``.
    ``skip_around_intra`` (SpeechBrain default ``True``) keeps the encoder
    representation accessible to the InterTransformer.
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
        whole_stack_residual: bool = True,
        ffn_activation: str = "relu",
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
            whole_stack_residual=whole_stack_residual,
            ffn_activation=ffn_activation,
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
            whole_stack_residual=whole_stack_residual,
            ffn_activation=ffn_activation,
        )
        self.inter_norm = _ChannelLayerNorm(d_model)
        self.inter_linear = torch.nn.Linear(d_model, d_model)

        self.skip_around_intra = bool(skip_around_intra)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, S, K) with C=d_model, S=n_chunks, K=chunk_size
        b, c, s, k = x.shape

        # IntraTransformer: attention WITHIN each chunk (along chunk_size axis).
        intra_in = x.permute(0, 2, 3, 1).reshape(b * s, k, c)
        intra_out = self.intra(intra_in)
        intra_out = self.intra_linear(intra_out)
        intra_out = intra_out.reshape(b, s, k, c).permute(0, 3, 1, 2).contiguous()
        intra_out = self.intra_norm(intra_out)
        if self.skip_around_intra:
            intra_out = intra_out + x

        # InterTransformer: attention ACROSS chunks (along n_chunks axis).
        inter_in = intra_out.permute(0, 3, 2, 1).reshape(b * k, s, c)
        inter_out = self.inter(inter_in)
        inter_out = self.inter_linear(inter_out)
        inter_out = inter_out.reshape(b, k, s, c).permute(0, 3, 2, 1).contiguous()
        inter_out = self.inter_norm(inter_out)
        return inter_out + intra_out


# ---------------------------------------------------------------------------
# Top-level paper-accurate SepFormer model
# ---------------------------------------------------------------------------


class SepFormerPaperAccurateNet(torch.nn.Module):
    """Paper-faithful SepFormer that predicts the centre-epoch artifact.

    Input contract: a context of shape ``(B, context_epochs, 1, S)`` where
    ``S`` is ``epoch_samples``. The epochs are flattened to a contiguous
    waveform of length ``context_epochs * S`` and run through the full Fig. 2
    masking pipeline:

        encoder ReLU(Conv1d)
        -> LayerNorm + Linear            (pre-chunk bracketing stage)
        -> Chunking (FEATURE axis, 50% overlap)
        -> N x DualPathBlock(IntraT -> InterT, skip_around_intra)
        -> PReLU + Linear                (post-block bracketing stage)
        -> OverlapAdd
        -> FeedForward + ReLU            (mask-generation stage, Ns=1 mask)
        -> mask * encoder_features
        -> ConvTranspose1d decode (same kernel/stride as encoder)

    The decoded waveform is reshaped to ``(B, context_epochs, 1, S)`` and the
    centre epoch is sliced out as the ``(B, 1, S)`` artifact prediction.
    """

    def __init__(
        self,
        epoch_samples: int = 512,
        context_epochs: int = 7,
        encoder_channels: int = 128,
        encoder_kernel: int = 16,
        encoder_stride: int = 8,
        chunk_size: int | None = None,
        n_blocks: int = 2,
        intra_layers: int = 8,
        inter_layers: int = 4,
        intra_heads: int = 8,
        inter_heads: int = 8,
        d_ffn: int = 512,
        dropout: float = 0.1,
        skip_around_intra: bool = True,
        whole_stack_residual: bool = True,
        mask_activation: str = "relu",
        ffn_activation: str = "relu",
    ) -> None:
        super().__init__()
        if context_epochs < 1 or context_epochs % 2 == 0:
            raise ValueError("context_epochs must be a positive odd integer")
        if encoder_kernel % 2 != 0 or encoder_kernel <= 0:
            raise ValueError("encoder_kernel must be a positive even integer")
        if encoder_stride <= 0 or encoder_stride >= encoder_kernel:
            raise ValueError("encoder_stride must satisfy 0 < stride < encoder_kernel")
        if encoder_channels % max(intra_heads, inter_heads) != 0:
            raise ValueError("encoder_channels must be divisible by every attention head count")

        self.epoch_samples = int(epoch_samples)
        self.context_epochs = int(context_epochs)
        self.center_index = self.context_epochs // 2
        self.encoder_channels = int(encoder_channels)
        self.encoder_kernel = int(encoder_kernel)
        self.encoder_stride = int(encoder_stride)
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

        # Feature-space chunking. The paper's C=250 is speech-scale; for the
        # ~hundreds-of-frames EEG feature sequence we tie C to the feature
        # length (~feature_length//8, clamped to [16, 64]) so the model has a
        # small but meaningful number of overlapping chunks. Chunking is in
        # FEATURE space, never epoch space.
        if chunk_size is None:
            chunk_size = int(min(64, max(16, feature_length // 8)))
        chunk_size = int(chunk_size)
        if chunk_size <= 1:
            raise ValueError("chunk_size must be > 1")
        # Keep chunk_size from exceeding the (possibly tiny) feature length so
        # the static padding below stays bounded; allow up to feature_length.
        chunk_size = min(chunk_size, max(2, feature_length))
        if chunk_size % 2 != 0:
            chunk_size -= 1  # 50% hop needs an even chunk size
        if chunk_size < 2:
            chunk_size = 2
        self.chunk_size = chunk_size
        self.chunk_hop = self.chunk_size // 2

        # Precompute static padding so the chunker has no shape-conditional
        # Python (keeps torch.jit.trace happy).
        padded_feature_length = feature_length
        if padded_feature_length < self.chunk_size:
            self.pre_chunk_pad = self.chunk_size - padded_feature_length
            padded_feature_length = self.chunk_size
        else:
            self.pre_chunk_pad = 0
        gap_needed = (self.chunk_hop - (padded_feature_length - self.chunk_size) % self.chunk_hop) % self.chunk_hop
        self.chunk_gap = int(gap_needed)
        padded_length = padded_feature_length + self.chunk_gap
        self.n_chunks = (padded_length - self.chunk_size) // self.chunk_hop + 1
        self.padded_feature_length = padded_length
        max_inter_len = max(8, self.n_chunks + 4)

        # --- Fig. 2 bracketing stage 1: LayerNorm + Linear BEFORE chunking ---
        self.pre_chunk_norm = _ChannelLayerNorm(self.encoder_channels)
        self.pre_chunk_linear = torch.nn.Conv1d(self.encoder_channels, self.encoder_channels, kernel_size=1)

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
                    whole_stack_residual=whole_stack_residual,
                    ffn_activation=ffn_activation,
                )
                for _ in range(n_blocks)
            ]
        )

        # --- Fig. 2 bracketing stage 2: PReLU + Linear after the block stack,
        # applied on the chunked feature axis, BEFORE overlap-add. ---
        self.post_block_prelu = torch.nn.PReLU()
        self.post_block_linear = torch.nn.Conv1d(self.encoder_channels, self.encoder_channels, kernel_size=1)

        # --- Fig. 2 bracketing stage 3: FeedForward + ReLU mask generator
        # after overlap-add (two 1x1 convs with a ReLU in between, then the
        # mask activation). ---
        self.mask_ffn = torch.nn.Sequential(
            torch.nn.Conv1d(self.encoder_channels, self.encoder_channels, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(self.encoder_channels, self.encoder_channels, kernel_size=1),
        )

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
        total_pad = self.pre_chunk_pad + self.chunk_gap
        if total_pad > 0:
            x = torch.nn.functional.pad(x, (0, total_pad))
        unfolded = x.unfold(dimension=-1, size=self.chunk_size, step=self.chunk_hop)
        return unfolded.contiguous()

    def _overlap_add(self, chunks: torch.Tensor) -> torch.Tensor:
        # Use stored Python ints (not chunks.shape values, which become traced
        # tensors and break F.fold's kernel_size validation).
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
        b = x.shape[0]
        waveform = x.reshape(b, 1, self.context_epochs * self.epoch_samples)

        # Encoder (learned STFT-like front-end).
        features = self.encoder_activation(self.encoder(waveform))

        # Fig. 2 stage 1: LayerNorm + Linear before chunking.
        chunk_input = self.pre_chunk_linear(self.pre_chunk_norm(features))

        # Chunking on the FEATURE axis, 50% overlap.
        chunks = self._chunk(chunk_input)

        # N dual-path SepFormer blocks.
        for block in self.blocks:
            chunks = block(chunks)

        # Fig. 2 stage 2: PReLU + Linear on the chunked features (before
        # overlap-add). Applied per (chunk, frame) over the feature axis: fold
        # the 4-D chunk tensor to 3-D so the 1x1 conv acts on the feature dim.
        b4, c4, s4, k4 = chunks.shape
        flat = chunks.reshape(b4, c4, s4 * k4)
        flat = self.post_block_linear(self.post_block_prelu(flat))
        chunks = flat.reshape(b4, c4, s4, k4)

        # OverlapAdd reassembly.
        mask_features = self._overlap_add(chunks)

        # Fig. 2 stage 3: FeedForward + ReLU mask generator.
        mask = self.mask_ffn(mask_features)
        if self.mask_activation == "relu":
            mask = torch.relu(mask)
        elif self.mask_activation == "sigmoid":
            mask = torch.sigmoid(mask)

        # Multiplicative source mask against the ORIGINAL encoder features.
        masked = features * mask

        # Decoder.
        decoded = self.decoder(masked)
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
    """Scale-invariant SI-SNR loss with the paper's 30 dB clipping.

    Returns the negative SI-SNR (lower is better). The SI-SNR is clamped at
    ``si_snr_max`` dB (paper default 30 dB) BEFORE negation to stabilise
    training, exactly as in the SepFormer paper. PIT is inapplicable here:
    there is a single target (the centre-epoch artifact), so there is no
    source-permutation ambiguity to resolve.
    """

    def __init__(self, eps: float = 1e-8, si_snr_max: float = 30.0) -> None:
        super().__init__()
        self.eps = eps
        self.si_snr_max = float(si_snr_max)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = prediction.reshape(prediction.shape[0], -1)
        tgt = target.reshape(target.shape[0], -1)

        pred = pred - pred.mean(dim=-1, keepdim=True)
        tgt = tgt - tgt.mean(dim=-1, keepdim=True)

        s_target = ((pred * tgt).sum(dim=-1, keepdim=True) / (tgt.pow(2).sum(dim=-1, keepdim=True) + self.eps)) * tgt
        e_noise = pred - s_target
        ratio = s_target.pow(2).sum(dim=-1) / (e_noise.pow(2).sum(dim=-1) + self.eps)
        si_snr = 10.0 * torch.log10(ratio + self.eps)
        si_snr = torch.clamp(si_snr, max=self.si_snr_max)
        return -si_snr.mean()


class _SISNRPlusMSELoss(torch.nn.Module):
    """30 dB-clipped SI-SNR with a small MSE term to anchor amplitude scale.

    SI-SNR is scale-invariant; for EEG-fMRI the predicted artifact is
    subtracted from the signal, so absolute amplitude is load-bearing. The
    MSE term re-anchors the amplitude.
    """

    def __init__(self, mse_weight: float = 0.1, eps: float = 1e-8, si_snr_max: float = 30.0) -> None:
        super().__init__()
        self.si_snr = _NegSISNRLoss(eps=eps, si_snr_max=si_snr_max)
        self.mse = torch.nn.MSELoss()
        self.mse_weight = float(mse_weight)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.si_snr(prediction, target) + self.mse_weight * self.mse(prediction, target)


def build_loss(name: str = "si_snr", **kwargs: Any) -> torch.nn.Module:
    """Loss factory referenced by the training YAML.

    The paper-faithful default is the 30 dB-clipped negative SI-SNR. MSE,
    L1, SmoothL1 and SI-SNR+MSE remain selectable (MSE/SI-SNR+MSE anchor the
    absolute amplitude that EEG-fMRI artifact subtraction depends on).
    """
    normalized = name.strip().lower()
    if normalized == "mse":
        return torch.nn.MSELoss()
    if normalized == "l1":
        return torch.nn.L1Loss()
    if normalized in {"smooth_l1", "huber"}:
        return torch.nn.SmoothL1Loss()
    if normalized in {"si_snr", "sisnr"}:
        return _NegSISNRLoss(
            eps=float(kwargs.get("eps", 1e-8)),
            si_snr_max=float(kwargs.get("si_snr_max", 30.0)),
        )
    if normalized in {"si_snr_mse", "sisnr_mse"}:
        return _SISNRPlusMSELoss(
            mse_weight=float(kwargs.get("mse_weight", 0.1)),
            eps=float(kwargs.get("eps", 1e-8)),
            si_snr_max=float(kwargs.get("si_snr_max", 30.0)),
        )
    raise ValueError(f"Unsupported loss '{name}'")


# ---------------------------------------------------------------------------
# Channel-wise context dataset (self-contained, proof-fit-bundle compatible)
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
    """Expose ``(context_epochs, 1, S) -> (1, S)`` channel-wise examples.

    Reads the Niazy proof-fit bundle (``noisy_center``, ``clean_center``,
    ``artifact_center`` each ``(n_examples, n_channels, n_samples)``, plus
    ``sfreq``) and builds sliding odd-length context windows over the example
    axis on the fly. This makes the package self-contained and works with the
    proof-fit bundle directly (no separate 4-D ``noisy_context`` key needed).

    ``target_type`` selects ``artifact_center`` (default, faithful to the
    artifact-removal task) or ``clean_center``. The centre epoch of each
    window is the prediction target.
    """

    def __init__(
        self,
        noisy_center: np.ndarray,
        target_center: np.ndarray,
        *,
        target_type: str = "artifact",
        context_epochs: int = 7,
        demean_input: bool = True,
        demean_target: bool = True,
        sfreq: float = float("nan"),
        max_examples: int | None = None,
    ) -> None:
        if context_epochs < 1 or context_epochs % 2 == 0:
            raise ValueError("context_epochs must be a positive odd integer")
        noisy = np.asarray(noisy_center, dtype=np.float32)
        target = np.asarray(target_center, dtype=np.float32)
        if noisy.ndim != 3 or target.ndim != 3:
            raise ValueError("noisy_center and target_center must each have shape (n_examples, n_channels, n_samples)")
        if noisy.shape != target.shape:
            raise ValueError("noisy_center and target_center must have identical shapes")

        self._noisy = noisy
        self._target = target
        self.context_epochs = int(context_epochs)
        self.demean_input = bool(demean_input)
        self.demean_target = bool(demean_target)
        self.target_type = str(target_type)
        self.trigger_aligned = True
        self.sfreq = float(sfreq)

        self._radius = self.context_epochs // 2
        n_examples = int(noisy.shape[0])
        n_centers = n_examples - 2 * self._radius
        if n_centers < 1:
            raise ValueError(
                f"Need at least {self.context_epochs} examples to build one context window, got {n_examples}"
            )
        self.n_channels = int(noisy.shape[1])
        self.epoch_samples = int(noisy.shape[2])
        self.chunk_size = self.epoch_samples  # native epoch length (sample-axis chunk semantics)

        total = n_centers * self.n_channels
        self._length = total if max_examples is None else max(0, min(int(max_examples), total))
        self._n_centers = n_centers

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        center_local = int(idx) // self.n_channels
        channel_idx = int(idx) % self.n_channels
        center_example = center_local + self._radius
        lo = center_example - self._radius
        hi = center_example + self._radius + 1

        noisy_window = self._noisy[lo:hi, channel_idx, :]  # (context_epochs, S)
        noisy_out = noisy_window[:, None, :].astype(np.float32, copy=True)  # (context_epochs, 1, S)
        target_out = self._target[center_example, channel_idx : channel_idx + 1, :].astype(np.float32, copy=True)

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
    chunk_size: int | None = None,
    n_blocks: int = 2,
    intra_layers: int = 8,
    inter_layers: int = 4,
    intra_heads: int = 8,
    inter_heads: int = 8,
    d_ffn: int = 512,
    dropout: float = 0.1,
    skip_around_intra: bool = True,
    whole_stack_residual: bool = True,
    mask_activation: str = "relu",
    ffn_activation: str = "relu",
    **_: object,
) -> SepFormerPaperAccurateNet:
    """Construct the paper-accurate SepFormer.

    Accepts every facet-train injected kwarg via ``**_`` (n_channels, sfreq,
    target_type, training_config, target_shape, ...). ``context_epochs`` and
    ``epoch_samples`` are resolved from ``input_shape`` when not given.
    Defaults follow the paper's structure (whole-stack residual,
    IntraT depth 8 > InterT depth 4 per the ablation) at a compact,
    dataset-sensible capacity (see module docstring / review).
    """
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
    return SepFormerPaperAccurateNet(
        epoch_samples=int(resolved_samples),
        context_epochs=int(resolved_context),
        encoder_channels=int(encoder_channels),
        encoder_kernel=int(encoder_kernel),
        encoder_stride=int(encoder_stride),
        chunk_size=None if chunk_size is None else int(chunk_size),
        n_blocks=int(n_blocks),
        intra_layers=int(intra_layers),
        inter_layers=int(inter_layers),
        intra_heads=int(intra_heads),
        inter_heads=int(inter_heads),
        d_ffn=int(d_ffn),
        dropout=float(dropout),
        skip_around_intra=bool(skip_around_intra),
        whole_stack_residual=bool(whole_stack_residual),
        mask_activation=str(mask_activation),
        ffn_activation=str(ffn_activation),
    )


def _load_center_arrays(path: Path, target_type: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Load the proof-fit bundle and return (noisy_center, target_center, sfreq).

    Supports the bundle that stores 3-D ``*_center`` arrays directly. If only
    the 4-D ``noisy_context`` key is present (older bundles), derive the
    centre arrays from it.
    """
    with np.load(path, allow_pickle=False) as bundle:
        keys = set(bundle.files)
        sfreq = float(bundle["sfreq"][0]) if "sfreq" in keys else float("nan")
        target_key = "clean_center" if str(target_type).lower() == "clean" else "artifact_center"

        if "noisy_center" in keys and target_key in keys:
            noisy = bundle["noisy_center"].astype(np.float32, copy=False)
            target = bundle[target_key].astype(np.float32, copy=False)
            return noisy, target, sfreq

        # Fallback: derive centre arrays from the 4-D context cube.
        if "noisy_context" in keys:
            ctx = bundle["noisy_context"].astype(np.float32, copy=False)  # (N, E, C, S)
            center = ctx.shape[1] // 2
            noisy = ctx[:, center]
            if target_key in keys:
                target = bundle[target_key].astype(np.float32, copy=False)
            elif "clean_context" in keys and target_key == "clean_center":
                target = bundle["clean_context"].astype(np.float32, copy=False)[:, center]
            elif "artifact_context" in keys:
                target = bundle["artifact_context"].astype(np.float32, copy=False)[:, center]
            else:
                raise KeyError(f"Bundle missing a target source for target_type='{target_type}'")
            return noisy, target, sfreq

    raise KeyError(
        f"NPZ bundle '{path}' must contain 'noisy_center'+'{target_key}' (or a 'noisy_context' cube)"
    )


def build_dataset(
    path: str | None = None,
    context_path: str | None = None,
    context_epochs: int = 7,
    max_examples: int | None = None,
    demean_input: bool = True,
    demean_target: bool = True,
    target_type: str = "artifact",
    **_: object,
) -> ChannelWiseContextArtifactDataset:
    """Dataset factory referenced by the training YAML.

    Builds channel-wise sliding context windows from the Niazy proof-fit
    bundle. ``target_type`` ("artifact" or "clean") selects the target source.
    """
    dataset_path = Path(path or context_path or "").expanduser()
    if not str(dataset_path) or str(dataset_path) == ".":
        raise ValueError("build_dataset requires path or context_path")
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    noisy_center, target_center, sfreq = _load_center_arrays(dataset_path, target_type)
    return ChannelWiseContextArtifactDataset(
        noisy_center,
        target_center,
        target_type=str(target_type),
        context_epochs=context_epochs,
        demean_input=demean_input,
        demean_target=demean_target,
        sfreq=sfreq,
        max_examples=max_examples,
    )
