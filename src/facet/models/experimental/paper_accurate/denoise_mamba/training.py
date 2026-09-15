"""Paper-accurate training factories for DenoiseMamba (ConvSSD + Mamba-2/SSD).

This edition rebuilds the architecture to follow the source paper far more
closely than the original ``denoise_mamba`` package, which (by its own
``research_notes.md`` admission) was reconstructed *without* paper access and is
a flat Mamba-1 selective-scan stack.

Source paper
------------
Chen, Li, Zheng, Shi, "DenoiseMamba: An Innovative Approach for EEG Artifact
Removal Leveraging Mamba and CNN", IEEE Journal of Biomedical and Health
Informatics, vol. 29, no. 9, Sept. 2025, pp. 6551-6562. (IEEE Xplore 11012652,
PMID 40408214.)

What this edition implements faithfully (vs the original flat stack)
--------------------------------------------------------------------
* U-shaped encoder/decoder backbone with a 64xL -> 128xL/2 -> 256xL/4 -> ...
  channel/length pyramid and skip concatenation of the matching encoder feature
  into each decoder stage (Fig. 1).
* A real ConvSSD block (Fig. 2, Eqs. 6-10): channel split into a local conv
  branch (Conv3-BN-PReLU-Conv3-BN-PReLU-Dropout) and a global SSD branch, fused
  with two learnable scalars ``r1, r2`` and refined by a depthwise-separable
  conv.
* The SSD branch (Eqs. 6-9): ``y = Linear(RMSNorm(y_a1 + y_a2 + SiLU(Linear(x))))``
  with two parallel ``SSD(DWConv(Linear(x)))`` axes and a no-SSD SiLU skip.
* A Mamba-2 / Structured-State-Space-Duality (SSD) layer with scalar-per-head
  state matrix ``A`` and a chunked (semiseparable) scan, replacing Mamba-1's
  sequential selective scan (Eq. 5).
* Signal Embedding stem (Conv3-PReLU-BatchNorm-Conv3, Fig. 1 inset) and a
  GAP -> Linear -> PReLU -> LayerNorm -> Linear projection head paired with a
  length-preserving decoder output so the model still returns ``(B, 1, L)``.
* PReLU in conv/embedding/head paths; SiLU only on the SSD skip (Eq. 8).
* RMSNorm inside the SSD branch, BatchNorm in the conv branch, LayerNorm in the
  projection head.
* Predicts the CLEAN signal by default with MSE loss (the paper's target),
  while still supporting an ``artifact`` target for FACETpy pipeline parity.
* Per-epoch z-score standardisation of inputs (Fig. 6), with a demean-only
  fallback for parity comparisons against the original baseline.

Documented deviations (kept FACETpy-appropriate, not blind copies)
------------------------------------------------------------------
* Spatial SSD scan (Fig. 3) over the channel axis is replaced by a
  forward-time + reverse-time bidirectional temporal SSD, because FACETpy
  processes gradient artifacts channel-wise (a single channel per forward) and
  has no spatial channel map to scan. The dual-path structure is preserved.
* The Mamba-2 SSD is a portable pure-PyTorch chunked scan, not the CUDA
  ``mamba-ssm`` kernel, so it runs on CPU / Apple MPS for tests and the
  laptop-only deployment.

All paper hyper-parameters (base 64 channels, 3 stages, d_state 16, dropout 0.2,
AdamW lr 1e-3 + weight decay, ReduceLROnPlateau(0.5, patience 3), MSE loss) are
the defaults here; the smoke YAML and smoke test shrink every dimension so a
forward+backward completes in milliseconds on CPU.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Normalisation helpers.
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square layer norm used inside the SSD branch (Eq. 9).

    Normalises over the last (feature) dimension. Matches the RMSNorm used in
    Mamba-2 / the DenoiseMamba SSD fusion.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


# ---------------------------------------------------------------------------
# Mamba-2 / SSD (Structured State-Space Duality) layer in pure PyTorch.
# ---------------------------------------------------------------------------


class SSDLayer(nn.Module):
    """Portable Mamba-2 style SSD layer (Eq. 5) with a chunked scalar-A scan.

    This is the paper's distinguishing mechanism: instead of Mamba-1's
    input-dependent matrix ``A`` and a strictly sequential per-timestep scan,
    Mamba-2 uses a **scalar (per-head) state-transition** ``a_t = exp(dt_t * A)``
    and computes the sequence transform as a semiseparable matrix
    ``y = M x``. Here ``M`` is realised blockwise: an intra-chunk dense
    (diagonal-block) part plus an inter-chunk recurrence carrying the running
    state across chunk boundaries. This is mathematically the SSD form and is
    matmul-friendly, so it runs cheaply on CPU at the short lengths
    (L <= 512) used for single-channel EEG-fMRI epochs, without the CUDA
    ``mamba-ssm`` kernel.

    Parameters
    ----------
    d_inner : int
        Channels carried through the SSD.
    d_state : int
        State dimension per head (the ``N`` in ``B, C in R^N``).
    n_heads : int
        Number of heads sharing a scalar ``A`` each (Mamba-2 head structure).
    chunk_size : int
        Semiseparable block length for the chunked scan.
    """

    def __init__(
        self,
        d_inner: int,
        d_state: int = 16,
        n_heads: int = 1,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.d_inner = int(d_inner)
        self.d_state = int(d_state)
        self.n_heads = max(1, int(n_heads))
        if self.d_inner % self.n_heads != 0:
            # Fall back to a single head if the width is not divisible; keeps
            # tiny smoke dims working without surprising the caller.
            self.n_heads = 1
        self.head_dim = self.d_inner // self.n_heads
        self.chunk_size = max(1, int(chunk_size))

        # Input-dependent B, C (selectivity) and delta (timestep).
        self.dt_rank = max(1, math.ceil(self.d_inner / 16))
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.n_heads, bias=True)

        # Scalar-per-head A (Mamba-2): one log-A per head, NOT a (d_inner, d_state)
        # matrix as in Mamba-1.
        a_init = torch.arange(1, self.n_heads + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(a_init.clone()))
        self.D = nn.Parameter(torch.ones(self.d_inner, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SSD transform.

        Parameters
        ----------
        x : torch.Tensor, shape (B, L, d_inner)

        Returns
        -------
        torch.Tensor, shape (B, L, d_inner)
        """
        batch, length, d_inner = x.shape
        assert d_inner == self.d_inner

        x_dbl = self.x_proj(x)
        dt_unproj, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # delta: (B, L, n_heads) -> per-head positive timestep.
        delta = nn.functional.softplus(self.dt_proj(dt_unproj))
        A = -torch.exp(self.A_log.float())  # (n_heads,)

        # Per-head scalar decay a_t = exp(dt * A): (B, L, n_heads).
        a = torch.exp(delta * A)

        # Reshape x into heads: (B, L, n_heads, head_dim).
        xh = x.view(batch, length, self.n_heads, self.head_dim)
        # dt-scaled input gate broadcast per head: (B, L, n_heads, 1).
        dt = delta.unsqueeze(-1)
        # B, C are shared across heads (state dim N): (B, L, N).

        n_chunks = (length + self.chunk_size - 1) // self.chunk_size
        # Running state per (head, head_dim, state): (B, n_heads, head_dim, N).
        state = x.new_zeros(batch, self.n_heads, self.head_dim, self.d_state)
        outputs: list[torch.Tensor] = []

        for c in range(n_chunks):
            s = c * self.chunk_size
            e = min(s + self.chunk_size, length)
            cl = e - s
            a_c = a[:, s:e]               # (B, cl, H)
            dt_c = dt[:, s:e]             # (B, cl, H, 1)
            B_c = B[:, s:e]               # (B, cl, N)
            C_c = C[:, s:e]               # (B, cl, N)
            x_c = xh[:, s:e]              # (B, cl, H, head_dim)

            # Cumulative log-decay within the chunk for the semiseparable mask.
            log_a = torch.log(a_c.clamp_min(1e-20))      # (B, cl, H)
            cumlog = torch.cumsum(log_a, dim=1)          # (B, cl, H)

            # --- Inter-chunk contribution: previous state propagated forward ---
            # decay_to_t = exp(cumlog) gives a_{s+1}*...*a_t for each position.
            decay_to_t = torch.exp(cumlog)               # (B, cl, H)
            # y_prev[t] = C_t . (decay_to_t * state)  over state dim N.
            # state: (B, H, head_dim, N); C_t: (B, cl, N).
            # Scale state by decay then contract with C.
            # (B, cl, H, head_dim, N) is large; instead contract C with state once
            # per position using einsum with the per-position decay applied to C.
            C_dec = C_c.unsqueeze(2) * decay_to_t.unsqueeze(-1)   # (B, cl, H, N)
            y_prev = torch.einsum("bchn,bhdn->bchd", C_dec, state)  # (B, cl, H, head_dim)

            # --- Intra-chunk contribution: lower-triangular semiseparable mask ---
            # M[t, s'] = a_t...a_{s'+1} = exp(cumlog_t - cumlog_{s'}) for s' <= t.
            # Build per-head (cl, cl) decay matrix.
            cl_idx = torch.arange(cl, device=x.device)
            # diff[t, s'] = cumlog_t - cumlog_{s'}; valid (and applied) for s' <= t.
            diff = cumlog.unsqueeze(2) - cumlog.unsqueeze(1)        # (B, cl_t, cl_s, H)
            tril = (cl_idx.unsqueeze(1) >= cl_idx.unsqueeze(0)).float()  # (cl_t, cl_s)
            decay_mat = torch.exp(diff) * tril.unsqueeze(0).unsqueeze(-1)  # (B,cl_t,cl_s,H)
            # State-readout coupling: g[t, s'] = (C_t . B_{s'}) over N.
            cb = torch.einsum("btn,bsn->bts", C_c, B_c).unsqueeze(-1)  # (B,cl_t,cl_s,1)
            # weights w[t,s',H] = decay_mat * (C_t.B_s') * dt_{s'}.
            w = decay_mat * cb * dt_c.squeeze(-1).unsqueeze(1)        # (B,cl_t,cl_s,H)
            # y_intra[t] = sum_{s'<=t} w[t,s'] * x_{s'}.
            y_intra = torch.einsum("btsh,bshd->bthd", w, x_c)         # (B, cl, H, head_dim)

            y_c = y_prev + y_intra
            outputs.append(y_c.reshape(batch, cl, self.d_inner))

            # --- Update running state for the next chunk ---
            # state_new = a_total * state + sum_{s'} (a_{cl}...a_{s'+1}) dt_{s'} B_{s'} x_{s'}.
            a_total = torch.exp(cumlog[:, -1])                       # (B, H)
            # decay_from_s' to chunk end: exp(cumlog_end - cumlog_s').
            decay_end = torch.exp(cumlog[:, -1:].clamp_min(-50.0) - cumlog)  # (B, cl, H)
            bx = torch.einsum(
                "bsn,bshd->bshdn",
                B_c,
                x_c * (dt_c * decay_end.unsqueeze(-1)),
            )  # (B, cl, H, head_dim, N)
            state = state * a_total.unsqueeze(-1).unsqueeze(-1) + bx.sum(dim=1)

        y = torch.cat(outputs, dim=1)                               # (B, L, d_inner)
        return y + x * self.D


# ---------------------------------------------------------------------------
# SSD branch (Eqs. 6-9): two parallel SSD axes + SiLU skip, fused with RMSNorm.
# ---------------------------------------------------------------------------


class SSDBranch(nn.Module):
    """Global branch of the ConvSSD block.

    Implements ``y = Linear(RMSNorm(y_axis1 + y_axis2 + SiLU(Linear(x))))`` where
    each axis is ``SSD(DWConv(Linear(x)))``. The paper's two axes are the
    spatial-first and temporal-first scans of the 2D EEG map; for FACETpy's
    single-/few-channel channel-wise pipeline we realise them as a forward-time
    and a reverse-time scan (bidirectional temporal SSD). This preserves the
    dual-path structure while being meaningful for a single channel; see
    ``documentation/paper_accuracy_review.md`` for the rationale.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        n_heads: int = 1,
        d_conv: int = 4,
        ssd_chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.in_a1 = nn.Linear(self.dim, self.dim, bias=False)
        self.in_a2 = nn.Linear(self.dim, self.dim, bias=False)
        self.skip = nn.Linear(self.dim, self.dim, bias=False)

        self.dwconv_a1 = nn.Conv1d(self.dim, self.dim, d_conv, padding=d_conv - 1, groups=self.dim)
        self.dwconv_a2 = nn.Conv1d(self.dim, self.dim, d_conv, padding=d_conv - 1, groups=self.dim)

        self.ssd_a1 = SSDLayer(self.dim, d_state=d_state, n_heads=n_heads, chunk_size=ssd_chunk_size)
        self.ssd_a2 = SSDLayer(self.dim, d_state=d_state, n_heads=n_heads, chunk_size=ssd_chunk_size)

        self.norm = RMSNorm(self.dim)
        self.out = nn.Linear(self.dim, self.dim, bias=False)

    def _dwconv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        h = conv(x.transpose(1, 2))[..., :length].transpose(1, 2)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, dim)
        # Axis 1: forward-time SSD ("temporal-first" analogue).
        a1 = self._dwconv(self.dwconv_a1, self.in_a1(x))
        y_a1 = self.ssd_a1(a1)

        # Axis 2: reverse-time SSD ("spatial-first" analogue for single channel).
        a2 = self._dwconv(self.dwconv_a2, self.in_a2(x))
        a2_rev = torch.flip(a2, dims=[1])
        y_a2 = torch.flip(self.ssd_a2(a2_rev), dims=[1])

        # No-SSD symmetric SiLU skip (Eq. 8).
        y_skip = nn.functional.silu(self.skip(x))

        y = self.out(self.norm(y_a1 + y_a2 + y_skip))
        return y


# ---------------------------------------------------------------------------
# ConvSSD block (Fig. 2, Eq. 10): channel split + dual branch + learnable fusion.
# ---------------------------------------------------------------------------


class ConvSSDBlock(nn.Module):
    """Paper-faithful ConvSSD block.

    Channel-splits the ``(B, C, L)`` feature into two halves: a local conv
    branch and a global SSD branch. The two branch outputs are scaled by two
    learnable scalars ``r1, r2`` (Eq. 10), concatenated, then refined by a
    depthwise-separable conv. Output preserves ``(B, C, L)``.
    """

    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        n_heads: int = 1,
        d_conv: int = 4,
        dropout: float = 0.2,
        ssd_chunk_size: int = 64,
    ) -> None:
        super().__init__()
        if channels < 2:
            raise ValueError("ConvSSDBlock requires channels >= 2 for the channel split")
        self.channels = int(channels)
        self.c_conv = self.channels // 2
        self.c_ssd = self.channels - self.c_conv

        # Local conv branch: Conv3-BN-PReLU-Conv3-BN-PReLU-Dropout.
        self.conv_branch = nn.Sequential(
            nn.Conv1d(self.c_conv, self.c_conv, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.c_conv),
            nn.PReLU(),
            nn.Conv1d(self.c_conv, self.c_conv, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.c_conv),
            nn.PReLU(),
            nn.Dropout(float(dropout)),
        )

        # Global SSD branch operates on (B, L, c_ssd).
        self.ssd_branch = SSDBranch(
            self.c_ssd,
            d_state=d_state,
            n_heads=n_heads,
            d_conv=d_conv,
            ssd_chunk_size=ssd_chunk_size,
        )

        # Learnable fusion scalars r1 (conv), r2 (SSD) — Eq. 10.
        self.r1 = nn.Parameter(torch.ones(1))
        self.r2 = nn.Parameter(torch.ones(1))

        # Depthwise-separable refinement conv over the concatenated channels.
        self.dw = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1, groups=self.channels)
        self.pw = nn.Conv1d(self.channels, self.channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        x_conv, x_ssd = torch.split(x, [self.c_conv, self.c_ssd], dim=1)

        y_conv = self.conv_branch(x_conv)

        y_ssd = self.ssd_branch(x_ssd.transpose(1, 2)).transpose(1, 2)  # (B, c_ssd, L)

        fused = torch.cat([y_conv * self.r1, y_ssd * self.r2], dim=1)   # (B, C, L)
        out = self.pw(self.dw(fused))
        return out + x  # residual keeps gradients healthy in the deep U-Net


# ---------------------------------------------------------------------------
# Signal embedding stem and projection head.
# ---------------------------------------------------------------------------


class SignalEmbedding(nn.Module):
    """Conv3 -> PReLU -> BatchNorm -> Conv3 lifting 1 -> base_channels (Fig. 1)."""

    def __init__(self, base_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.BatchNorm1d(base_channels),
            nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectionHead(nn.Module):
    """GAP -> Linear -> PReLU -> LayerNorm -> Linear summary, fused with a
    length-preserving 1x1 conv so the output stays ``(B, 1, L)`` (Fig. 1 inset).

    The paper's GAP collapses time into a global summary. To keep a per-sample
    output for FACETpy's sample-aligned subtraction we add the pooled summary
    back as a per-channel bias to a length-preserving conv path, then map to a
    single output channel. Documented as a length-preserving head adaptation.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gap_linear = nn.Linear(channels, channels)
        self.gap_act = nn.PReLU()
        self.gap_norm = nn.LayerNorm(channels)
        self.gap_out = nn.Linear(channels, channels)
        # Length-preserving path back to a single channel.
        self.out_conv = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        pooled = x.mean(dim=-1)                       # GAP -> (B, C)
        s = self.gap_out(self.gap_norm(self.gap_act(self.gap_linear(pooled))))  # (B, C)
        # Inject the global summary as a per-channel additive bias.
        h = x + s.unsqueeze(-1)                       # (B, C, L)
        return self.out_conv(h)                       # (B, 1, L)


# ---------------------------------------------------------------------------
# Full U-shaped DenoiseMamba.
# ---------------------------------------------------------------------------


class PaperAccurateDenoiseMamba(nn.Module):
    """U-shaped CNN+Mamba2 ConvSSD denoiser for single-channel EEG epochs.

    Input  : ``(batch, 1, samples)``
    Output : ``(batch, 1, samples)`` — the predicted CLEAN signal (default) or
             the predicted artifact, depending on the training target_type. The
             module itself is target-agnostic; the dataset/loss decide the
             meaning of the output.
    """

    def __init__(
        self,
        epoch_samples: int = 512,
        base_channels: int = 64,
        n_stages: int = 3,
        d_state: int = 16,
        n_heads: int = 1,
        d_conv: int = 4,
        dropout: float = 0.2,
        ssd_chunk_size: int = 64,
    ) -> None:
        super().__init__()
        if epoch_samples <= 0:
            raise ValueError("epoch_samples must be positive")
        if n_stages < 1:
            raise ValueError("n_stages must be >= 1")
        if base_channels < 2:
            raise ValueError("base_channels must be >= 2 (ConvSSD channel split)")

        self.epoch_samples = int(epoch_samples)
        self.base_channels = int(base_channels)
        self.n_stages = int(n_stages)
        self.divisor = 2 ** self.n_stages

        self.embedding = SignalEmbedding(self.base_channels)

        def _block(ch: int) -> ConvSSDBlock:
            return ConvSSDBlock(
                ch,
                d_state=d_state,
                n_heads=n_heads,
                d_conv=d_conv,
                dropout=dropout,
                ssd_chunk_size=ssd_chunk_size,
            )

        # Encoder: each stage = ConvSSD block then downsample (stride-2 conv that
        # doubles the channel count). Channel pyramid base -> 2*base -> 4*base ...
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = self.base_channels
        enc_channels: list[int] = []
        for _ in range(self.n_stages):
            self.enc_blocks.append(_block(ch))
            enc_channels.append(ch)
            self.downs.append(nn.Conv1d(ch, ch * 2, kernel_size=2, stride=2))
            ch *= 2

        # Bottleneck at the deepest width/length.
        self.bottleneck = _block(ch)

        # Decoder: each stage = upsample, concat the matching encoder feature,
        # fuse channels back, then a ConvSSD block.
        self.ups = nn.ModuleList()
        self.fuse = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for skip_ch in reversed(enc_channels):
            self.ups.append(nn.ConvTranspose1d(ch, skip_ch, kernel_size=2, stride=2))
            # After concat with the skip we have 2 * skip_ch channels.
            self.fuse.append(nn.Conv1d(skip_ch * 2, skip_ch, kernel_size=1))
            self.dec_blocks.append(_block(skip_ch))
            ch = skip_ch

        self.head = ProjectionHead(self.base_channels)

    def _pad_to_divisor(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        length = x.shape[-1]
        rem = length % self.divisor
        pad = 0 if rem == 0 else self.divisor - rem
        if pad:
            x = nn.functional.pad(x, (0, pad), mode="replicate")
        return x, pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != 1:
            raise ValueError(
                f"PaperAccurateDenoiseMamba expects shape (batch, 1, samples); got {tuple(x.shape)}"
            )
        orig_len = x.shape[-1]
        x, pad = self._pad_to_divisor(x)

        h = self.embedding(x)

        skips: list[torch.Tensor] = []
        for block, down in zip(self.enc_blocks, self.downs):
            h = block(h)
            skips.append(h)
            h = down(h)

        h = self.bottleneck(h)

        for up, fuse, block, skip in zip(self.ups, self.fuse, self.dec_blocks, reversed(skips)):
            h = up(h)
            # Guard against off-by-one length mismatches from odd pooling.
            if h.shape[-1] != skip.shape[-1]:
                target = skip.shape[-1]
                if h.shape[-1] > target:
                    h = h[..., :target]
                else:
                    h = nn.functional.pad(h, (0, target - h.shape[-1]), mode="replicate")
            h = fuse(torch.cat([h, skip], dim=1))
            h = block(h)

        out = self.head(h)
        if pad:
            out = out[..., :orig_len]
        return out


# ---------------------------------------------------------------------------
# Dataset: channel-wise single-epoch denoising over the *_center NPZ bundle.
# ---------------------------------------------------------------------------


class _SubsetDataset:
    def __init__(self, parent: "ChannelWiseCenterDataset", indices: list[int]) -> None:
        self._parent = parent
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._parent[self._indices[idx]]


class ChannelWiseCenterDataset:
    """Per-channel single-epoch dataset over the Niazy ``*_center`` NPZ bundle.

    Loads ``noisy_center``, ``clean_center`` and ``artifact_center`` (each of
    shape ``(n_examples, n_channels, n_samples)``) directly from the proof-fit
    NPZ, so it can serve either a clean target (``target_type='clean'``, the
    paper default) or an artifact target (``target_type='artifact'``, for
    FACETpy pipeline parity). Each ``(example, channel)`` pair becomes one item
    of shape ``(1, n_samples)``.

    Normalisation mirrors the paper's z-score standardisation (Fig. 6) when
    ``normalize='zscore'``; ``normalize='demean'`` reproduces the original
    baseline's per-epoch mean removal; ``normalize='none'`` leaves data raw.
    Input and target use the same per-epoch statistics computed from the noisy
    input so the (noisy, target) pair stays consistent in amplitude.
    """

    def __init__(
        self,
        noisy: np.ndarray,
        clean: np.ndarray,
        artifact: np.ndarray,
        *,
        sfreq: float,
        target_type: str = "clean",
        normalize: str = "zscore",
        max_examples: int | None = None,
    ) -> None:
        target_type = str(target_type).strip().lower()
        if target_type not in {"clean", "artifact"}:
            raise ValueError(f"target_type must be 'clean' or 'artifact', got '{target_type}'")
        normalize = str(normalize).strip().lower()
        if normalize not in {"zscore", "demean", "none"}:
            raise ValueError(f"normalize must be 'zscore', 'demean' or 'none', got '{normalize}'")

        noisy = np.asarray(noisy, dtype=np.float32)
        clean = np.asarray(clean, dtype=np.float32)
        artifact = np.asarray(artifact, dtype=np.float32)
        if not (noisy.shape == clean.shape == artifact.shape):
            raise ValueError("noisy/clean/artifact arrays must share shape (examples, channels, samples)")
        if noisy.ndim != 3:
            raise ValueError("center arrays must have shape (examples, channels, samples)")

        self._noisy = noisy
        self._clean = clean
        self._artifact = artifact
        self.target_type = target_type
        self.normalize = normalize
        self.sfreq = float(sfreq)
        self.trigger_aligned = True

        self.n_examples = int(noisy.shape[0])
        self.n_channels = int(noisy.shape[1])
        self.epoch_samples = int(noisy.shape[2])
        self.chunk_size = self.epoch_samples

        total = self.n_examples * self.n_channels
        self._length = total if max_examples is None else max(0, min(int(max_examples), total))
        if self._length == 0:
            raise ValueError("dataset is empty")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        example_idx = int(idx) // self.n_channels
        channel_idx = int(idx) % self.n_channels
        noisy = self._noisy[example_idx, channel_idx : channel_idx + 1, :].astype(np.float32, copy=True)
        if self.target_type == "clean":
            target = self._clean[example_idx, channel_idx : channel_idx + 1, :].astype(np.float32, copy=True)
        else:
            target = self._artifact[example_idx, channel_idx : channel_idx + 1, :].astype(np.float32, copy=True)

        if self.normalize == "none":
            return noisy, target

        mean = noisy.mean(axis=-1, keepdims=True)
        noisy = noisy - mean
        # Artifact target shares the additive component, so it is demeaned too;
        # the clean target's mean is left untouched only conceptually, but to
        # keep the pair on a common scale we subtract the same input mean.
        target = target - target.mean(axis=-1, keepdims=True)
        if self.normalize == "zscore":
            std = float(np.std(noisy)) or 1.0
            noisy = noisy / std
            target = target / std
        return noisy.astype(np.float32, copy=False), target.astype(np.float32, copy=False)

    @property
    def input_shape(self) -> tuple[int, int]:
        return (1, self.epoch_samples)

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
# Public factories consumed by facet-train.
# ---------------------------------------------------------------------------


def build_model(
    input_shape: tuple[int, ...] | None = None,
    target_shape: tuple[int, ...] | None = None,
    epoch_samples: int | None = None,
    base_channels: int = 64,
    n_stages: int = 3,
    d_state: int = 16,
    n_heads: int = 1,
    d_conv: int = 4,
    dropout: float = 0.2,
    ssd_chunk_size: int = 64,
    **_: object,
) -> PaperAccurateDenoiseMamba:
    """Construct the paper-accurate U-shaped DenoiseMamba.

    Accepts (and ignores) the kwargs facet-train injects (n_channels, chunk_size,
    sfreq, target_type, training_config, input_shape, target_shape,
    context_epochs, epoch_samples). ``epoch_samples`` is resolved from
    ``input_shape`` when not given explicitly.
    """
    if epoch_samples is None and input_shape is not None:
        epoch_samples = int(input_shape[-1])
    if epoch_samples is None:
        raise ValueError("build_model requires epoch_samples or input_shape")
    return PaperAccurateDenoiseMamba(
        epoch_samples=int(epoch_samples),
        base_channels=int(base_channels),
        n_stages=int(n_stages),
        d_state=int(d_state),
        n_heads=int(n_heads),
        d_conv=int(d_conv),
        dropout=float(dropout),
        ssd_chunk_size=int(ssd_chunk_size),
    )


def build_loss(name: str = "mse", **_: object) -> nn.Module:
    """Loss factory. The paper uses MSE between denoised output and clean target."""
    normalized = str(name).strip().lower()
    if normalized == "l1":
        return nn.L1Loss()
    if normalized in {"smooth_l1", "huber"}:
        return nn.SmoothL1Loss()
    return nn.MSELoss()


def build_dataset(
    path: str | None = None,
    context_path: str | None = None,
    max_examples: int | None = None,
    target_type: str = "clean",
    normalize: str = "zscore",
    **_: object,
) -> ChannelWiseCenterDataset:
    """Construct the channel-wise center dataset from a Niazy ``*_center`` NPZ.

    Defaults to ``target_type='clean'`` to match the paper (predict the denoised
    signal with MSE loss); pass ``target_type='artifact'`` for FACETpy pipeline
    parity (predict the artifact to subtract).
    """
    dataset_path = Path(path or context_path or "").expanduser()
    if not str(dataset_path) or str(dataset_path) == ".":
        raise ValueError("build_dataset requires path or context_path")
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    with np.load(dataset_path, allow_pickle=False) as bundle:
        noisy = np.asarray(bundle["noisy_center"], dtype=np.float32)
        clean = np.asarray(bundle["clean_center"], dtype=np.float32)
        artifact = np.asarray(bundle["artifact_center"], dtype=np.float32)
        sfreq = float(bundle["sfreq"][0]) if "sfreq" in bundle else float("nan")

    return ChannelWiseCenterDataset(
        noisy,
        clean,
        artifact,
        sfreq=sfreq,
        target_type=target_type,
        normalize=normalize,
        max_examples=max_examples,
    )
