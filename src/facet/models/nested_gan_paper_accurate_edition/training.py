"""Paper-accurate training factories for the Nested-GAN model (edition 2).

This edition is a more *Restormer-faithful* re-implementation of the inner,
spectral generator branch of the Nested-GAN. The primary Nested-GAN paper
(Biomed. Phys. Eng. Express 2025, DOI 10.1088/2057-1976/ae1a8c, PMID
41183389) is paywalled and discloses only summary metrics in its abstract; it
does NOT disclose architecture depth/width, STFT parameters, the optimizer
schedule, or the exact discriminator structure. The documented backbone of the
generator is therefore Restormer (Zamir et al., CVPR 2022, "Restormer:
Efficient Transformer for High-Resolution Image Restoration"), and that paper
*is* available and is the basis for this paper-accuracy pass.

What changed vs. ``facet.models.nested_gan`` (see README.md and
``documentation/paper_accuracy_review.md`` for the full discrepancy table):

* The inner spectral branch is now a genuine **hierarchical** Restormer-style
  encoder-decoder (``HierarchicalSpectrogramRestormer``) with per-level channel
  doubling, per-level head counts, per-level block depth, pixel-unshuffle /
  pixel-shuffle down/up sampling, skip-concat + 1x1 channel-halving, an optional
  refinement stage, and a **global residual** (output = input_spec + R) -- all
  faithful to Restormer's actual multi-scale design. The original used a flat
  single-resolution stack of identical blocks with no residual.
* GDFN channel-expansion default is now ``gamma = 2.66`` (Restormer's value).
* An optional **bias-free LayerNorm** flag matches Restormer's denoising config.
* The inner branch can optionally also see the center +/-1 neighbour epochs as
  extra STFT input channels (an EEG-fMRI-motivated context improvement, *not* a
  paper claim; default off).

What deliberately stayed the same (documented, not a bug):

* Generator-only recipe with a multi-resolution STFT magnitude loss as a
  *surrogate* for the paper's multi-resolution discriminators. The facet-train
  CLI takes a single ``(pred, target) -> scalar`` loss and one optimizer; a true
  alternating 4-discriminator nested GAN needs a custom training-loop wrapper
  that is out of scope and not CPU-cheap, and -- critically -- the GAN/nesting
  structure cannot be verified against the paywalled paper.
* MDTA channel cross-covariance attention with a learnable per-head temperature
  (a deliberate, faithful Restormer departure from Vaswani 2017's fixed
  ``1/sqrt(d_k)`` scaling) is preserved exactly.
* The outer time-domain U-Net refiner over the multi-epoch context with
  center-slot residual injection (the FACETpy-appropriate boundary-continuity
  fix) is preserved.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from facet.training.dataset import NPZContextArtifactDataset


# ---------------------------------------------------------------------------
# Restormer primitives (faithful to Zamir et al. 2022)
# ---------------------------------------------------------------------------


class _LayerNorm2d(torch.nn.Module):
    """Channel-wise layer norm for ``(B, C, H, W)`` tensors.

    Restormer's denoising configuration uses a *bias-free* LayerNorm (only a
    learnable scale, no shift), which empirically improves generalisation across
    noise levels. ``bias_free=True`` selects that variant; the with-bias variant
    matches the original FACETpy edition and is the default for backward
    compatibility.
    """

    def __init__(self, channels: int, *, bias_free: bool = False) -> None:
        super().__init__()
        self.bias_free = bool(bias_free)
        self.weight = torch.nn.Parameter(torch.ones(channels))
        self.bias = None if self.bias_free else torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias_free:
            # Mean-free (variance-only) normalisation, per Restormer denoising.
            var = x.var(dim=1, keepdim=True, unbiased=False)
            x = x / torch.sqrt(var + 1e-6)
            return x * self.weight.view(1, -1, 1, 1)
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + 1e-6)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class _MDTA(torch.nn.Module):
    """Multi-DConv head transposed attention from Restormer (Zamir 2022).

    Attention is computed across the *channel* dimension (cross-covariance)
    instead of spatial positions, which keeps complexity linear in the spatial
    size. The fixed ``1/sqrt(d_k)`` scaling of Vaswani et al. (2017) is replaced
    by a learnable per-head temperature ``alpha`` -- this is intentional and
    faithful to Restormer, not a bug.
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
    """Gated DConv feed-forward network from Restormer (Zamir 2022).

    The channel-expansion ratio ``gamma`` defaults to 2.66, Restormer's value
    (chosen to roughly match the parameter/compute budget of a regular FFN).
    """

    def __init__(self, channels: int, expansion: float = 2.66) -> None:
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
    """One Restormer block: MDTA -> GDFN with layer-norm residuals."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        expansion: float = 2.66,
        *,
        bias_free_norm: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = _LayerNorm2d(channels, bias_free=bias_free_norm)
        self.attn = _MDTA(channels, num_heads=num_heads)
        self.norm2 = _LayerNorm2d(channels, bias_free=bias_free_norm)
        self.ffn = _GDFN(channels, expansion=expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# --- Down/Up sampling (pixel-unshuffle / pixel-shuffle, as in Restormer) -----


class _Downsample(torch.nn.Module):
    """Halve H and W, double channels (1x1 conv then pixel-unshuffle).

    Restormer downsamples with a 3x3 conv that *halves* channels followed by a
    2x pixel-unshuffle (which quadruples channels), netting a 2x channel
    increase. We use a 1x1 conv for CPU cheapness; the channel arithmetic is
    identical.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # conv halves channels -> pixel_unshuffle(2) multiplies by 4 -> net 2x.
        self.body = torch.nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False)
        self.pix = torch.nn.PixelUnshuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pix(self.body(x))


class _Upsample(torch.nn.Module):
    """Double H and W, halve channels (1x1 conv then pixel-shuffle).

    Mirror of :class:`_Downsample`: a conv doubles channels, then a 2x
    pixel-shuffle quarters them, netting a 2x channel decrease.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = torch.nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False)
        self.pix = torch.nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pix(self.body(x))


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    """Reflect-pad H and W up to a multiple of ``multiple`` for clean down/up.

    Returns the padded tensor plus the original (H, W) so the caller can crop
    back after the decoder.
    """
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, h, w


# ---------------------------------------------------------------------------
# Inner branch: HIERARCHICAL complex-STFT Restormer encoder-decoder
# ---------------------------------------------------------------------------


class HierarchicalSpectrogramRestormer(torch.nn.Module):
    """Inner branch: a small hierarchical Restormer over the center-epoch STFT.

    This is the headline paper-accuracy fix vs. the original flat block stack.
    The branch builds a genuine multi-scale encoder-decoder:

    * input projection ``in_channels -> base channels`` (3x3 conv),
    * ``levels`` encoder stages, each = a stack of Restormer blocks then a
      pixel-unshuffle downsample that doubles channels,
    * a bottleneck stack of Restormer blocks,
    * matching decoder stages, each = pixel-shuffle upsample, skip-concat with
      the encoder feature, a 1x1 conv halving channels back, then Restormer
      blocks,
    * an optional refinement stage at full resolution,
    * a final 3x3 conv producing a 2-channel residual ``R`` that is added to the
      *projected input spectrogram* (Restormer global residual learning).

    Per level the channel width doubles (base, 2x, 4x, ...) and the head count
    follows ``head_counts``. The whole thing is inverse-STFT'd to the time
    domain to yield the inner artifact estimate.
    """

    def __init__(
        self,
        *,
        n_fft: int = 64,
        hop_length: int = 16,
        win_length: int | None = None,
        in_channels: int = 2,
        base_channels: int = 48,
        levels: int = 3,
        block_counts: tuple[int, ...] | None = None,
        head_counts: tuple[int, ...] | None = None,
        refinement_blocks: int = 2,
        expansion: float = 2.66,
        bias_free_norm: bool = False,
        target_samples: int = 512,
    ) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length or n_fft)
        self.target_samples = int(target_samples)
        self.in_channels = int(in_channels)
        levels = int(levels)
        if levels < 1:
            raise ValueError("levels must be >= 1")
        self.levels = levels
        self.register_buffer(
            "stft_window",
            torch.hann_window(self.win_length, periodic=True),
            persistent=False,
        )

        # Per-level (encoder + bottleneck) defaults, scaled down from Restormer's
        # [4, 6, 6, 8] so a tiny version stays CPU-cheap.
        if block_counts is None:
            full = [2, 3, 3, 4]
            block_counts = tuple(full[: levels + 1])
        block_counts = tuple(int(b) for b in block_counts)
        if len(block_counts) != levels + 1:
            raise ValueError(f"block_counts needs {levels + 1} entries (levels + bottleneck)")

        if head_counts is None:
            full_heads = [1, 2, 4, 8]
            head_counts = tuple(full_heads[: levels + 1])
        head_counts = tuple(int(h) for h in head_counts)
        if len(head_counts) != levels + 1:
            raise ValueError(f"head_counts needs {levels + 1} entries (levels + bottleneck)")

        # Channel widths per level: base, 2x, 4x, ...
        widths = [base_channels * (2 ** i) for i in range(levels + 1)]

        # Input projection (degraded spectrogram -> features).
        self.input_proj = torch.nn.Conv2d(self.in_channels, base_channels, kernel_size=3, padding=1, bias=False)

        # Encoder: blocks then downsample.
        self.encoders = torch.nn.ModuleList()
        self.downs = torch.nn.ModuleList()
        for lvl in range(levels):
            self.encoders.append(
                torch.nn.Sequential(
                    *[
                        _RestormerBlock(
                            widths[lvl], num_heads=head_counts[lvl], expansion=expansion, bias_free_norm=bias_free_norm
                        )
                        for _ in range(block_counts[lvl])
                    ]
                )
            )
            self.downs.append(_Downsample(widths[lvl]))

        # Bottleneck.
        self.bottleneck = torch.nn.Sequential(
            *[
                _RestormerBlock(
                    widths[levels], num_heads=head_counts[levels], expansion=expansion, bias_free_norm=bias_free_norm
                )
                for _ in range(block_counts[levels])
            ]
        )

        # Decoder: upsample, skip-concat, 1x1 reduce, blocks.
        self.ups = torch.nn.ModuleList()
        self.reduces = torch.nn.ModuleList()
        self.decoders = torch.nn.ModuleList()
        for lvl in reversed(range(levels)):
            self.ups.append(_Upsample(widths[lvl + 1]))
            # After upsample we have widths[lvl] channels; concat with skip
            # (also widths[lvl]) -> 2*widths[lvl]; 1x1 conv halves back. Skip the
            # reduction at the *top* level only if the original Restormer does
            # (it keeps a reduce at every level except the topmost). We mirror
            # that: top level (lvl == 0) has no reduce, concat stays doubled.
            if lvl == 0:
                self.reduces.append(torch.nn.Identity())
                dec_in = widths[lvl] * 2
            else:
                self.reduces.append(torch.nn.Conv2d(widths[lvl] * 2, widths[lvl], kernel_size=1, bias=False))
                dec_in = widths[lvl]
            self.decoders.append(
                torch.nn.Sequential(
                    *[
                        _RestormerBlock(
                            dec_in, num_heads=head_counts[lvl], expansion=expansion, bias_free_norm=bias_free_norm
                        )
                        for _ in range(block_counts[lvl])
                    ]
                )
            )

        # Decoder output width (top level may be doubled, per the comment above).
        dec_out_width = widths[0] * 2 if levels >= 1 else widths[0]

        # Optional refinement stage at full resolution.
        refinement_blocks = int(refinement_blocks)
        if refinement_blocks > 0:
            self.refinement = torch.nn.Sequential(
                *[
                    _RestormerBlock(
                        dec_out_width, num_heads=head_counts[0], expansion=expansion, bias_free_norm=bias_free_norm
                    )
                    for _ in range(refinement_blocks)
                ]
            )
        else:
            self.refinement = torch.nn.Identity()

        # Output residual projection: features -> 2-channel residual spectrogram.
        self.output_proj = torch.nn.Conv2d(dec_out_width, 2, kernel_size=3, padding=1, bias=False)

    # --- STFT helpers --------------------------------------------------------

    def _stft(self, signal: torch.Tensor) -> torch.Tensor:
        """STFT a ``(B, T)`` signal into a ``(B, 2, F, frames)`` real/imag image."""
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

    def forward(self, inner_input: torch.Tensor) -> torch.Tensor:
        """Run the hierarchical Restormer.

        ``inner_input`` is ``(B, n_in, T)`` where ``n_in`` is the number of
        time-domain channels fed to the spectral stage (the center epoch, and
        optionally its neighbours). Each is STFT'd and stacked, giving an image
        with ``2 * n_in`` channels. The output is the time-domain inner artifact
        for the center epoch only, shape ``(B, T)``.
        """
        if inner_input.dim() == 2:
            inner_input = inner_input.unsqueeze(1)
        b, n_in, _ = inner_input.shape
        specs = [self._stft(inner_input[:, c, :]) for c in range(n_in)]
        spec_image = torch.cat(specs, dim=1)  # (B, 2*n_in, F, frames)

        # Project input and keep a copy of the projected spectrogram as the
        # global-residual baseline. We predict R such that the artifact spectrum
        # ~= projected_input + R, in the 2-channel real/imag space. The center
        # epoch's own STFT is the natural baseline for an "artifact" target since
        # noisy = clean + artifact and the artifact dominates the magnitude.
        center_spec = specs[0]  # (B, 2, F, frames) -- the center epoch
        x = self.input_proj(spec_image)

        # Pad spatial dims to a multiple of 2**levels for clean pixel-(un)shuffle.
        x, orig_h, orig_w = _pad_to_multiple(x, 2 ** self.levels)

        skips: list[torch.Tensor] = []
        for lvl in range(self.levels):
            x = self.encoders[lvl](x)
            skips.append(x)
            x = self.downs[lvl](x)

        x = self.bottleneck(x)

        for i, lvl in enumerate(reversed(range(self.levels))):
            x = self.ups[i](x)
            skip = skips[lvl]
            x = torch.cat([x, skip], dim=1)
            x = self.reduces[i](x)
            x = self.decoders[i](x)

        x = self.refinement(x)
        residual_full = self.output_proj(x)

        # Crop back to the un-padded spectrogram size.
        residual = residual_full[:, :, :orig_h, :orig_w]

        artifact_spec = center_spec + residual  # Restormer global residual.
        return self._istft(artifact_spec)


# ---------------------------------------------------------------------------
# Outer branch: 1D U-Net refiner over the multi-epoch context (unchanged)
# ---------------------------------------------------------------------------


def _conv_block(in_channels: int, out_channels: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
        torch.nn.GELU(),
        torch.nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
        torch.nn.GELU(),
    )


class OuterTimeRefiner(torch.nn.Module):
    """Outer branch: 1D residual U-Net over the multi-epoch context.

    Input is the context stack with the center epoch replaced by the inner
    branch's time-domain output. Output is a refined center-epoch artifact.
    This is a FACETpy-appropriate adaptation (not a Restormer feature): it
    injects neighbour-epoch context to fix trigger-boundary phase
    discontinuities.
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
# Nested-GAN generator (paper-accurate edition: hierarchical inner branch)
# ---------------------------------------------------------------------------


class NestedGANGenerator(torch.nn.Module):
    """Hierarchical inner spectral Restormer cascaded into an outer time refiner.

    Input shape:  ``(batch, context_epochs, 1, target_samples)`` (channel-wise).
    Output shape: ``(batch, 1, target_samples)`` predicted center-epoch artifact.

    When ``inner_neighbor_epochs > 0`` the spectral stage also receives that many
    neighbour epochs on each side of the center (as extra STFT input channels);
    this is an EEG-fMRI-motivated improvement, gated off by default.
    """

    def __init__(
        self,
        *,
        context_epochs: int = 7,
        target_samples: int = 512,
        inner_channels: int = 48,
        inner_levels: int = 3,
        inner_block_counts: tuple[int, ...] | None = None,
        inner_head_counts: tuple[int, ...] | None = None,
        inner_refinement_blocks: int = 2,
        inner_expansion: float = 2.66,
        inner_bias_free_norm: bool = False,
        inner_neighbor_epochs: int = 0,
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

        radius = self.context_epochs // 2
        self.inner_neighbor_epochs = max(0, min(int(inner_neighbor_epochs), radius))
        # Center epoch first, then +/-1, +/-2, ... up to inner_neighbor_epochs.
        n_inner_time_channels = 1 + 2 * self.inner_neighbor_epochs

        self.inner = HierarchicalSpectrogramRestormer(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            in_channels=2 * n_inner_time_channels,
            base_channels=inner_channels,
            levels=inner_levels,
            block_counts=inner_block_counts,
            head_counts=inner_head_counts,
            refinement_blocks=inner_refinement_blocks,
            expansion=inner_expansion,
            bias_free_norm=inner_bias_free_norm,
            target_samples=self.target_samples,
        )
        self.outer = OuterTimeRefiner(
            context_epochs=self.context_epochs,
            base_channels=outer_base_channels,
            target_samples=self.target_samples,
        )

    def _gather_inner_input(self, context_2d: torch.Tensor) -> torch.Tensor:
        """Build the inner branch input: center epoch + symmetric neighbours."""
        center = self.center_index
        idxs = [center]
        for offset in range(1, self.inner_neighbor_epochs + 1):
            idxs.append(center - offset)
            idxs.append(center + offset)
        return torch.stack([context_2d[:, i, :] for i in idxs], dim=1)

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

        inner_input = self._gather_inner_input(context_2d)
        inner_artifact = self.inner(inner_input)

        refined_context = context_2d.clone()
        refined_context[:, self.center_index, :] = context_2d[:, self.center_index, :] - inner_artifact

        residual = self.outer(refined_context)
        return inner_artifact.unsqueeze(1) + residual


# ---------------------------------------------------------------------------
# Loss: L1 in time plus multi-resolution log-magnitude STFT loss (unchanged)
# ---------------------------------------------------------------------------


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Sum of L1 errors on log-magnitude STFT at several window sizes.

    This is a *deterministic surrogate* for the paper's multi-resolution
    discriminators (HiFi-GAN lineage). It is differentiable, single-loss-CLI
    compatible, and CPU-cheap.
    """

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

    def _log_mag(self, signal: torch.Tensor, n_fft: int) -> torch.Tensor:
        hop = max(1, int(n_fft * self.hop_fraction))
        window = torch.hann_window(n_fft, periodic=True, dtype=signal.dtype, device=signal.device)
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
# Per-channel context dataset (unchanged from the original edition)
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
    inner_levels: int = 3,
    inner_block_counts: list[int] | tuple[int, ...] | None = None,
    inner_head_counts: list[int] | tuple[int, ...] | None = None,
    inner_refinement_blocks: int = 2,
    inner_expansion: float = 2.66,
    inner_bias_free_norm: bool = False,
    inner_neighbor_epochs: int = 0,
    outer_base_channels: int = 32,
    n_fft: int = 64,
    hop_length: int = 16,
    win_length: int | None = None,
    **_: object,
) -> NestedGANGenerator:
    """Construct the paper-accurate Nested-GAN generator.

    Accepts the facet-train injected kwargs (``input_shape``, ``epoch_samples``,
    ``context_epochs``, plus others swallowed by ``**_``). Explicit YAML
    ``model.kwargs`` override the injected defaults.
    """
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

    block_counts = None if inner_block_counts is None else tuple(int(b) for b in inner_block_counts)
    head_counts = None if inner_head_counts is None else tuple(int(h) for h in inner_head_counts)

    return NestedGANGenerator(
        context_epochs=int(ctx),
        target_samples=target_samples,
        inner_channels=int(inner_channels),
        inner_levels=int(inner_levels),
        inner_block_counts=block_counts,
        inner_head_counts=head_counts,
        inner_refinement_blocks=int(inner_refinement_blocks),
        inner_expansion=float(inner_expansion),
        inner_bias_free_norm=bool(inner_bias_free_norm),
        inner_neighbor_epochs=int(inner_neighbor_epochs),
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
