"""DHCT-GAN, strict edition: the complete paper, including the adversarial half.

Source
------
Cai et al., *DHCT-GAN: Improving EEG Signal Quality with a Dual-Branch Hybrid
CNN-Transformer Network*, MDPI Sensors 25(1):231, 2025 —
`doi:10.3390/s25010231 <https://doi.org/10.3390/s25010231>`_.
Equation and section numbers below refer to that paper.

What "strict" adds over ``dhct_gan_paper_accurate_edition``
-----------------------------------------------------------
The existing paper-accurate edition already contains three discriminators, LSGAN
and feature matching. What it could not do was *train* them properly, because the
``facet-train`` contract handed the loss exactly one generator tensor. Two of the
generator's three paper outputs therefore received no gradient at all, and the
discriminators lived inside the loss module where nothing checkpointed them. This
edition fixes both, and three further architecture deviations:

1. **Three supervised outputs (§2.2.2, Eqs. 6-9).** The generator returns
   ``{"clean", "noise", "fused"}`` and is trained through
   :class:`~facet.training.adversarial.AdversarialModelWrapper`, so
   ``L_total = Loss1 + Loss2 + Loss3`` is the objective the paper writes down
   rather than one third of it.
2. **Parallel CNN ‖ LGTB (§2.2.2).** The paper states the encoding blocks
   "include parallel CNN and Local-Global Transformer Block (LGTB)". The v2
   paper-accurate edition runs them sequentially (``CNN -> LGTB``); the audit
   records this as an unexplained deviation. Here both paths see the block input
   and are combined by a fusion layer.
3. **Paper gating (Eqs. 4-5).** ``Y_mask = f_gate(X_raw)`` with two fully
   connected layers and tanh, then
   ``Y_pre = Y_mask1 * Y1 + Y_mask2 * (X_raw - Y2)``.
4. **Paper discriminator (Fig. 2a).** M = 8 convolutions with output channels
   64, 64, 128, 128, 256, 256, 512, 512, kernel 3, stride 2, padding 1, each
   followed by batch norm and an activation, with a feature tap for Eq. 11.
5. **Paper optimizers (§2.2.4).** Adam, generator betas (0.5, 0.9), discriminator
   betas (0.9, 0.999) — different per network, which the single-optimizer wrapper
   could not express either.

Deviations that remain, and why
-------------------------------
*These are deliberate and load-bearing; none of them is silent.* The full
argument, with the numbers, is in ``documentation/paper_accuracy_review.md``.

* **Two configurations, and only one of them is the paper.**

  DHCT-GAN is single-channel: §2.1 segments everything into one-dimensional 2 s
  windows (1024 samples at 512 Hz), and the paper's own limitations section says
  the work was done "without considering correction amongst channels".

  ``training_weg_a_paper.yaml`` — ``max_channels=1``, seven epochs. The epochs are
  concatenated in time into one 3584-sample 1-D waveform, which is the format the
  paper's branch consumes; the **only** input deviation is the window length. No
  bridge, no channel mixer, no extra parameters.

  ``training_weg_a_extended.yaml`` — the target electrode plus its geodesic
  neighbours, with the cross-electrode attention bridge active. This is *our*
  addition, aimed at the limitation the paper names, and must never be reported
  as paper replication.

  At ``context_epochs=1, n_channels=1`` the model collapses **exactly** onto the
  paper's forward pass; the equivalence is asserted in
  ``tests/models/test_dhct_gan_strict_edition.py`` so the claim stays true. That
  configuration is not trained here, because a single epoch of a single channel
  contains nothing to compare the centre against — the artifact's epoch-to-epoch
  periodicity is the only thing that distinguishes it from a genuine transient.
* **Decoder head.** The paper's decoding module ends in a fully connected layer
  mapping the flattened bottleneck to the signal length. Its size is quadratic in
  the segment length: at the paper's own 1024-sample segment it is
  ``256 x 16 x 1024 = 4.2 M`` parameters per branch, but at our
  ``7 x 512 = 3584``-sample context it is ``256 x 56 x 3584 = 51 M`` per branch,
  103 M for the pair — more than half the model spent on one linear layer.
  ``decoder_head="fc"`` reproduces the paper and ``"auto"`` keeps it whenever the
  head stays under ``max_fc_head_params``; beyond that a transposed-convolution
  decoder is substituted. Which one was used is recorded in ``describe()`` and
  must be stated with any result.
* **Target domain.** The paper removes EOG/EMG from resting EEG; here the noise
  branch predicts the fMRI gradient artifact. The decomposition
  ``noisy = clean + artifact`` is the same one Eq. 5 assumes, so branch 2 maps
  onto the artifact without reinterpretation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from facet.training.adversarial import (
    AdversarialModelWrapper,
    AdversarialObjective,
    feature_matching_loss,
    lsgan_discriminator_loss,
    lsgan_generator_loss,
)
from facet.training.dataset import NPZSpatioTemporalDataset

DEFAULT_DATASET = "./output/weg_a_farm_v7_k6_512/weg_a_spatiotemporal_dataset.npz"

#: Paper §2.2.2 — encoder width doubles across five CNN-LGTB blocks.
PAPER_ENCODER_DIMS: tuple[int, ...] = (64, 128, 256, 512, 1024)
#: Paper Fig. 2a — M = 8 discriminator convolutions.
PAPER_DISCRIMINATOR_DIMS: tuple[int, ...] = (64, 64, 128, 128, 256, 256, 512, 512)
#: Paper §2.2.2 — the local self-attention module splits the sequence into 8 blocks.
PAPER_LSA_BLOCKS = 8
#: Paper §2.2.2 — the preprocessing module lifts the raw signal to 32 dimensions.
PAPER_STEM_DIM = 32


# ---------------------------------------------------------------------------
# Encoder building blocks (paper §2.2.2)
# ---------------------------------------------------------------------------


class PreprocessingModule(nn.Module):
    """Two 1-D convolutions plus average pooling, lifting the signal to 32 dims.

    Paper §2.2.2 item 1. The average pooling is the only length reduction the
    paper names before the encoder, so it is kept at stride 2 rather than folded
    into the encoder.
    """

    def __init__(self, in_channels: int = 1, out_dim: int = PAPER_STEM_DIM) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_channels, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.body(x))


class CNNBlock(nn.Module):
    """Two kernel-3 convolutions, batch norm and LReLU (paper §2.2.2)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class LocalSelfAttention(nn.Module):
    """Attention inside each of ``n_blocks`` contiguous partitions (paper §2.2.2).

    "The local attention module divides the input data into multiple blocks (set
    to 8 in this study), calculates local attention for each part, and then
    concatenates them." Splitting rather than sliding is what keeps the cost
    linear in sequence length, which matters here: the FACETpy context is 3.5x
    the paper's segment.
    """

    def __init__(self, dim: int, n_heads: int = 4, n_blocks: int = PAPER_LSA_BLOCKS) -> None:
        super().__init__()
        self.n_blocks = int(n_blocks)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, _fit_heads(dim, n_heads), batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        b, t, d = x.shape
        blocks = max(1, min(self.n_blocks, t))
        pad = (-t) % blocks
        h = torch.nn.functional.pad(x.transpose(1, 2), (0, pad), mode="replicate").transpose(1, 2)
        chunk = h.shape[1] // blocks
        h = h.reshape(b * blocks, chunk, d)
        h_norm = self.norm(h)
        attended, _ = self.attn(h_norm, h_norm, h_norm, need_weights=False)
        h = (h + attended).reshape(b, blocks * chunk, d)
        return h[:, :t]


class GlobalSelfAttention(nn.Module):
    """Attention over the whole sequence (paper §2.2.2)."""

    def __init__(self, dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, _fit_heads(dim, n_heads), batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        attended, _ = self.attn(h, h, h, need_weights=False)
        return x + attended


def _fit_heads(dim: int, requested: int) -> int:
    heads = max(1, min(int(requested), dim))
    while dim % heads != 0:
        heads -= 1
    return heads


class LocalGlobalTransformerBlock(nn.Module):
    """LSA -> FFN -> GSA -> FFN -> conv -> BN -> LReLU (paper §2.2.2).

    The order is the paper's: "The output is then passed through a feedforward
    network to the global attention module, where global attention features are
    computed. Subsequently, the output undergoes further processing through a
    feedforward network, a convolutional layer, a batch normalization layer, and
    an LReLU layer."
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, lsa_blocks: int = PAPER_LSA_BLOCKS) -> None:
        super().__init__()
        self.project = nn.Conv1d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()
        self.lsa = LocalSelfAttention(out_dim, n_heads, lsa_blocks)
        self.ffn_local = _FeedForward(out_dim)
        self.gsa = GlobalSelfAttention(out_dim, n_heads)
        self.ffn_global = _FeedForward(out_dim)
        self.tail = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.project(x).transpose(1, 2)  # (B, T, D)
        h = self.ffn_local(self.lsa(h))
        h = self.ffn_global(self.gsa(h))
        return self.tail(h.transpose(1, 2))


class CNNLGTBBlock(nn.Module):
    """One encoding block: **parallel** CNN and LGTB, then fusion (paper §2.2.2).

    The parallel topology is the point. Running the LGTB on the CNN's output — as
    ``dhct_gan_v2_paper_accurate_edition`` does — makes the transformer see
    convolutionally smoothed features rather than the block input, which is a
    different model, not a different implementation of the same one.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int = 4,
        lsa_blocks: int = PAPER_LSA_BLOCKS,
        downsample: bool = True,
    ) -> None:
        super().__init__()
        self.cnn = CNNBlock(in_dim, out_dim)
        self.lgtb = LocalGlobalTransformerBlock(in_dim, out_dim, n_heads, lsa_blocks)
        self.fuse = nn.Sequential(
            nn.Conv1d(2 * out_dim, out_dim, kernel_size=1),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.downsample = nn.AvgPool1d(2, 2) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.fuse(torch.cat([self.cnn(x), self.lgtb(x)], dim=1))
        return self.downsample(fused)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class _FCDecoder(nn.Module):
    """Paper decoding module: two convolutions plus one fully connected layer."""

    def __init__(self, in_dim: int, bottleneck_length: int, out_length: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, in_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_dim // 2, in_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear((in_dim // 4) * bottleneck_length, out_length)
        self.out_length = out_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        return self.fc(h.flatten(1)).unsqueeze(1)


class _ConvDecoder(nn.Module):
    """Transposed-convolution decoder for context lengths the FC head cannot afford."""

    def __init__(self, in_dim: int, n_upsamples: int, out_length: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(n_upsamples):
            out_dim = max(16, dim // 2)
            layers += [
                nn.ConvTranspose1d(dim, out_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            dim = out_dim
        layers += [nn.Conv1d(dim, 1, kernel_size=3, padding=1)]
        self.body = nn.Sequential(*layers)
        self.out_length = out_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        return _fit_length(h, self.out_length)


def _fit_length(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[-1] == length:
        return x
    if x.shape[-1] > length:
        return x[..., :length]
    return torch.nn.functional.pad(x, (0, length - x.shape[-1]), mode="replicate")


class _Branch(nn.Module):
    """Preprocessing -> 5 CNN-LGTB blocks -> fusion -> decoder (paper §2.2.2)."""

    def __init__(
        self,
        signal_length: int,
        encoder_dims: tuple[int, ...],
        n_heads: int,
        lsa_blocks: int,
        decoder_head: str,
    ) -> None:
        super().__init__()
        self.stem = PreprocessingModule(1, PAPER_STEM_DIM)
        blocks: list[nn.Module] = []
        in_dim = PAPER_STEM_DIM
        for out_dim in encoder_dims:
            blocks.append(CNNLGTBBlock(in_dim, out_dim, n_heads, lsa_blocks))
            in_dim = out_dim
        self.blocks = nn.ModuleList(blocks)

        # "The output information of the last encoding block is combined through a
        # feature fusion layer, and after passing through a convolutional layer and
        # batch normalization layer, it is output as extracted features."
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, kernel_size=1),
            nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        n_reductions = 1 + len(encoder_dims)  # stem pool + one per block
        bottleneck = max(1, math.ceil(signal_length / (2**n_reductions)))
        self.decoder_head = decoder_head
        if decoder_head == "fc":
            self.decoder: nn.Module = _FCDecoder(in_dim, bottleneck, signal_length)
        else:
            self.decoder = _ConvDecoder(in_dim, n_reductions, signal_length)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocessing, the five CNN-LGTB blocks, and the feature fusion layer."""
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return self.feature_fusion(h)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class _GatingNetwork(nn.Module):
    """Paper Eq. 4: two fully connected layers with tanh, producing two masks."""

    def __init__(self, signal_length: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(signal_length, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2 * signal_length),
            nn.Tanh(),
        )
        self.signal_length = signal_length

    def forward(self, x_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masks = self.net(x_raw.squeeze(1))
        mask1, mask2 = masks.chunk(2, dim=-1)
        return mask1.unsqueeze(1), mask2.unsqueeze(1)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class DHCTGanStrictGenerator(nn.Module):
    """DHCT-GAN generator: two branches, two gating networks, three outputs.

    Input is ``(B, context_epochs, channels, T)``; output is a dict with

    ``clean``
        branch 1's clean estimate ``Y1`` (paper Eq. 6's supervision target),
    ``noise``
        branch 2's noise estimate ``Y2`` — the gradient artifact here,
    ``fused``
        ``Y_pre = mask1 * Y1 + mask2 * (X_raw - Y2)`` (Eq. 5),

    each ``(B, 1, core_samples)`` for the centre epoch of the target channel.

    Context handling (**FACETpy extension**, not the paper)
    ------------------------------------------------------
    The epochs are concatenated in time so each electrode becomes one long
    1-D waveform, which is exactly the shape the paper's branch consumes; the
    branch weights are shared across electrodes, and a permutation-equivariant
    attention bridge lets them exchange information at the encoder's deepest
    level. At ``context_epochs=1, n_channels=1`` no concatenation and no bridge
    apply, and the forward pass is the paper's.
    """

    def __init__(
        self,
        context_epochs: int = 7,
        n_channels: int = 3,
        core_samples: int = 512,
        encoder_dims: tuple[int, ...] | list[int] = PAPER_ENCODER_DIMS,
        n_heads: int = 4,
        lsa_blocks: int = PAPER_LSA_BLOCKS,
        decoder_head: str = "auto",
        cross_channel_attention: bool = True,
        max_fc_head_params: int = 16_000_000,
        input_normalisation: bool = True,
    ) -> None:
        super().__init__()
        self.input_normalisation = bool(input_normalisation)
        self.context_epochs = int(context_epochs)
        self.n_channels = int(n_channels)
        self.core_samples = int(core_samples)
        self.signal_length = self.context_epochs * self.core_samples
        encoder_dims = tuple(int(v) for v in encoder_dims)

        resolved = _resolve_decoder_head(decoder_head, encoder_dims, self.signal_length, max_fc_head_params)
        self.decoder_head = resolved

        self.clean_branch = _Branch(self.signal_length, encoder_dims, n_heads, lsa_blocks, resolved)
        self.noise_branch = _Branch(self.signal_length, encoder_dims, n_heads, lsa_blocks, resolved)
        self.gate = _GatingNetwork(self.signal_length)

        self.cross_channel = (
            _CrossChannelBridge(encoder_dims[-1], n_heads) if cross_channel_attention and self.n_channels > 1 else None
        )

        # Mix the electrodes down to the target channel. Channel 0 is the target by
        # the Weg-A builder's ordering (self first, then montage neighbours), so the
        # mixer is initialised as "pass channel 0 through" and has to earn any use
        # of the neighbours. With a single channel there is nothing to mix and the
        # layer is dropped entirely, so paper mode carries no extra parameters.
        if self.n_channels > 1:
            mixer = nn.Conv1d(self.n_channels, 1, kernel_size=1)
            with torch.no_grad():
                mixer.weight.zero_()
                mixer.weight[0, 0, 0] = 1.0
                if mixer.bias is not None:
                    mixer.bias.zero_()
            self.channel_mixer: nn.Module = mixer
        else:
            self.channel_mixer = nn.Identity()

    # ------------------------------------------------------------------

    def _to_paper_shape(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """``(B, E, C, T)`` -> ``(B*C, 1, E*T)``: one 1-D waveform per electrode."""
        b, ep, ch, t = x.shape
        if ep != self.context_epochs or ch != self.n_channels:
            raise ValueError(
                f"Generator built for {self.context_epochs} epochs x {self.n_channels} channels, got {ep} x {ch}"
            )
        if t != self.core_samples:
            raise ValueError(f"Generator built for {self.core_samples} samples per epoch, got {t}")
        return x.permute(0, 2, 1, 3).reshape(b * ch, 1, ep * t), b

    def _centre_and_mix(self, y: torch.Tensor, batch: int) -> torch.Tensor:
        """``(B*C, 1, E*T)`` -> ``(B, 1, T)``: centre epoch, then mix electrodes."""
        ep, t = self.context_epochs, self.core_samples
        centre = y.reshape(batch, self.n_channels, ep * t)
        start = (ep // 2) * t
        return self.channel_mixer(centre[..., start : start + t])

    def _input_scale(self, flat: torch.Tensor) -> torch.Tensor:
        """Per-sequence RMS of the input, detached, as a broadcastable divisor."""
        if not self.input_normalisation:
            return torch.ones((), device=flat.device, dtype=flat.dtype)
        rms = flat.detach().pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()
        return rms.clamp_min(1e-12)

    def _run_branch(self, branch: _Branch, flat: torch.Tensor, batch: int) -> torch.Tensor:
        """Run one paper branch with the electrode bridge inserted at the bottleneck.

        The bridge sits after the paper's feature fusion layer and before the
        decoder: the one place where every electrode is represented at the same
        abstraction and the sequence is short enough for attention to be cheap.
        """
        features = branch.encode(flat)
        if self.cross_channel is not None:
            _, dim, length = features.shape
            bridged = self.cross_channel(features.reshape(batch, self.n_channels, dim, length))
            features = bridged.reshape(batch * self.n_channels, dim, length)
        return branch.decode(features)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected (batch, context_epochs, channels, samples), got {tuple(x.shape)}")
        flat, batch = self._to_paper_shape(x)

        # Work in the paper's amplitude regime. The paper's segments are O(1);
        # FACETpy signals are volts, so the decoder's final convolution would have
        # to emit ~1e-3 with no normalisation after it. Dividing by the per-example
        # input RMS on the way in and multiplying back on the way out puts the whole
        # branch in O(1) space and leaves the returned values in volts.
        scale = self._input_scale(flat)
        flat = flat / scale

        y1 = self._run_branch(self.clean_branch, flat, batch)  # Y1: clean estimate
        y2 = self._run_branch(self.noise_branch, flat, batch)  # Y2: noise estimate
        mask1, mask2 = self.gate(flat)  # Eq. 4
        fused = mask1 * y1 + mask2 * (flat - y2)  # Eq. 5
        y1, y2, fused = y1 * scale, y2 * scale, fused * scale

        return {
            "clean": self._centre_and_mix(y1, batch),
            "noise": self._centre_and_mix(y2, batch),
            "fused": self._centre_and_mix(fused, batch),
        }

    def export_module(self) -> DHCTGanStrictArtifactExport:
        """Single-tensor view for TorchScript export (``cli._export_pytorch_torchscript``).

        Without this hook the exporter would try to trace a three-output generator
        into a single-tensor TorchScript module and fail. What ships is the
        artifact the correction pipeline subtracts, derived from the gated output
        the paper actually deploys.
        """
        return DHCTGanStrictArtifactExport(self, mode="fused")

    def describe(self) -> dict[str, Any]:
        """Configuration facts a run report should not have to re-derive."""
        return {
            "context_epochs": self.context_epochs,
            "n_channels": self.n_channels,
            "core_samples": self.core_samples,
            "signal_length": self.signal_length,
            "decoder_head": self.decoder_head,
            "cross_channel_attention": self.cross_channel is not None,
            "input_normalisation": self.input_normalisation,
            "paper_single_channel_mode": self.context_epochs == 1 and self.n_channels == 1,
            "n_params": sum(p.numel() for p in self.parameters()),
        }


class _CrossChannelBridge(nn.Module):
    """Attention across electrodes at the encoder bottleneck (FACETpy extension).

    Electrodes are a set, not a sequence: no positional encoding is added, so the
    layer is equivariant to permuting them and cannot learn a fixed slot order.
    Which electrode is the target is decided by the channel mixer downstream.
    """

    def __init__(self, dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, _fit_heads(dim, n_heads), batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, D, T)`` in and out; attention runs over C at each time step.

        The layout is an explicit tensor shape rather than module state. An earlier
        version stashed ``(batch, channels)`` on the module between calls, which
        works in eager mode but makes the module non-reentrant and made
        ``torch.jit.trace`` fail its own graph-consistency check.
        """
        b, c, d, t = x.shape
        h = x.permute(0, 3, 1, 2).reshape(b * t, c, d)
        h_norm = self.norm(h)
        attended, _ = self.attn(h_norm, h_norm, h_norm, need_weights=False)
        return (h + attended).reshape(b, t, c, d).permute(0, 2, 3, 1)


def _resolve_decoder_head(
    requested: str,
    encoder_dims: tuple[int, ...],
    signal_length: int,
    max_fc_head_params: int,
) -> str:
    """Pick the paper's FC head when it is affordable, otherwise the conv decoder."""
    requested = requested.strip().lower()
    if requested in {"fc", "conv"}:
        return requested
    if requested != "auto":
        raise ValueError(f"decoder_head must be 'auto', 'fc' or 'conv', got '{requested}'")
    n_reductions = 1 + len(encoder_dims)
    bottleneck = max(1, math.ceil(signal_length / (2**n_reductions)))
    fc_params = (encoder_dims[-1] // 4) * bottleneck * signal_length
    return "fc" if fc_params <= max_fc_head_params else "conv"


# ---------------------------------------------------------------------------
# Discriminator (paper Fig. 2a)
# ---------------------------------------------------------------------------


class DHCTGanStrictDiscriminator(nn.Module):
    """M = 8 strided convolutions with batch norm, plus a feature tap.

    Paper Fig. 2a and §2.2.2: "The input signals undergo processing through M
    convolutional layers, each layer followed by a batch normalization layer and
    an activation function layer. In this research, M is defined as 8. The output
    channels ... are 64, 64, 128, 128, 256, 256, 512, and 512 ... kernel size ...
    3, with a stride of 2 and padding of 1."

    Returns ``(logits, features)``; *features* are the intermediate activations
    Eq. 11 matches on.
    """

    def __init__(self, in_channels: int = 1, dims: tuple[int, ...] = PAPER_DISCRIMINATOR_DIMS) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_channels
        for out_dim in dims:
            layers.append(
                nn.Sequential(
                    nn.Conv1d(dim, out_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(out_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            dim = out_dim
        self.layers = nn.ModuleList(layers)
        self.head = nn.Conv1d(dim, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features: list[torch.Tensor] = []
        h = x
        for layer in self.layers:
            h = layer(h)
            if h.shape[-1] < 1:
                break
            features.append(h)
        return self.head(h), features


# ---------------------------------------------------------------------------
# Objective (paper Eqs. 6-13)
# ---------------------------------------------------------------------------


class DHCTGanStrictObjective(AdversarialObjective):
    """``L_total = Loss1 + Loss2 + Loss3`` with three discriminators.

    Each ``Loss_k = L_mse + lambda1 * L_feat + lambda2 * L_adv`` (Eqs. 6-8),
    with LSGAN adversarial and discriminator terms (Eqs. 12-13) and
    discriminator feature matching (Eq. 11).

    Discriminator-to-target pairing follows §2.2.2 exactly:

    ==========  ====================  =========================
    Head        real signal           what it judges
    ==========  ====================  =========================
    ``clean``   clean EEG             branch 1's clean estimate
    ``noise``   artifact              branch 2's noise estimate
    ``fused``   clean EEG             the gated final output
    ==========  ====================  =========================

    The target tensor is the Weg-A layout ``[artifact, clean]`` (build the dataset
    with ``target_extras=("clean",)``); the ordering is asserted at construction
    time rather than assumed, because silently swapping clean and artifact would
    train a perfectly convergent, perfectly wrong model.

    Amplitude scale (``normalise_scale``)
    ------------------------------------
    ``lambda1 = 1.0`` and ``lambda2 = 0.1`` are only meaningful when the signal is
    O(1), which it is in the paper: EEGdenoiseNet segments are microvolt-scale and
    mixed at -7...+2 dB SNR. FACETpy signals are stored in **volts**, so
    ``L_mse ~ 1e-10`` while ``L_feat`` and ``L_adv`` -- computed on discriminator
    activations and logits -- are O(1). Ten orders of magnitude apart.

    That is not a rounding concern, it is the whole objective. Measured on the
    first strict run (paper configuration, 84 epochs): the reconstruction term
    contributed **1e-8 % of the loss**, and the model it produced amplified spikes
    3.5x at a morphology correlation of 0.053, scoring 22.2 uV overall RMSE --
    *worse* than the 18.8 uV a model that outputs nothing scores. It had optimised
    the discriminators' feature space and ignored the signal.

    With ``normalise_scale=True`` every quantity the loss touches -- generator
    outputs and references alike, for MSE, feature matching and the adversarial
    term -- is divided by **one** per-example scale: the RMS of the reconstructed
    input ``artifact + clean``. See :meth:`_scale` for why it must be shared across
    the heads rather than per-head. All three terms are then O(1) with respect to
    the input and the paper's weights recover their intended balance. Model outputs
    stay in volts, so export and evaluation are unaffected.

    This normalisation is **not stated in the paper**; it is what makes the paper's
    published weights transferable to a signal four orders of magnitude larger.
    ``normalise_scale=False`` reproduces the unscaled behaviour.
    """

    def __init__(
        self,
        lambda_feat: float = 1.0,
        lambda_adv: float = 0.1,
        artifact_row: int = 0,
        clean_row: int = 1,
        in_channels: int = 1,
        discriminator_dims: tuple[int, ...] | list[int] = PAPER_DISCRIMINATOR_DIMS,
        adversarial_warmup_steps: int = 0,
        normalise_scale: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.lambda_feat = float(lambda_feat)
        self.lambda_adv = float(lambda_adv)
        self.normalise_scale = bool(normalise_scale)
        self.eps = float(eps)
        self.artifact_row = int(artifact_row)
        self.clean_row = int(clean_row)
        self.adversarial_warmup_steps = int(adversarial_warmup_steps)
        self._steps = 0
        dims = tuple(int(v) for v in discriminator_dims)
        for name in ("clean", "noise", "fused"):
            self.register_discriminator(name, DHCTGanStrictDiscriminator(in_channels, dims))

    # -- target plumbing -------------------------------------------------

    def _row(self, target: torch.Tensor, index: int) -> torch.Tensor:
        if target.shape[-2] <= index:
            raise ValueError(
                f"DHCTGanStrictObjective needs a target with at least {index + 1} rows "
                f"([artifact, clean]); got shape {tuple(target.shape)}. Build the dataset "
                "with target_extras=('clean',)."
            )
        return target[..., index : index + 1, :]

    def _real_for(self, name: str, target: torch.Tensor) -> torch.Tensor:
        row = self.artifact_row if name == "noise" else self.clean_row
        return self._row(target, row)

    def _scale(self, target: torch.Tensor) -> torch.Tensor:
        """One divisor per example: the RMS of the *input* signal.

        Derived from ``artifact + clean``, which reconstructs the noisy input the
        generator saw. Deliberately **one** scale shared by all three heads rather
        than each head's own reference RMS: the artifact is ~56x the clean here, so
        per-head scaling divides the clean head's error by a 56x smaller number and
        amplifies it by 56^2. That was measured — the clean branch went unstable
        within four epochs, swinging from ``mse_clean`` 25 to 3.8e6 while the noise
        branch sat at 1.05.

        With a shared scale the relative magnitudes between heads are preserved,
        every term is O(1) with respect to the input, and the paper's lambda
        weights keep their meaning. Detached: a unit conversion, not a term the
        generator can game.
        """
        if not self.normalise_scale:
            return torch.ones((), device=target.device, dtype=target.dtype)
        noisy = self._row(target, self.artifact_row) + self._row(target, self.clean_row)
        rms = noisy.detach().pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()
        return rms.clamp_min(self.eps)

    # -- objective -------------------------------------------------------

    def discriminator_losses(self, outputs, target):
        losses: dict[str, torch.Tensor] = {}
        scale = self._scale(target)
        for name, disc in self.discriminators.items():
            real = self._real_for(name, target)
            d_real, _ = disc(real / scale)
            d_fake, _ = disc(outputs[name] / scale)
            losses[name] = lsgan_discriminator_loss(d_real, d_fake)
        return losses

    def generator_loss(self, outputs, target):
        adversarial_on = self._steps >= self.adversarial_warmup_steps
        if self.training:
            self._steps += 1

        total = torch.zeros((), device=target.device, dtype=target.dtype)
        metrics: dict[str, float] = {}
        scale = self._scale(target)
        for name, disc in self.discriminators.items():
            real = self._real_for(name, target) / scale
            prediction = outputs[name] / scale

            mse = torch.nn.functional.mse_loss(prediction, real)  # Eq. 10
            d_fake, feats_fake = disc(prediction)
            _, feats_real = disc(real)
            feat = feature_matching_loss(feats_fake, feats_real)  # Eq. 11
            adv = lsgan_generator_loss(d_fake) if adversarial_on else torch.zeros_like(mse)  # Eq. 12

            total = total + mse + self.lambda_feat * feat + self.lambda_adv * adv
            metrics[f"mse_{name}"] = float(mse.detach())
            metrics[f"feat_{name}"] = float(feat.detach())
            metrics[f"adv_{name}"] = float(adv.detach())
        metrics["adversarial_on"] = float(adversarial_on)
        return total, metrics

    def primary_output(self, outputs):
        """The gated output is what the paper deploys (§2.2.4, testing phase)."""
        return outputs["fused"]


# ---------------------------------------------------------------------------
# facet-train factories
# ---------------------------------------------------------------------------


def build_model(
    input_shape: tuple[int, int, int] | None = None,
    context_epochs: int | None = None,
    n_channels: int | None = None,
    core_samples: int | None = None,
    encoder_dims: tuple[int, ...] | list[int] = PAPER_ENCODER_DIMS,
    n_heads: int = 4,
    lsa_blocks: int = PAPER_LSA_BLOCKS,
    decoder_head: str = "auto",
    cross_channel_attention: bool = True,
    max_fc_head_params: int = 16_000_000,
    input_normalisation: bool = True,
    **_: object,
) -> DHCTGanStrictGenerator:
    """Build the generator; shape comes from the dataset unless overridden."""
    if input_shape is not None:
        ds_epochs, ds_channels, ds_samples = (int(v) for v in input_shape)
    else:
        ds_epochs, ds_channels, ds_samples = 7, 3, 512
    return DHCTGanStrictGenerator(
        context_epochs=int(context_epochs if context_epochs is not None else ds_epochs),
        n_channels=int(n_channels if n_channels is not None else ds_channels),
        core_samples=int(core_samples if core_samples is not None else ds_samples),
        encoder_dims=encoder_dims,
        n_heads=int(n_heads),
        lsa_blocks=int(lsa_blocks),
        decoder_head=decoder_head,
        cross_channel_attention=bool(cross_channel_attention),
        max_fc_head_params=int(max_fc_head_params),
        input_normalisation=bool(input_normalisation),
    )


def build_loss(
    lambda_feat: float = 1.0,
    lambda_adv: float = 0.1,
    artifact_row: int = 0,
    clean_row: int = 1,
    discriminator_dims: tuple[int, ...] | list[int] = PAPER_DISCRIMINATOR_DIMS,
    adversarial_warmup_steps: int = 0,
    normalise_scale: bool = True,
    **_: object,
) -> DHCTGanStrictObjective:
    """Build the three-discriminator objective.

    Named ``build_loss`` so the existing config schema applies, but this returns an
    :class:`~facet.training.adversarial.AdversarialObjective`, not a plain loss:
    it owns the discriminators and is only usable with
    :func:`build_wrapper`.
    """
    return DHCTGanStrictObjective(
        lambda_feat=float(lambda_feat),
        lambda_adv=float(lambda_adv),
        artifact_row=int(artifact_row),
        clean_row=int(clean_row),
        discriminator_dims=discriminator_dims,
        adversarial_warmup_steps=int(adversarial_warmup_steps),
        normalise_scale=bool(normalise_scale),
    )


def build_wrapper(
    model: DHCTGanStrictGenerator,
    loss_fn: DHCTGanStrictObjective | None = None,
    device: str = "cpu",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    grad_clip_norm: float | None = 1.0,
    discriminator_learning_rate: float | None = 1e-4,
    generator_betas: tuple[float, float] | list[float] = (0.5, 0.9),
    discriminator_betas: tuple[float, float] | list[float] = (0.9, 0.999),
    n_discriminator_steps: int = 1,
    micro_batch_size: int | None = None,
    warmup_steps: int = 0,
    scheduler_cls: type | None = None,
    scheduler_kwargs: dict | None = None,
    **_: object,
) -> AdversarialModelWrapper:
    """Wire generator, objective and the paper's two distinct Adam configurations.

    Paper §2.2.4: "the generator's Adam parameters set to beta1 = 0.5 and
    beta2 = 0.9, and the discriminator's Adam parameters set to beta1 = 0.9 and
    beta2 = 0.999". Two networks with different betas is precisely what the
    single-optimizer wrapper had no way to express.

    ``micro_batch_size`` keeps the paper's batch size reachable when the model does
    not fit it. The multi-electrode configuration is memory-hungry — the LGTB's
    global attention spans the full 1792-token post-pooling window in the first
    encoding block — so without accumulation a memory-limited run would silently
    train at a much smaller effective batch than the paper's 40, which is a
    different recipe rather than a slower one.
    """
    objective = loss_fn if loss_fn is not None else build_loss()
    if not isinstance(objective, DHCTGanStrictObjective):
        raise TypeError(
            "dhct_gan_strict_edition requires model.loss_factory to build a "
            f"DHCTGanStrictObjective, got {type(objective).__name__}"
        )
    return AdversarialModelWrapper(
        model=model,
        objective=objective,
        device=device,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={"betas": tuple(float(v) for v in generator_betas)},
        discriminator_optimizer_cls=torch.optim.Adam,
        discriminator_optimizer_kwargs={"betas": tuple(float(v) for v in discriminator_betas)},
        discriminator_learning_rate=discriminator_learning_rate,
        n_discriminator_steps=int(n_discriminator_steps),
        micro_batch_size=micro_batch_size,
        warmup_steps=int(warmup_steps),
        scheduler_cls=scheduler_cls,
        scheduler_kwargs=scheduler_kwargs,
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        grad_clip_norm=grad_clip_norm,
    )


def build_dataset(
    path: str = DEFAULT_DATASET,
    max_examples: int | None = None,
    max_channels: int | None = None,
    max_shift: int | None = None,
    demean_input: bool = False,
    demean_target: bool = False,
    residual_mode: bool = False,
    seed: int = 0,
    **_: object,
) -> NPZSpatioTemporalDataset:
    """Weg-A spatio-temporal dataset with ``[artifact, clean]`` targets.

    The clean row is not optional here: D1 and D3 judge against the real clean EEG
    and D2 against the real artifact, so a target carrying only one of the two
    cannot express the paper's objective at all.

    ``max_channels`` selects the configuration:

    ``1``
        The **paper configuration**. The seven epochs are concatenated into one
        3584-sample 1-D waveform, which is the input format DHCT-GAN was written
        for (§2.1 uses 1024-sample 1-D segments); only the window length differs.
        The multi-epoch window is also what keeps this out of the
        one-epoch-and-one-channel case that carries no comparison at all.
    ``None`` (or > 1)
        The **FACETpy extension**: the target electrode plus its nearest geodesic
        neighbours, with the cross-electrode attention bridge active.

    Running both and comparing them is the ablation that says what spatial context
    is worth here; neither number may be reported as the other's.
    """
    return NPZSpatioTemporalDataset(
        path=Path(path).expanduser(),
        target_key="artifact_center",
        target_extras=("clean",),
        max_examples=max_examples,
        max_channels=max_channels,
        max_shift=max_shift,
        demean_input=demean_input,
        demean_target=demean_target,
        residual_mode=residual_mode,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Inference export
# ---------------------------------------------------------------------------


class DHCTGanStrictArtifactExport(nn.Module):
    """Single-tensor view of the generator for TorchScript export and inference.

    The FACETpy correction pipeline subtracts one artifact tensor; the generator
    returns three. Which one to export is a real choice, not a formality:

    * ``noise`` is branch 2's raw artifact estimate,
    * ``noisy - fused`` is the artifact implied by what the paper actually deploys
      (§2.2.4: "the denoised EEG signal predicted by the generator is used as the
      final output"), i.e. it passes through the gating of Eq. 5.

    The default is ``"fused"`` because deploying branch 2 alone would silently ship
    a different model than the one the paper evaluates. ``mode="noise"`` is kept
    for the ablation that asks how much the gating contributes.
    """

    def __init__(self, generator: DHCTGanStrictGenerator, mode: str = "fused") -> None:
        super().__init__()
        if mode not in {"fused", "noise"}:
            raise ValueError(f"mode must be 'fused' or 'noise', got '{mode}'")
        self.generator = generator
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.generator(x)
        if self.mode == "noise":
            return outputs["noise"]
        ep, t = self.generator.context_epochs, self.generator.core_samples
        start = (ep // 2) * t
        noisy_centre = x[:, ep // 2, 0:1, :] if x.shape[-1] == t else x[..., start : start + t]
        return noisy_centre - outputs["fused"]
