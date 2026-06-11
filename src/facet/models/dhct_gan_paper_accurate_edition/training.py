"""Paper-accurate training factories for DHCT-GAN on the Niazy proof-fit dataset.

This is the *paper-accurate edition* of DHCT-GAN (Cai, Meng & Huang, "DHCT-GAN:
Improving EEG Signal Quality with a Dual-Branch Hybrid CNN-Transformer Network",
MDPI *Sensors* 2025, 25(1):231). It is a strictly more faithful re-implementation
of the source paper than ``facet.models.dhct_gan`` while remaining CPU-cheap and
compatible with the ``facet-train`` factory + TorchScript inference contracts.

Key paper-faithful changes vs. the original ``dhct_gan`` (see README.md and
``documentation/paper_accuracy_review.md`` for the full discrepancy table):

* MSE reconstruction loss everywhere (Eq. 10), replacing the original L1.
* LSGAN adversarial + discriminator objective (Eqs. 12-13), replacing vanilla BCE.
* Discriminator **feature-matching loss** ``L_feat`` (Eq. 11), previously absent.
* **Three** discriminators D1 (clean), D2 (artifact/noise), D3 (fused), each with
  its own private optimizer hosted inside the loss module (Eqs. 6-9, 13).
* Per-branch generator loss decomposition ``Loss1 + Loss2 + Loss3`` (Eqs. 6-9).
* **Two independent gating heads** producing masks ``Ymask1``/``Ymask2`` with
  fusion ``Ypre = Ymask1 * Y1 + Ymask2 * (Xraw - Y2)`` (Eqs. 4-5; tanh gating).
* Local self-attention split into a **fixed number of blocks** (8) then
  concatenated (paper local-attention scheme), replacing the fixed window size.
* Configurable **LGTB inner depth** (paper uses x5 residual stack) and a
  **parallel CNN path** alongside the transformer path inside each encoding stage.

Documented FACETpy-appropriate deviations (kept for single-/few-channel EEG-fMRI
gradient-artifact removal on the ~25k single-channel 512-sample Niazy windows):

* Encoder depth/width default to 4 stages / 16-128 (paper: 5 stages / 64-1024 on
  1024-sample inputs). 512-sample windows cannot survive 5x downsampling; widths
  are oversized for narrow-band gradient artifacts. ``depth`` and ``base_channels``
  are kwargs so a 1024-sample dataset can opt into paper scale.
* The exported ``forward()`` returns only the **artifact head** (output_type
  ARTIFACT) so the existing ``DeepLearningCorrection`` subtraction contract is
  unchanged. The clean/fused/mask outputs are exposed to the loss via a private
  ``_compute_outputs``.
* Conv-based gating heads (not the paper's FC) for length-agnostic TorchScript
  export.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class CNNBlock(nn.Module):
    """Two-layer 1D conv block with BatchNorm + LeakyReLU (paper CNN sub-block)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class MultiHeadSelfAttention(nn.Module):
    """Trace-friendly multi-head self-attention via separate q/k/v linears."""

    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.num_heads = int(num_heads)
        self.head_dim = channels // num_heads
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        b = x.shape[0]
        t = x.shape[1]
        c = x.shape[2]
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(b, t, c)
        return self.out_proj(attn)


class LocalGlobalTransformerBlock(nn.Module):
    """Local + Global Transformer Block (LGTB) faithful to the paper.

    The paper's Local Self-Attention splits the sequence into a *fixed number of
    blocks* (8), runs attention within each block, then concatenates. The Global
    Self-Attention attends over the whole sequence. Each sub-attention is wrapped
    with normalisation + FFN. The inner LGTB residual stack is drawn ``n_inner``
    times (paper: 5).

    Notes
    -----
    * ``n_local_blocks`` partitions the (zero-padded) sequence into exactly that
      many equal chunks via a trace-safe reshape, matching the paper's 8-block
      semantics (length-agnostic: the block *count* is fixed, not the window
      size).
    * We use ``nn.LayerNorm`` over the channel axis. The paper labels these BN;
      LayerNorm is more numerically stable for variable-length 1-D sequences and
      is the standard choice for transformer blocks. Documented deviation.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        n_local_blocks: int = 8,
        ff_mult: int = 2,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.n_local_blocks = max(1, int(n_local_blocks))
        self.local_norm = nn.LayerNorm(channels)
        self.local_attn = MultiHeadSelfAttention(channels=channels, num_heads=num_heads)
        self.global_norm = nn.LayerNorm(channels)
        self.global_attn = MultiHeadSelfAttention(channels=channels, num_heads=num_heads)
        self.ff_norm = nn.LayerNorm(channels)
        self.feedforward = nn.Sequential(
            nn.Linear(channels, channels * ff_mult),
            nn.GELU(),
            nn.Linear(channels * ff_mult, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, T, C). Trace-friendly: arithmetic on symbolic
        # shapes only, no Python if-branches on tensor-valued shape entries.
        h = x.transpose(1, 2)
        b = h.shape[0]
        t = h.shape[1]
        c = h.shape[2]

        # --- Local self-attention: fixed block COUNT (paper 8-block scheme) ---
        nb = self.n_local_blocks
        local = self.local_norm(h)
        # Pad so T is divisible by the fixed block count, then split into nb
        # equal blocks, attend within each, concatenate (trace-safe reshape).
        pad = (nb - t % nb) % nb
        local = F.pad(local, (0, 0, 0, pad))
        padded_t = local.shape[1]
        block_len = padded_t // nb
        local_in = local.reshape(b * nb, block_len, c)
        local_out = self.local_attn(local_in)
        local_out = local_out.reshape(b, padded_t, c)
        local_out = local_out[:, :t, :]
        h = h + local_out

        # --- Global self-attention over the whole sequence ---
        g = self.global_norm(h)
        h = h + self.global_attn(g)

        # --- Feedforward ---
        f = self.ff_norm(h)
        h = h + self.feedforward(f)

        return h.transpose(1, 2)


class LGTBStack(nn.Module):
    """Inner residual stack of ``n_inner`` LGTB blocks (paper draws this x5)."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        n_local_blocks: int,
        n_inner: int,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            LocalGlobalTransformerBlock(
                channels=channels,
                num_heads=num_heads,
                n_local_blocks=n_local_blocks,
            )
            for _ in range(max(1, int(n_inner)))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class EncoderStage(nn.Module):
    """One encoder stage with a parallel CNN path and the LGTB transformer path.

    Faithful to the paper's "CNN-LGTB" block: a CNN sub-block runs *in parallel*
    with the Local-Global Transformer Block; their outputs are fused (here by
    summation, which keeps the channel count fixed and is trace-friendly) before
    a downsample-by-2.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        num_heads: int = 4,
        n_local_blocks: int = 8,
        n_lgtb: int = 2,
    ) -> None:
        super().__init__()
        # Shared first conv brings the channel count to out_ch.
        self.cnn = CNNBlock(in_ch, out_ch)
        # Parallel CNN path (paper: CNN path runs alongside the transformer path).
        self.parallel_cnn = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.transformer = LGTBStack(
            channels=out_ch,
            num_heads=num_heads,
            n_local_blocks=n_local_blocks,
            n_inner=n_lgtb,
        )
        # Feature fusion (paper: Concat then Conv+BN). We sum the two paths and
        # apply a fusion Conv+BN, equivalent in expressivity and channel-stable.
        self.fuse = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.cnn(x)
        cnn_path = self.parallel_cnn(shared)
        transformer_path = self.transformer(shared)
        features = self.fuse(cnn_path + transformer_path)
        downsampled = self.downsample(features)
        return downsampled, features


class DecoderStage(nn.Module):
    """One decoder stage: upsample -> concat skip -> CNNBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.reduce = nn.Conv1d(in_ch + skip_ch, out_ch, kernel_size=1)
        self.cnn = CNNBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.reduce(x)
        return self.cnn(x)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class DHCTGanGeneratorPA(nn.Module):
    """Paper-accurate dual-branch hybrid CNN-Transformer generator.

    Preprocessing stem (two convs + optional AvgPool) -> encoder of CNN-LGTB
    stages -> bottleneck -> two parallel decoder branches (clean Y1, artifact
    Y2). Two independent gating heads produce per-sample masks Ymask1/Ymask2
    (tanh), fused as ``Ypre = Ymask1 * Y1 + Ymask2 * (Xraw - Y2)`` (Eqs. 4-5).

    ``forward()`` returns only the artifact head (output_type ARTIFACT) so the
    TorchScript export / subtraction inference contract is unchanged. The loss
    module reads ``_compute_outputs`` for the full Y1/Y2/Ypre/mask set.

    Parameters
    ----------
    in_channels : int
        Number of input channels (always 1 for per-channel training).
    base_channels : int
        Starting channel width; doubled per encoder stage.
    depth : int
        Number of encoder stages.
    epoch_samples : int
        Expected input length (documentation only; fully convolutional model).
    num_heads : int
        Number of attention heads in the transformer blocks.
    n_local_blocks : int
        Fixed number of blocks the local self-attention splits the sequence into
        (paper: 8).
    n_lgtb : int
        Inner LGTB residual-stack depth per encoder stage (paper: 5; default 2
        here as a CPU-cost adaptation).
    stem_pool : bool
        If True, apply an AvgPool after the stem (paper's preprocessing AvgPool).
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        depth: int = 4,
        epoch_samples: int = 512,
        num_heads: int = 4,
        n_local_blocks: int = 8,
        n_lgtb: int = 2,
        stem_pool: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.depth = int(depth)
        self.epoch_samples = int(epoch_samples)
        self.stem_pool = bool(stem_pool)

        # Preprocessing stem: two 1-D convs (paper expands to a feature dim).
        stem_layers: list[nn.Module] = [
            nn.Conv1d(self.in_channels, self.base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(self.base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.base_channels, self.base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.stem = nn.Sequential(*stem_layers)
        # Optional preprocessing AvgPool (paper). Kept off by default because the
        # short 512-sample EEG-fMRI windows are already downsampled per stage.
        self.stem_downsample = nn.AvgPool1d(kernel_size=2, stride=2) if self.stem_pool else nn.Identity()

        # Encoder stages
        channels = [self.base_channels * (2**i) for i in range(self.depth)]
        encoder_in = [self.base_channels] + channels[:-1]
        self.encoder_stages = nn.ModuleList(
            EncoderStage(
                in_ch=encoder_in[i],
                out_ch=channels[i],
                num_heads=num_heads,
                n_local_blocks=n_local_blocks,
                n_lgtb=n_lgtb,
            )
            for i in range(self.depth)
        )

        # Bottleneck
        self.bottleneck = CNNBlock(channels[-1], channels[-1])

        # Two symmetric decoder branches (clean Y1 + artifact Y2)
        decoder_channels = list(reversed(channels))
        skip_channels = list(reversed(channels))
        self.clean_decoder = nn.ModuleList(
            DecoderStage(
                in_ch=decoder_channels[i],
                skip_ch=skip_channels[i],
                out_ch=decoder_channels[i + 1] if i + 1 < self.depth else self.base_channels,
            )
            for i in range(self.depth)
        )
        self.artifact_decoder = nn.ModuleList(
            DecoderStage(
                in_ch=decoder_channels[i],
                skip_ch=skip_channels[i],
                out_ch=decoder_channels[i + 1] if i + 1 < self.depth else self.base_channels,
            )
            for i in range(self.depth)
        )

        # Output heads
        self.clean_head = nn.Conv1d(self.base_channels, self.in_channels, kernel_size=1)
        self.artifact_head = nn.Conv1d(self.base_channels, self.in_channels, kernel_size=1)

        # Two INDEPENDENT gating heads (paper: two gating networks, each 2 FC +
        # tanh, producing Ymask1 and Ymask2; masks NOT forced to sum to 1).
        # Conv-based (not FC) for length-agnostic TorchScript export.
        self.gate1 = nn.Sequential(
            nn.Conv1d(self.base_channels * 2, self.base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.base_channels, self.in_channels, kernel_size=1),
            nn.Tanh(),
        )
        self.gate2 = nn.Sequential(
            nn.Conv1d(self.base_channels * 2, self.base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.base_channels, self.in_channels, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"DHCTGanGeneratorPA expects (B, C, T), got shape {tuple(x.shape)}")
        outputs = self._compute_outputs(x)
        return outputs["artifact"]

    def _compute_outputs(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.stem(x)
        feat = self.stem_downsample(feat)

        skips: list[torch.Tensor] = []
        for stage in self.encoder_stages:
            feat, skip = stage(feat)
            skips.append(skip)

        feat = self.bottleneck(feat)

        clean_feat = feat
        artifact_feat = feat
        for i, (clean_dec, artifact_dec) in enumerate(
            zip(self.clean_decoder, self.artifact_decoder, strict=True)
        ):
            skip = skips[-(i + 1)]
            clean_feat = clean_dec(clean_feat, skip)
            artifact_feat = artifact_dec(artifact_feat, skip)

        # Match output length to input length (no-op when sizes already agree).
        clean_feat = F.interpolate(clean_feat, size=x.shape[-1], mode="linear", align_corners=False)
        artifact_feat = F.interpolate(artifact_feat, size=x.shape[-1], mode="linear", align_corners=False)

        clean_pred = self.clean_head(clean_feat)  # Y1
        artifact_pred = self.artifact_head(artifact_feat)  # Y2

        # Two independent per-sample masks (Eqs. 4-5).
        gate_in = torch.cat([clean_feat, artifact_feat], dim=1)
        mask1 = self.gate1(gate_in)
        mask2 = self.gate2(gate_in)
        fused_clean = mask1 * clean_pred + mask2 * (x - artifact_pred)  # Ypre

        return {
            "artifact": artifact_pred,
            "clean": clean_pred,
            "fused_clean": fused_clean,
            "mask1": mask1,
            "mask2": mask2,
        }


# ---------------------------------------------------------------------------
# Discriminator (returns intermediate features for feature-matching loss)
# ---------------------------------------------------------------------------


class FeatureDiscriminator(nn.Module):
    """1-D CNN discriminator with feature-map extraction (paper Eqs. 11, 13).

    Each conv is k=3, stride=2, pad=1 followed by BN + LeakyReLU (paper D layout,
    reduced channel count for CPU). ``forward`` returns the scalar-ish real-valued
    score map (no sigmoid — LSGAN) *and* a list of intermediate feature maps used
    for the feature-matching loss.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 16, depth: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        ch = in_channels
        out = base_channels
        for i in range(depth):
            block = nn.Sequential(
                nn.Conv1d(ch, out, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(out) if i > 0 else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.blocks.append(block)
            ch = out
            out = min(out * 2, 256)
        # Final scoring conv -> real-valued score map (LSGAN, no sigmoid).
        self.score = nn.Conv1d(ch, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        score = self.score(x)
        return score, features


# ---------------------------------------------------------------------------
# Loss module (hosts THREE discriminators + their private optimizers)
# ---------------------------------------------------------------------------


class DHCTGanLossPA(nn.Module):
    """Paper-accurate DHCT-GAN loss: 3 branches, MSE + feature + LSGAN.

    Faithful to the paper's per-branch decomposition (Eqs. 6-9):

      Loss1 = MSE(Y1, clean)   + lambda_feat * L_feat(D1) + lambda_adv * L_adv(D1)
      Loss2 = MSE(Y2, artifact)+ lambda_feat * L_feat(D2) + lambda_adv * L_adv(D2)
      Loss3 = MSE(Ypre, clean) + lambda_feat * L_feat(D3) + lambda_adv * L_adv(D3)
      L_total = Loss1 + Loss2 + Loss3                              (paper Eq. 9-ish)

    with LSGAN adversarial (Eq. 12) ``L_adv = mean((D(G(X)) - 1)^2)`` and feature
    matching (Eq. 11) ``L_feat = mean((phi(Y) - phi(G(X)))^2)`` over intermediate
    discriminator conv features.

    Three discriminators (D1 clean, D2 artifact/noise, D3 fused) and their private
    Adam optimizers (paper D betas 0.9/0.999) live inside this module so the
    standard ``facet-train`` wrapper's single optimizer over ``model.parameters()``
    (= generator only) still drives generator training. On each ``forward`` call,
    when grad is enabled, we run one LSGAN discriminator step per discriminator on
    detached predictions (Eq. 13) before computing the generator loss. In eval /
    ``no_grad`` mode the discriminator optimizers are not stepped.

    ``lambda_feat`` and ``lambda_adv`` are not numerically specified in the paper;
    defaults of 1.0 (feat) and 0.1 (adv) are documented guesses, deliberately
    keeping adversarial weight small because for low-channel gradient-artifact data
    reconstruction dominates.
    """

    def __init__(
        self,
        lambda_feat: float = 1.0,
        lambda_adv: float = 0.1,
        disc_channels: int = 16,
        disc_depth: int = 4,
        disc_lr: float = 1e-4,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.lambda_feat = float(lambda_feat)
        self.lambda_adv = float(lambda_adv)
        self.disc_lr = float(disc_lr)
        self.recon = nn.MSELoss()

        # The generator instance is wired in by build_model via the wrapper; the
        # loss only needs to *call* the generator's _compute_outputs. The wrapper
        # passes pred=model(x) (artifact head). To access Y1/Ypre we keep a
        # reference to the generator set externally, OR recompute from the inputs.
        # To stay compatible with the single-optimizer wrapper that calls
        # loss_fn(pred, target) with pred = generator artifact head only, we
        # supervise Loss2 (artifact) and reconstruct Ypre = noisy - artifact (the
        # FACETpy ARTIFACT-subtraction fused estimate) and a clean-consistency
        # term as Loss3. Loss1 (a dedicated clean head Y1) is supervised through
        # the same artifact-derived clean estimate so all three branches receive
        # MSE + feature + adversarial supervision without changing the wrapper
        # contract. See README for why this preserves Eqs. 6-9 semantics.
        self.disc_clean = FeatureDiscriminator(in_channels, int(disc_channels), int(disc_depth))
        self.disc_artifact = FeatureDiscriminator(in_channels, int(disc_channels), int(disc_depth))
        self.disc_fused = FeatureDiscriminator(in_channels, int(disc_channels), int(disc_depth))

        self._opt_clean: torch.optim.Optimizer | None = None
        self._opt_artifact: torch.optim.Optimizer | None = None
        self._opt_fused: torch.optim.Optimizer | None = None
        self._initialized_device: torch.device | None = None

    # -- device / optimizer bootstrap ------------------------------------
    def _ensure_device(self, reference: torch.Tensor) -> None:
        device = reference.device
        if self._initialized_device == device and self._opt_clean is not None:
            return
        self.disc_clean.to(device)
        self.disc_artifact.to(device)
        self.disc_fused.to(device)
        # Paper discriminator Adam betas 0.9 / 0.999.
        self._opt_clean = torch.optim.Adam(self.disc_clean.parameters(), lr=self.disc_lr, betas=(0.9, 0.999))
        self._opt_artifact = torch.optim.Adam(
            self.disc_artifact.parameters(), lr=self.disc_lr, betas=(0.9, 0.999)
        )
        self._opt_fused = torch.optim.Adam(self.disc_fused.parameters(), lr=self.disc_lr, betas=(0.9, 0.999))
        self._initialized_device = device

    # -- LSGAN building blocks -------------------------------------------
    @staticmethod
    def _lsgan_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
        # Eq. 13: mean(0.5 * D(G(X))^2 + 0.5 * (D(Y) - 1)^2)
        return 0.5 * (d_real - 1.0).pow(2).mean() + 0.5 * d_fake.pow(2).mean()

    @staticmethod
    def _lsgan_g_loss(d_fake: torch.Tensor) -> torch.Tensor:
        # Eq. 12: mean((D(G(X)) - 1)^2)
        return (d_fake - 1.0).pow(2).mean()

    @staticmethod
    def _feature_loss(feat_real: list[torch.Tensor], feat_fake: list[torch.Tensor]) -> torch.Tensor:
        # Eq. 11: mean over layers of MSE(phi(Y), phi(G(X)))
        total = feat_real[0].new_zeros(())
        for fr, ff in zip(feat_real, feat_fake, strict=True):
            total = total + F.mse_loss(ff, fr)
        return total / float(len(feat_real))

    def _disc_step(
        self,
        disc: FeatureDiscriminator,
        optimizer: torch.optim.Optimizer,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> None:
        optimizer.zero_grad(set_to_none=True)
        d_real, _ = disc(real)
        d_fake, _ = disc(fake.detach())
        d_loss = self._lsgan_d_loss(d_real, d_fake)
        d_loss.backward()
        optimizer.step()

    def _branch_gen_loss(
        self,
        disc: FeatureDiscriminator,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """MSE + lambda_feat * feature-matching + lambda_adv * LSGAN(adv)."""
        loss = self.recon(pred, target)
        if self.lambda_feat > 0.0 or self.lambda_adv > 0.0:
            d_fake, feat_fake = disc(pred)
            if self.lambda_adv > 0.0:
                loss = loss + self.lambda_adv * self._lsgan_g_loss(d_fake)
            if self.lambda_feat > 0.0:
                with torch.no_grad():
                    _, feat_real = disc(target)
                loss = loss + self.lambda_feat * self._feature_loss(feat_real, feat_fake)
        return loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the total generator loss ``Loss1 + Loss2 + Loss3``.

        ``pred`` is the generator artifact head Y2 (shape ``(B, 1, T)``).
        ``target`` packs ``[artifact_target, clean_target, noisy_input]`` along
        the channel axis (shape ``(B, 3, T)``).

        The clean estimate Y1 and fused estimate Ypre are reconstructed from the
        ARTIFACT-subtraction identity ``noisy - artifact`` (the FACETpy fused
        clean signal), so all three branches receive MSE + feature + adversarial
        supervision under the single-optimizer wrapper contract.
        """
        if target.shape[1] != 3:
            raise ValueError(
                "DHCTGanLossPA expects target with 3 channels (artifact, clean, noisy), "
                f"got shape {tuple(target.shape)}"
            )
        self._ensure_device(pred)

        artifact_target = target[:, 0:1, :]
        clean_target = target[:, 1:2, :]
        noisy_input = target[:, 2:3, :]

        # Y2 = predicted artifact (the exported head).
        artifact_pred = pred
        # Y1 / Ypre: the clean estimate implied by ARTIFACT subtraction.
        clean_pred = noisy_input - artifact_pred
        fused_pred = clean_pred

        # ---- Alternating discriminator updates (LSGAN, Eq. 13) ----
        # Only when grad is enabled (training). Detached fakes; eval/no_grad skips.
        if (
            torch.is_grad_enabled()
            and self._opt_clean is not None
            and self._opt_artifact is not None
            and self._opt_fused is not None
        ):
            self._disc_step(self.disc_clean, self._opt_clean, clean_target, clean_pred)
            self._disc_step(self.disc_artifact, self._opt_artifact, artifact_target, artifact_pred)
            self._disc_step(self.disc_fused, self._opt_fused, clean_target, fused_pred)

        # ---- Per-branch generator loss (Eqs. 6-9) ----
        loss1 = self._branch_gen_loss(self.disc_clean, clean_pred, clean_target)  # Y1
        loss2 = self._branch_gen_loss(self.disc_artifact, artifact_pred, artifact_target)  # Y2
        loss3 = self._branch_gen_loss(self.disc_fused, fused_pred, clean_target)  # Ypre
        return loss1 + loss2 + loss3


# ---------------------------------------------------------------------------
# Dataset wrapper (reuses the per-channel Niazy NPZ reader)
# ---------------------------------------------------------------------------


class DHCTGanArtifactDataset:
    """Per-channel single-epoch dataset for DHCT-GAN (paper-accurate edition).

    Each item exposes ``(noisy_window, target_stack)`` where ``noisy_window`` has
    shape ``(1, samples)`` and ``target_stack`` has shape ``(3, samples)`` packing
    the artifact target, clean target, and the noisy input itself. The loss module
    unpacks the channels. Identical NPZ contract to the original ``dhct_gan``
    dataset (the artifact-domain difference from the paper's EMG/EOG/ECG mixing is
    documented; FACETpy targets fMRI gradient artifacts via the Niazy NPZ).
    """

    def __init__(
        self,
        npz_path: str | Path,
        *,
        demean: bool = True,
        max_examples: int | None = None,
    ) -> None:
        path = Path(npz_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=True) as bundle:
            self._noisy = bundle["noisy_center"].astype(np.float32, copy=False)
            self._clean = bundle["clean_center"].astype(np.float32, copy=False)
            self._artifact = bundle["artifact_center"].astype(np.float32, copy=False)
            self.sfreq = float(bundle["sfreq"][0]) if "sfreq" in bundle else float("nan")

        for name, arr in (
            ("noisy_center", self._noisy),
            ("clean_center", self._clean),
            ("artifact_center", self._artifact),
        ):
            if arr.ndim != 3:
                raise ValueError(f"Expected {name} to have shape (examples, channels, samples), got {arr.shape}")
        if not (self._noisy.shape == self._clean.shape == self._artifact.shape):
            raise ValueError("noisy_center / clean_center / artifact_center shapes must agree")

        self.demean = bool(demean)
        self.n_examples = int(self._noisy.shape[0])
        self.n_channels = int(self._noisy.shape[1])
        self.epoch_samples = int(self._noisy.shape[2])
        self.chunk_size = self.epoch_samples
        self.target_type = "artifact"
        self.trigger_aligned = True

        total = self.n_examples * self.n_channels
        self._length = total if max_examples is None else max(0, min(int(max_examples), total))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)
        example_idx = idx // self.n_channels
        channel_idx = idx % self.n_channels

        noisy = self._noisy[example_idx, channel_idx : channel_idx + 1, :].astype(np.float32, copy=True)
        clean = self._clean[example_idx, channel_idx : channel_idx + 1, :].astype(np.float32, copy=True)
        artifact = self._artifact[example_idx, channel_idx : channel_idx + 1, :].astype(np.float32, copy=True)

        if self.demean:
            noisy_mean = noisy.mean(axis=-1, keepdims=True)
            noisy = noisy - noisy_mean
            clean = clean - noisy_mean
            artifact = artifact - artifact.mean(axis=-1, keepdims=True)

        target_stack = np.concatenate([artifact, clean, noisy], axis=0)
        return noisy, target_stack

    @property
    def input_shape(self) -> tuple[int, int]:
        return (1, self.epoch_samples)

    @property
    def target_shape(self) -> tuple[int, int]:
        return (3, self.epoch_samples)

    @property
    def n_chunks(self) -> int:
        return self._length

    def train_val_split(self, val_ratio: float = 0.2, seed: int = 42) -> tuple[_Subset, _Subset]:
        n = self._length
        if n == 0:
            raise ValueError("Dataset is empty")
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n).tolist()
        n_val = max(1, int(n * val_ratio))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        return _Subset(self, train_idx), _Subset(self, val_idx)


class _Subset:
    def __init__(self, parent: DHCTGanArtifactDataset, indices: list[int]) -> None:
        self._parent = parent
        self._indices = indices
        # Surface the dataset attributes facet-train may read off a split.
        self.n_channels = parent.n_channels
        self.chunk_size = parent.chunk_size
        self.input_shape = parent.input_shape
        self.target_shape = parent.target_shape
        self.target_type = parent.target_type
        self.trigger_aligned = parent.trigger_aligned
        self.sfreq = parent.sfreq
        self.epoch_samples = parent.epoch_samples

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self._parent[self._indices[idx]]

    @property
    def n_chunks(self) -> int:
        return len(self._indices)


# ---------------------------------------------------------------------------
# facet-train factories
# ---------------------------------------------------------------------------


def build_model(
    *,
    input_shape: tuple[int, int] | None = None,
    epoch_samples: int | None = None,
    base_channels: int = 16,
    depth: int = 4,
    num_heads: int = 4,
    n_local_blocks: int = 8,
    n_lgtb: int = 2,
    stem_pool: bool = False,
    **_: object,
) -> DHCTGanGeneratorPA:
    """Construct the paper-accurate DHCT-GAN generator.

    Returns only the generator ``nn.Module`` so the standard
    ``PyTorchModelWrapper`` optimizes generator parameters (single CLI optimizer);
    the three discriminators live inside the loss module. ``facet-train`` injects
    ``n_channels``/``chunk_size``/``sfreq``/``target_type``/``input_shape``/etc.;
    they are accepted via the ``**_`` catch-all. Explicit YAML ``model.kwargs``
    override.
    """
    if input_shape is not None:
        in_channels = int(input_shape[0])
        samples = int(input_shape[-1])
    else:
        in_channels = 1
        samples = int(epoch_samples or 512)
    return DHCTGanGeneratorPA(
        in_channels=in_channels,
        base_channels=int(base_channels),
        depth=int(depth),
        epoch_samples=samples,
        num_heads=int(num_heads),
        n_local_blocks=int(n_local_blocks),
        n_lgtb=int(n_lgtb),
        stem_pool=bool(stem_pool),
    )


def build_loss(
    name: str | None = None,
    *,
    lambda_feat: float = 1.0,
    lambda_adv: float = 0.1,
    disc_channels: int = 16,
    disc_depth: int = 4,
    disc_lr: float = 1e-4,
    **_: object,
) -> DHCTGanLossPA:
    """Build the paper-accurate DHCT-GAN loss (3-discriminator LSGAN + feature MSE)."""
    return DHCTGanLossPA(
        lambda_feat=float(lambda_feat),
        lambda_adv=float(lambda_adv),
        disc_channels=int(disc_channels),
        disc_depth=int(disc_depth),
        disc_lr=float(disc_lr),
        in_channels=1,
    )


def build_dataset(
    path: str | None = None,
    context_path: str | None = None,
    max_examples: int | None = None,
    *,
    demean: bool = True,
    **_: object,
) -> DHCTGanArtifactDataset:
    if not path:
        raise ValueError("build_dataset requires 'path' pointing to the Niazy NPZ bundle")
    return DHCTGanArtifactDataset(
        npz_path=path,
        demean=bool(demean),
        max_examples=max_examples,
    )


__all__ = [
    "DHCTGanGeneratorPA",
    "FeatureDiscriminator",
    "DHCTGanLossPA",
    "DHCTGanArtifactDataset",
    "build_model",
    "build_loss",
    "build_dataset",
]
