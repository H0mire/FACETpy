"""Paper-accurate training factories for DHCT-GAN v2.

This edition makes the FACETpy DHCT-GAN v2 adaptation substantially more
faithful to the source paper:

    Cai, Meng & Huang, "DHCT-GAN: Improving EEG Signal Quality with a
    Dual-Branch Hybrid CNN-Transformer Network", MDPI Sensors 25(1) 231 (2025).

Paper-faithful changes vs the original ``dhct_gan_v2`` (see README.md and
``documentation/paper_accuracy_review.md`` for the full discrepancy table):

* **LSGAN** least-squares adversarial + discriminator loss (Eq. 12-13)
  replaces vanilla BCE.
* **Feature-matching loss** ``L_feat`` (Eq. 11) on an intermediate
  discriminator activation.
* **MSE reconstruction** (Eq. 10) is the faithful default (L1 retained as a
  documented option).
* **Three discriminators** (clean branch / noise branch / fused output) as in
  the paper's multi-discriminator design (Algorithm 1), all driven by a single
  private Adam so the facet-train single-optimizer contract still holds.
* **Two independent tanh gating networks** producing ``Y_mask1`` and
  ``Y_mask2`` (Eq. 4-5) replace the single complementary sigmoid gate.
* **Paper-faithful LGTB**: Local Self-Attention splits the sequence into a
  configurable number of *equal blocks* (default 8, the paper value) instead of
  a sliding window, a feedforward follows *both* the local and the global
  attention, and the LGTB is repeated ``lgtb_depth`` times per stage.

Deliberate, documented deviations kept for EEG-fMRI / CPU economy:

* 7-epoch trigger-aligned context input (the gradient artifact is strongly
  TR-periodic; the paper is single-segment).
* Shared encoder + dual decoders (instead of two fully-duplicated branch
  generators) to roughly halve parameters.
* Reduced dims (512-sample epochs, depth 4, smaller channel widths) matching
  the tiny Niazy proof-fit bundle; full-scale dims are exposed as kwargs.
* The generator ``forward`` returns the *artifact* (``noisy_center -
  fused_clean``) for the FACETpy subtractive-correction contract rather than
  the clean signal the paper emits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from facet.training.dataset import NPZContextArtifactDataset

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class CNNBlock(nn.Module):
    """Two-layer 1D conv block with BatchNorm + LeakyReLU (paper kernel 3)."""

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
        b = x.shape[0]
        t = x.shape[1]
        c = x.shape[2]
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(b, t, c)
        return self.out_proj(attn)


class _FeedForward(nn.Module):
    """Per-attention feedforward (LayerNorm -> Linear -> GELU -> Linear)."""

    def __init__(self, channels: int, ff_mult: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.net = nn.Sequential(
            nn.Linear(channels, channels * ff_mult),
            nn.GELU(),
            nn.Linear(channels * ff_mult, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class LocalGlobalTransformerBlock(nn.Module):
    """Paper-faithful Local-Global Transformer Block (LGTB).

    Pipeline (Fig. 2a, Sec. 3.2):

        LSA -> FeedForward -> GSA -> FeedForward

    Local Self-Attention (LSA) splits the sequence into ``local_blocks``
    *equal* contiguous blocks (default 8 as in the paper) and attends within
    each block, then concatenates — not the original v2 sliding window. Global
    Self-Attention (GSA) attends over the full sequence. A feedforward follows
    *each* attention (the paper's two-FF pattern), unlike the original's single
    trailing FF.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        local_blocks: int = 8,
        ff_mult: int = 2,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.local_blocks = max(1, int(local_blocks))
        self.local_norm = nn.LayerNorm(channels)
        self.local_attn = MultiHeadSelfAttention(channels=channels, num_heads=num_heads)
        self.local_ff = _FeedForward(channels, ff_mult=ff_mult)
        self.global_norm = nn.LayerNorm(channels)
        self.global_attn = MultiHeadSelfAttention(channels=channels, num_heads=num_heads)
        self.global_ff = _FeedForward(channels, ff_mult=ff_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> work in (B, T, C)
        h = x.transpose(1, 2)
        b = h.shape[0]
        t = h.shape[1]
        c = h.shape[2]

        # --- Local Self-Attention over `local_blocks` equal blocks -----------
        n_blocks = self.local_blocks
        if n_blocks > t:
            n_blocks = t if t > 0 else 1
        block_len = (t + n_blocks - 1) // n_blocks  # ceil
        pad = block_len * n_blocks - t
        local = self.local_norm(h)
        if pad > 0:
            local = F.pad(local, (0, 0, 0, pad))
        padded_t = local.shape[1]
        local_in = local.reshape(b * n_blocks, block_len, c)
        local_out = self.local_attn(local_in)
        local_out = local_out.reshape(b, padded_t, c)[:, :t, :]
        h = h + local_out
        h = self.local_ff(h)

        # --- Global Self-Attention over the full sequence --------------------
        g = self.global_norm(h)
        g_out = self.global_attn(g)
        h = h + g_out
        h = self.global_ff(h)

        return h.transpose(1, 2)


class LGTBStack(nn.Module):
    """``lgtb_depth`` repeated LGTBs (the paper repeats the LGTB x5 per stage)."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        local_blocks: int = 8,
        lgtb_depth: int = 2,
        ff_mult: int = 2,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            LocalGlobalTransformerBlock(
                channels=channels,
                num_heads=num_heads,
                local_blocks=local_blocks,
                ff_mult=ff_mult,
            )
            for _ in range(max(1, int(lgtb_depth)))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class EncoderStage(nn.Module):
    """One encoder stage: CNNBlock -> LGTB stack -> downsample-by-2."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        num_heads: int = 4,
        local_blocks: int = 8,
        lgtb_depth: int = 2,
    ) -> None:
        super().__init__()
        self.cnn = CNNBlock(in_ch, out_ch)
        self.transformer = LGTBStack(
            channels=out_ch,
            num_heads=num_heads,
            local_blocks=local_blocks,
            lgtb_depth=lgtb_depth,
        )
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.transformer(self.cnn(x))
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


class _GatingNetwork(nn.Module):
    """One gating network: the paper uses two FC layers + tanh (Eq. 4-5).

    FACETpy operates on variable-length resampled epochs, so a fixed FC over the
    time axis is awkward; we use the 1D-conv equivalent (kernel 3 + 1x1) with a
    tanh activation. The two gating networks (mask1, mask2) are *independent*
    (not constrained to sum to 1), as in the paper.
    """

    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden_ch, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv1d(hidden_ch, out_ch, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Generator (multi-epoch context input)
# ---------------------------------------------------------------------------


class DHCTGanV2PaperAccurateGenerator(nn.Module):
    """Paper-accurate DHCT-GAN v2 generator with multi-epoch context input.

    Input shape ``(B, context_epochs, T)`` — the center epoch is at index
    ``context_epochs // 2``. The 7 context epochs are stacked as channels so
    the stem mixes them at every sample.

    The shared encoder feeds two decoders (clean branch ``Y1`` and noise branch
    ``Y2``). Two independent tanh gating networks produce ``mask1`` and
    ``mask2``, and the fused clean signal is (paper Eq. 4-5)::

        Y_pre = mask1 * Y1 + mask2 * (X_raw - Y2)

    where ``X_raw`` is the noisy center epoch and ``Y2`` is the noise-branch
    output (the artifact head). ``forward`` returns the *artifact*
    ``noisy_center - Y_pre`` for the FACETpy subtractive-correction contract;
    :meth:`_compute_outputs` exposes ``clean_pred``, ``artifact_pred``,
    ``fused_clean`` and the masks for the loss.
    """

    def __init__(
        self,
        context_epochs: int = 7,
        out_channels: int = 1,
        base_channels: int = 16,
        depth: int = 4,
        epoch_samples: int = 512,
        num_heads: int = 4,
        local_blocks: int = 8,
        lgtb_depth: int = 2,
    ) -> None:
        super().__init__()
        if context_epochs < 1:
            raise ValueError("context_epochs must be >= 1")
        self.context_epochs = int(context_epochs)
        self.out_channels = int(out_channels)
        self.base_channels = int(base_channels)
        self.depth = int(depth)
        self.epoch_samples = int(epoch_samples)
        self.num_heads = int(num_heads)
        self.local_blocks = int(local_blocks)
        self.lgtb_depth = int(lgtb_depth)
        self.center_index = self.context_epochs // 2

        self.stem = nn.Sequential(
            nn.Conv1d(self.context_epochs, self.base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(self.base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.base_channels, self.base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        channels = [self.base_channels * (2**i) for i in range(self.depth)]
        encoder_in = [self.base_channels] + channels[:-1]
        self.encoder_stages = nn.ModuleList(
            EncoderStage(
                in_ch=encoder_in[i],
                out_ch=channels[i],
                num_heads=num_heads,
                local_blocks=local_blocks,
                lgtb_depth=lgtb_depth,
            )
            for i in range(self.depth)
        )

        self.bottleneck = CNNBlock(channels[-1], channels[-1])

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

        self.clean_head = nn.Conv1d(self.base_channels, self.out_channels, kernel_size=1)
        self.artifact_head = nn.Conv1d(self.base_channels, self.out_channels, kernel_size=1)

        # Two INDEPENDENT tanh gating networks (paper Eq. 4-5). Each sees both
        # branch feature maps so it can reconcile the clean and noise paths.
        self.gate1 = _GatingNetwork(self.base_channels * 2, self.base_channels, self.out_channels)
        self.gate2 = _GatingNetwork(self.base_channels * 2, self.base_channels, self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"DHCTGanV2PaperAccurateGenerator expects (B, C, T), got shape {tuple(x.shape)}"
            )
        outputs = self._compute_outputs(x)
        # Subtractive-correction contract: export the artifact derived from the
        # FULL dual-branch + gating machinery, so the gate influences inference.
        return outputs["artifact_from_fused"]

    def _compute_outputs(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        noisy_center = x[:, self.center_index : self.center_index + 1, :]

        feat = self.stem(x)

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

        clean_feat = F.interpolate(clean_feat, size=x.shape[-1], mode="linear", align_corners=False)
        artifact_feat = F.interpolate(
            artifact_feat, size=x.shape[-1], mode="linear", align_corners=False
        )

        clean_pred = self.clean_head(clean_feat)  # Y1
        artifact_pred = self.artifact_head(artifact_feat)  # Y2 (noise branch)

        gate_in = torch.cat([clean_feat, artifact_feat], dim=1)
        mask1 = self.gate1(gate_in)
        mask2 = self.gate2(gate_in)
        # Eq. 4-5: Y_pre = mask1 ⊙ Y1 + mask2 ⊙ (X_raw - Y2)
        fused_clean = mask1 * clean_pred + mask2 * (noisy_center - artifact_pred)
        artifact_from_fused = noisy_center - fused_clean

        return {
            "artifact": artifact_pred,
            "artifact_from_fused": artifact_from_fused,
            "clean": clean_pred,
            "fused_clean": fused_clean,
            "mask1": mask1,
            "mask2": mask2,
            "noisy_center": noisy_center,
        }


# ---------------------------------------------------------------------------
# Discriminator (paper structure: M strided convs + FC + feature tap)
# ---------------------------------------------------------------------------


class DHCTGanV2PaperAccurateDiscriminator(nn.Module):
    """Paper-faithful discriminator (Fig. 2b).

    ``depth`` strided conv blocks (kernel 3, stride 2, padding 1) with
    BatchNorm + LeakyReLU, then a global-average pool + FC producing a scalar
    score. The paper uses M=8 blocks with channels 64,64,128,128,256,256,
    512,512; here the channel progression and depth are configurable and scaled
    down for CPU. ``forward`` returns ``(score, feature)`` where ``feature`` is
    a mid-network activation used for the feature-matching loss ``L_feat``
    (Eq. 11).
    """

    # Paper channel progression (per pair of conv layers): 64,64,128,128,...
    _PAPER_WIDTHS = [64, 64, 128, 128, 256, 256, 512, 512]

    def __init__(self, in_channels: int = 1, base_channels: int = 16, depth: int = 8) -> None:
        super().__init__()
        depth = max(2, int(depth))
        # Scale the paper progression by base_channels / 64 so the smoke test can
        # shrink it (base_channels small -> tiny widths) while preserving the
        # paper's doubling-every-two-layers shape.
        scale = float(base_channels) / 64.0
        widths = [max(1, int(round(self._PAPER_WIDTHS[min(i, len(self._PAPER_WIDTHS) - 1)] * scale)))
                  for i in range(depth)]

        blocks: list[nn.Module] = []
        ch = in_channels
        for i, w in enumerate(widths):
            block = [nn.Conv1d(ch, w, kernel_size=3, stride=2, padding=1)]
            if i > 0:  # paper applies BN from the second block onward
                block.append(nn.BatchNorm1d(w))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            blocks.append(nn.Sequential(*block))
            ch = w
        self.blocks = nn.ModuleList(blocks)
        # Tap a middle block for feature-matching (perceptual) loss.
        self._feature_index = len(self.blocks) // 2
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(ch, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature: torch.Tensor | None = None
        h = x
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i == self._feature_index:
                feature = h
        if feature is None:
            feature = h
        pooled = self.pool(h).squeeze(-1)
        score = self.fc(pooled)  # raw LSGAN score (no sigmoid)
        return score, feature


# ---------------------------------------------------------------------------
# Loss module (LSGAN + feature-matching + MSE, three discriminators)
# ---------------------------------------------------------------------------


class DHCTGanV2PaperAccurateLoss(nn.Module):
    """Paper-accurate DHCT-GAN v2 loss.

    Implements the paper's three-branch objective (Eq. 6-13):

        L_total = Loss1 + Loss2 + Loss3
        Loss_i  = L_mse + lambda1 * L_feat + lambda2 * L_adv

    with **LSGAN** adversarial/discriminator losses, **feature-matching**
    ``L_feat``, **MSE** reconstruction, and **three** discriminators:

    * ``D_clean``  judges the clean branch ``Y1`` vs the clean target.
    * ``D_noise``  judges the noise branch ``Y2`` (artifact) vs the artifact
      target.
    * ``D_fused``  judges the fused output (``noisy_center - artifact_from_fused``,
      i.e. ``Y_pre``) vs the clean target.

    All three live inside this module and share ONE private Adam (betas
    0.9, 0.999, lr ``disc_lr``) so the facet-train single-optimizer contract is
    preserved. Each gradient-enabled forward runs the alternating LSGAN
    discriminator update against detached generator outputs, then returns the
    summed generator loss.

    The loss is called by facet-train as ``loss_fn(pred, target)``. Here
    ``pred`` is the artifact returned by the generator (shape ``(B, 1, T)``,
    equal to ``noisy_center - fused_clean``), and ``target`` packs
    ``[artifact_target, clean_target, noisy_center]`` along the channel axis
    (shape ``(B, 3, T)``). The clean/noise branch predictions are reconstructed
    from ``pred`` and ``target`` as ``Y_pre = noisy_center - pred`` (fused) and
    ``Y2 = pred`` (noise); the explicit clean-branch ``Y1`` is not separately
    exported through the facet-train loss contract, so its discriminator/recon
    terms use the fused clean estimate as a faithful proxy (documented in the
    review). This keeps the (B,3,T) packing intact while realising LSGAN +
    feature-matching + multi-discriminator stabilisation.
    """

    def __init__(
        self,
        alpha_consistency: float = 0.5,
        lambda_feat: float = 0.1,
        lambda_adv: float = 0.1,
        disc_channels: int = 16,
        disc_depth: int = 8,
        disc_lr: float = 1e-4,
        recon: str = "mse",
        in_channels: int = 1,
        # Backwards-compat alias: callers using the original `beta_adv` name.
        beta_adv: float | None = None,
    ) -> None:
        super().__init__()
        self.alpha_consistency = float(alpha_consistency)
        self.lambda_feat = float(lambda_feat)
        self.lambda_adv = float(beta_adv if beta_adv is not None else lambda_adv)
        self.disc_lr = float(disc_lr)
        recon = str(recon).lower()
        if recon not in {"mse", "l1"}:
            raise ValueError("recon must be 'mse' or 'l1'")
        self.recon_name = recon
        self.recon = nn.MSELoss() if recon == "mse" else nn.L1Loss()

        # Three discriminators sharing one architecture (paper D1/D2/D3).
        self.disc_clean = DHCTGanV2PaperAccurateDiscriminator(
            in_channels=in_channels, base_channels=int(disc_channels), depth=int(disc_depth)
        )
        self.disc_noise = DHCTGanV2PaperAccurateDiscriminator(
            in_channels=in_channels, base_channels=int(disc_channels), depth=int(disc_depth)
        )
        self.disc_fused = DHCTGanV2PaperAccurateDiscriminator(
            in_channels=in_channels, base_channels=int(disc_channels), depth=int(disc_depth)
        )

        self._disc_optimizer: torch.optim.Optimizer | None = None
        self._initialized_device: torch.device | None = None

    # -- helpers ----------------------------------------------------------
    def _discriminators(self) -> list[DHCTGanV2PaperAccurateDiscriminator]:
        return [self.disc_clean, self.disc_noise, self.disc_fused]

    def _ensure_device(self, reference: torch.Tensor) -> None:
        device = reference.device
        if self._initialized_device == device and self._disc_optimizer is not None:
            return
        params: list[torch.nn.Parameter] = []
        for disc in self._discriminators():
            disc.to(device)
            params += list(disc.parameters())
        self._disc_optimizer = torch.optim.Adam(params, lr=self.disc_lr, betas=(0.9, 0.999))
        self._initialized_device = device

    @staticmethod
    def _lsgan_d_loss(score_fake: torch.Tensor, score_real: torch.Tensor) -> torch.Tensor:
        # L_D = 0.5 * mean(D(fake)^2) + 0.5 * mean((D(real) - 1)^2)   (Eq. 13)
        return 0.5 * (score_fake**2).mean() + 0.5 * ((score_real - 1.0) ** 2).mean()

    @staticmethod
    def _lsgan_g_adv(score_fake: torch.Tensor) -> torch.Tensor:
        # L_adv = mean((D(G(X)) - 1)^2)   (Eq. 12)
        return ((score_fake - 1.0) ** 2).mean()

    # -- forward ----------------------------------------------------------
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.shape[1] != 3:
            raise ValueError(
                "DHCTGanV2PaperAccurateLoss expects target with 3 channels "
                f"(artifact, clean, noisy_center), got shape {tuple(target.shape)}"
            )

        self._ensure_device(pred)

        artifact_target = target[:, 0:1, :]
        clean_target = target[:, 1:2, :]
        noisy_center = target[:, 2:3, :]

        # Reconstruct the branch signals from the facet-train (pred, target)
        # contract. pred == noisy_center - fused_clean (artifact_from_fused).
        fused_clean = noisy_center - pred  # Y_pre
        noise_pred = pred  # Y2 proxy (artifact)
        clean_pred = fused_clean  # Y1 proxy (no separate FC export)

        # Pair each discriminator with its (fake, real) signals.
        disc_pairs = [
            (self.disc_clean, clean_pred, clean_target),
            (self.disc_noise, noise_pred, artifact_target),
            (self.disc_fused, fused_clean, clean_target),
        ]

        # ---- Discriminator update (LSGAN, against detached generator) -------
        if torch.is_grad_enabled() and self._disc_optimizer is not None:
            self._disc_optimizer.zero_grad(set_to_none=True)
            d_loss_total = pred.new_zeros(())
            for disc, fake, real in disc_pairs:
                score_fake, _ = disc(fake.detach())
                score_real, _ = disc(real)
                d_loss_total = d_loss_total + self._lsgan_d_loss(score_fake, score_real)
            d_loss_total.backward()
            self._disc_optimizer.step()

        # ---- Generator loss: sum over the three branches --------------------
        # Reconstruction (MSE by default, Eq. 10): noise branch + fused-clean
        # consistency term.
        recon_noise = self.recon(noise_pred, artifact_target)
        recon_consistency = self.recon(fused_clean, clean_target)
        generator_loss = recon_noise + self.alpha_consistency * recon_consistency

        for disc, fake, real in disc_pairs:
            score_fake, feat_fake = disc(fake)
            if self.lambda_adv > 0.0:
                generator_loss = generator_loss + self.lambda_adv * self._lsgan_g_adv(score_fake)
            if self.lambda_feat > 0.0:
                with torch.no_grad():
                    _, feat_real = disc(real)
                generator_loss = generator_loss + self.lambda_feat * F.mse_loss(feat_fake, feat_real)

        return generator_loss


# ---------------------------------------------------------------------------
# Dataset wrapper (channel-wise 7-epoch context view)
# ---------------------------------------------------------------------------


class DHCTGanV2PaperAccurateContextDataset:
    """Per-channel multi-epoch context dataset (paper-accurate edition).

    Identical contract to the original ``DHCTGanV2ContextDataset``: each item
    exposes ``(noisy_context, target_stack)`` where ``noisy_context`` has shape
    ``(context_epochs, samples)`` and ``target_stack`` has shape ``(3, samples)``
    packing ``[artifact_target, clean_target, noisy_center]``.

    Wraps :class:`NPZContextArtifactDataset`. The data domain (fMRI gradient
    artifact via the Niazy proof-fit NPZ) is a deliberate, documented deviation
    from the paper's physiological-artifact data.
    """

    def __init__(
        self,
        base_dataset: Any,
        *,
        context_epochs: int = 7,
        npz_path: str | Path | None = None,
        demean_input: bool = True,
        demean_target: bool = True,
        max_examples: int | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.context_epochs = int(context_epochs)
        if self.context_epochs < 1 or self.context_epochs % 2 == 0:
            raise ValueError("context_epochs must be a positive odd integer")
        self.center_index = self.context_epochs // 2
        self.demean_input = bool(demean_input)
        self.demean_target = bool(demean_target)
        self.npz_path = Path(npz_path) if npz_path is not None else None

        n_base = len(base_dataset)
        if n_base == 0:
            raise ValueError("base dataset must contain at least one example")
        first_noisy, first_target = base_dataset[0]
        if first_noisy.ndim != 3:
            raise ValueError("base dataset input must have shape (context_epochs, channels, samples)")
        if first_target.ndim != 2:
            raise ValueError("base dataset target must have shape (channels, samples)")
        if first_noisy.shape[0] != self.context_epochs:
            raise ValueError(
                f"expected {self.context_epochs} context epochs in base dataset, got {first_noisy.shape[0]}"
            )
        self.n_channels = int(first_noisy.shape[1])
        self.epoch_samples = int(first_noisy.shape[2])
        self.chunk_size = self.epoch_samples
        self.target_type = "artifact"
        self.trigger_aligned = True
        self.sfreq = float(getattr(base_dataset, "sfreq", float("nan")))

        self._clean_center: np.ndarray | None = None
        if self.npz_path is not None and self.npz_path.exists():
            with np.load(self.npz_path, allow_pickle=True) as bundle:
                if "clean_center" in bundle.files:
                    self._clean_center = bundle["clean_center"].astype(np.float32, copy=False)

        total = n_base * self.n_channels
        self._length = total if max_examples is None else max(0, min(int(max_examples), total))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)
        base_idx = int(idx) // self.n_channels
        channel_idx = int(idx) % self.n_channels

        noisy_context, artifact_target = self.base_dataset[base_idx]
        noisy = noisy_context[:, channel_idx, :].astype(np.float32, copy=True)
        artifact = artifact_target[channel_idx : channel_idx + 1, :].astype(np.float32, copy=True)

        noisy_center = noisy[self.center_index : self.center_index + 1, :].copy()
        if self._clean_center is not None:
            clean = self._clean_center[base_idx, channel_idx : channel_idx + 1, :].astype(np.float32, copy=True)
        else:
            clean = noisy_center - artifact

        if self.demean_input:
            noisy_mean = noisy.mean(axis=-1, keepdims=True)
            noisy = noisy - noisy_mean
            noisy_center = noisy_center - noisy_center.mean(axis=-1, keepdims=True)
            clean = clean - clean.mean(axis=-1, keepdims=True)
        if self.demean_target:
            artifact = artifact - artifact.mean(axis=-1, keepdims=True)

        target_stack = np.concatenate([artifact, clean, noisy_center], axis=0)
        return noisy, target_stack

    @property
    def input_shape(self) -> tuple[int, int]:
        return (self.context_epochs, self.epoch_samples)

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
    def __init__(self, parent: DHCTGanV2PaperAccurateContextDataset, indices: list[int]) -> None:
        self._parent = parent
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self._parent[self._indices[idx]]


# ---------------------------------------------------------------------------
# facet-train factories
# ---------------------------------------------------------------------------


def build_model(
    *,
    input_shape: tuple[int, int] | tuple[int, int, int] | None = None,
    epoch_samples: int | None = None,
    context_epochs: int = 7,
    base_channels: int = 16,
    depth: int = 4,
    num_heads: int = 4,
    local_blocks: int = 8,
    lgtb_depth: int = 2,
    **_: object,
) -> DHCTGanV2PaperAccurateGenerator:
    """Construct the paper-accurate DHCT-GAN v2 generator.

    ``input_shape`` may be ``(context_epochs, samples)`` (the shape exposed by
    :class:`DHCTGanV2PaperAccurateContextDataset`) or
    ``(context_epochs, 1, samples)``. Injected facet-train kwargs (n_channels,
    chunk_size, sfreq, target_type, training_config, target_shape,
    epoch_samples, ...) are accepted via ``**_``.
    """
    if input_shape is not None:
        if len(input_shape) == 2:
            in_channels = int(input_shape[0])
            samples = int(input_shape[1])
        else:
            in_channels = int(input_shape[0]) * int(input_shape[1])
            samples = int(input_shape[-1])
    else:
        in_channels = int(context_epochs)
        samples = int(epoch_samples or 512)
    return DHCTGanV2PaperAccurateGenerator(
        context_epochs=in_channels,
        out_channels=1,
        base_channels=int(base_channels),
        depth=int(depth),
        epoch_samples=samples,
        num_heads=int(num_heads),
        local_blocks=int(local_blocks),
        lgtb_depth=int(lgtb_depth),
    )


def build_loss(
    name: str | None = None,
    alpha_consistency: float = 0.5,
    lambda_feat: float = 0.1,
    lambda_adv: float = 0.1,
    beta_adv: float | None = None,
    disc_channels: int = 16,
    disc_depth: int = 8,
    disc_lr: float = 1e-4,
    recon: str = "mse",
    **_: object,
) -> DHCTGanV2PaperAccurateLoss:
    """Construct the paper-accurate LSGAN + feature-matching + MSE loss.

    ``recon`` selects the reconstruction term (``"mse"`` = paper default,
    ``"l1"`` = documented spike-robust alternative). ``beta_adv`` is accepted as
    a backwards-compatible alias for ``lambda_adv``.
    """
    return DHCTGanV2PaperAccurateLoss(
        alpha_consistency=float(alpha_consistency),
        lambda_feat=float(lambda_feat),
        lambda_adv=float(lambda_adv),
        beta_adv=beta_adv,
        disc_channels=int(disc_channels),
        disc_depth=int(disc_depth),
        disc_lr=float(disc_lr),
        recon=str(recon),
        in_channels=1,
    )


def build_dataset(
    path: str | None = None,
    context_path: str | None = None,
    context_epochs: int = 7,
    demean_input: bool = True,
    demean_target: bool = True,
    max_examples: int | None = None,
    **_: object,
) -> DHCTGanV2PaperAccurateContextDataset:
    dataset_path = Path(path or context_path or "").expanduser()
    if not str(dataset_path) or str(dataset_path) == ".":
        raise ValueError("build_dataset requires path or context_path")
    base = NPZContextArtifactDataset(
        path=dataset_path,
        input_key="noisy_context",
        target_key="artifact_center",
        demean_input=False,
        demean_target=False,
    )
    return DHCTGanV2PaperAccurateContextDataset(
        base,
        context_epochs=int(context_epochs),
        npz_path=dataset_path,
        demean_input=bool(demean_input),
        demean_target=bool(demean_target),
        max_examples=max_examples,
    )


__all__ = [
    "DHCTGanV2PaperAccurateGenerator",
    "DHCTGanV2PaperAccurateDiscriminator",
    "DHCTGanV2PaperAccurateLoss",
    "DHCTGanV2PaperAccurateContextDataset",
    "build_model",
    "build_loss",
    "build_dataset",
]
