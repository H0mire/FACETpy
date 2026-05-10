"""Training factories for DHCT-GAN on the Niazy proof-fit dataset."""

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
    """Two-layer 1D conv block with BatchNorm + LeakyReLU."""

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


class LocalGlobalTransformerBlock(nn.Module):
    """Combined local-window + global self-attention transformer block."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        window_size: int = 16,
        ff_mult: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.window_size = int(window_size)
        self.local_norm = nn.LayerNorm(channels)
        self.local_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.global_norm = nn.LayerNorm(channels)
        self.global_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff_norm = nn.LayerNorm(channels)
        self.feedforward = nn.Sequential(
            nn.Linear(channels, channels * ff_mult),
            nn.GELU(),
            nn.Linear(channels * ff_mult, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, T, C)
        b, c, t = x.shape
        h = x.transpose(1, 2)

        # Local windowed attention
        local = self.local_norm(h)
        win = max(1, min(self.window_size, t))
        pad = (win - t % win) % win
        if pad > 0:
            local = F.pad(local, (0, 0, 0, pad))
        n_win = local.shape[1] // win
        local_in = local.reshape(b * n_win, win, c)
        local_out, _ = self.local_attn(local_in, local_in, local_in, need_weights=False)
        local_out = local_out.reshape(b, n_win * win, c)
        if pad > 0:
            local_out = local_out[:, :t, :]
        h = h + local_out

        # Global attention
        g = self.global_norm(h)
        g_out, _ = self.global_attn(g, g, g, need_weights=False)
        h = h + g_out

        # Feedforward
        f = self.ff_norm(h)
        h = h + self.feedforward(f)

        return h.transpose(1, 2)


class EncoderStage(nn.Module):
    """One encoder stage: CNNBlock -> LGTB -> downsample-by-2."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        num_heads: int = 4,
        window_size: int = 16,
    ) -> None:
        super().__init__()
        self.cnn = CNNBlock(in_ch, out_ch)
        self.transformer = LocalGlobalTransformerBlock(
            channels=out_ch,
            num_heads=num_heads,
            window_size=window_size,
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
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.reduce(x)
        return self.cnn(x)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class DHCTGanGenerator(nn.Module):
    """Dual-branch hybrid CNN-Transformer generator from DHCT-GAN.

    Encodes the noisy input with stacked CNN + local/global transformer stages,
    then runs two parallel decoder branches. One predicts the clean signal,
    the other predicts the artifact. A learned gate fuses them sample-wise.
    The model returns the artifact prediction; the clean fusion is exposed
    via ``clean_branch_output`` for the loss module.

    Parameters
    ----------
    in_channels : int
        Number of input channels (always 1 for per-channel training).
    base_channels : int
        Starting channel width; doubled per encoder stage.
    depth : int
        Number of encoder stages.
    epoch_samples : int
        Expected input length (for documentation only; the model is fully convolutional).
    num_heads : int
        Number of attention heads in the transformer block.
    window_size : int
        Local-attention window size in samples (at the stage's temporal resolution).
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        depth: int = 4,
        epoch_samples: int = 512,
        num_heads: int = 4,
        window_size: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.depth = int(depth)
        self.epoch_samples = int(epoch_samples)

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(self.in_channels, self.base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(self.base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.base_channels, self.base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Encoder stages
        channels = [self.base_channels * (2 ** i) for i in range(self.depth)]
        encoder_in = [self.base_channels] + channels[:-1]
        self.encoder_stages = nn.ModuleList(
            EncoderStage(
                in_ch=encoder_in[i],
                out_ch=channels[i],
                num_heads=num_heads,
                window_size=max(2, window_size // (2 ** i)),
            )
            for i in range(self.depth)
        )

        # Bottleneck
        self.bottleneck = CNNBlock(channels[-1], channels[-1])

        # Two symmetric decoder branches (clean + artifact)
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

        # Gating network
        self.gate = nn.Sequential(
            nn.Conv1d(self.base_channels * 2, self.base_channels, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv1d(self.base_channels, self.in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"DHCTGanGenerator expects (B, C, T), got shape {tuple(x.shape)}")
        outputs = self._compute_outputs(x)
        return outputs["artifact"]

    def _compute_outputs(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.stem(x)

        skips: list[torch.Tensor] = []
        for stage in self.encoder_stages:
            feat, skip = stage(feat)
            skips.append(skip)

        feat = self.bottleneck(feat)

        clean_feat = feat
        artifact_feat = feat
        for i, (clean_dec, artifact_dec) in enumerate(zip(self.clean_decoder, self.artifact_decoder, strict=True)):
            skip = skips[-(i + 1)]
            clean_feat = clean_dec(clean_feat, skip)
            artifact_feat = artifact_dec(artifact_feat, skip)

        # Match output length to input length if needed (rounding)
        if clean_feat.shape[-1] != x.shape[-1]:
            clean_feat = F.interpolate(clean_feat, size=x.shape[-1], mode="linear", align_corners=False)
            artifact_feat = F.interpolate(artifact_feat, size=x.shape[-1], mode="linear", align_corners=False)

        clean_pred = self.clean_head(clean_feat)
        artifact_pred = self.artifact_head(artifact_feat)

        gate_in = torch.cat([clean_feat, artifact_feat], dim=1)
        gate = self.gate(gate_in)
        fused_clean = gate * clean_pred + (1.0 - gate) * (x - artifact_pred)

        return {
            "artifact": artifact_pred,
            "clean": clean_pred,
            "fused_clean": fused_clean,
            "gate": gate,
        }


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------


class PatchGANDiscriminator(nn.Module):
    """1D PatchGAN-style discriminator for artifact signals."""

    def __init__(self, in_channels: int = 1, base_channels: int = 16, depth: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_channels
        for _ in range(depth - 1):
            next_ch = min(ch * 2, 256)
            layers += [
                nn.Conv1d(ch, next_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(next_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = next_ch
        layers.append(nn.Conv1d(ch, 1, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Loss module (carries the discriminator + its private optimizer)
# ---------------------------------------------------------------------------


class DHCTGanLoss(nn.Module):
    """Reconstruction + consistency + adversarial loss for DHCT-GAN.

    The discriminator and its optimizer live inside this module rather than in
    the generator. The standard ``facet-train`` PyTorch wrapper builds a single
    optimizer over ``model.parameters()`` (= generator only). On each
    ``loss_fn(pred, target)`` call we run an internal discriminator step
    (using ``pred.detach()``) before computing the generator's adversarial
    contribution. This preserves alternating optimization with a single CLI
    optimizer.

    Parameters
    ----------
    alpha_consistency : float
        Weight of the consistency loss ``L1(fused_clean, clean_target)``.
    beta_adv : float
        Weight of the generator's adversarial loss.
    disc_channels : int
        Base channel width of the PatchGAN discriminator.
    disc_depth : int
        Number of strided discriminator layers.
    disc_lr : float
        Learning rate of the discriminator's private Adam optimizer.
    in_channels : int
        Channels per window (always 1 for per-channel training).
    """

    def __init__(
        self,
        alpha_consistency: float = 0.5,
        beta_adv: float = 0.1,
        disc_channels: int = 16,
        disc_depth: int = 4,
        disc_lr: float = 1e-4,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.alpha_consistency = float(alpha_consistency)
        self.beta_adv = float(beta_adv)
        self.disc_lr = float(disc_lr)
        self.discriminator = PatchGANDiscriminator(
            in_channels=in_channels,
            base_channels=int(disc_channels),
            depth=int(disc_depth),
        )
        self.recon = nn.L1Loss()
        self._disc_optimizer: torch.optim.Optimizer | None = None
        self._initialized_device: torch.device | None = None

    def _ensure_device(self, reference: torch.Tensor) -> None:
        device = reference.device
        if self._initialized_device == device and self._disc_optimizer is not None:
            return
        self.discriminator.to(device)
        self._disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.disc_lr,
            betas=(0.9, 0.999),
        )
        self._initialized_device = device

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the generator loss.

        ``pred`` is the artifact prediction (shape ``(B, 1, T)``).
        ``target`` packs ``[artifact_target, clean_target, noisy_input]``
        along the channel axis (shape ``(B, 3, T)``).
        """
        if target.shape[1] != 3:
            raise ValueError(
                f"DHCTGanLoss expects target with 3 channels (artifact, clean, noisy), "
                f"got shape {tuple(target.shape)}"
            )

        self._ensure_device(pred)

        artifact_target = target[:, 0:1, :]
        clean_target = target[:, 1:2, :]
        noisy_input = target[:, 2:3, :]

        # ---- Discriminator update (only when gradients are enabled) ----
        if torch.is_grad_enabled() and self._disc_optimizer is not None:
            self._disc_optimizer.zero_grad(set_to_none=True)
            d_real = self.discriminator(artifact_target)
            d_fake = self.discriminator(pred.detach())
            d_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
                + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            )
            d_loss.backward()
            self._disc_optimizer.step()

        # ---- Generator loss ----
        recon_artifact = self.recon(pred, artifact_target)
        fused_clean = noisy_input - pred
        recon_consistency = self.recon(fused_clean, clean_target)
        generator_loss = recon_artifact + self.alpha_consistency * recon_consistency

        if self.beta_adv > 0.0:
            d_for_gen = self.discriminator(pred)
            adv_loss = F.binary_cross_entropy_with_logits(
                d_for_gen, torch.ones_like(d_for_gen)
            )
            generator_loss = generator_loss + self.beta_adv * adv_loss

        return generator_loss


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------


class DHCTGanArtifactDataset:
    """Per-channel single-epoch dataset for DHCT-GAN.

    Each item exposes ``(noisy_window, target_stack)`` where ``noisy_window``
    has shape ``(1, samples)`` and ``target_stack`` has shape
    ``(3, samples)`` packing the artifact target, clean target, and the noisy
    input itself. The loss module unpacks the channels.
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

        for name, arr in (("noisy_center", self._noisy), ("clean_center", self._clean), ("artifact_center", self._artifact)):
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

    def train_val_split(self, val_ratio: float = 0.2, seed: int = 42) -> tuple["_Subset", "_Subset"]:
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

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self._parent[self._indices[idx]]


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
    window_size: int = 16,
    **_: object,
) -> DHCTGanGenerator:
    """Construct the DHCT-GAN generator. Returns only the generator nn.Module
    so the standard ``PyTorchModelWrapper`` optimizes generator parameters.
    """
    if input_shape is not None:
        in_channels = int(input_shape[0])
        samples = int(input_shape[-1])
    else:
        in_channels = 1
        samples = int(epoch_samples or 512)
    return DHCTGanGenerator(
        in_channels=in_channels,
        base_channels=int(base_channels),
        depth=int(depth),
        epoch_samples=samples,
        num_heads=int(num_heads),
        window_size=int(window_size),
    )


def build_loss(
    alpha_consistency: float = 0.5,
    beta_adv: float = 0.1,
    disc_channels: int = 16,
    disc_depth: int = 4,
    disc_lr: float = 1e-4,
    **_: object,
) -> DHCTGanLoss:
    return DHCTGanLoss(
        alpha_consistency=float(alpha_consistency),
        beta_adv=float(beta_adv),
        disc_channels=int(disc_channels),
        disc_depth=int(disc_depth),
        disc_lr=float(disc_lr),
        in_channels=1,
    )


def build_dataset(
    path: str | None = None,
    *,
    demean: bool = True,
    max_examples: int | None = None,
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
    "DHCTGanGenerator",
    "PatchGANDiscriminator",
    "DHCTGanLoss",
    "DHCTGanArtifactDataset",
    "build_model",
    "build_loss",
    "build_dataset",
]
