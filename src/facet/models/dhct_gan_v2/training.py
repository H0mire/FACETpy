"""Training factories for DHCT-GAN v2 (multi-epoch context variant).

The v2 generator consumes the full 7-epoch noisy context (stacked as input
channels per scalp channel) and predicts the artifact for the center epoch.
The discriminator and loss operate on single-channel artifact signals, as in
v1.
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
        h = x.transpose(1, 2)
        b = h.shape[0]
        t = h.shape[1]
        c = h.shape[2]

        local = self.local_norm(h)
        win = self.window_size
        pad = (win - t % win) % win
        local = F.pad(local, (0, 0, 0, pad))
        padded_t = local.shape[1]
        n_win = padded_t // win
        local_in = local.reshape(b * n_win, win, c)
        local_out = self.local_attn(local_in)
        local_out = local_out.reshape(b, padded_t, c)
        local_out = local_out[:, :t, :]
        h = h + local_out

        g = self.global_norm(h)
        g_out = self.global_attn(g)
        h = h + g_out

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
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.reduce(x)
        return self.cnn(x)


# ---------------------------------------------------------------------------
# Generator (multi-epoch context input)
# ---------------------------------------------------------------------------


class DHCTGanV2Generator(nn.Module):
    """DHCT-GAN generator that consumes a multi-epoch context tensor.

    Input shape ``(B, context_epochs, T)`` — the center epoch is at index
    ``context_epochs // 2``. The 7 context epochs are stacked as channels so
    the first conv mixes them at every sample. The output is the predicted
    artifact for the center epoch only, shape ``(B, 1, T)``.

    Compared to v1, the only structural change is the stem's input channel
    count (1 -> ``context_epochs``). Everything downstream (encoder,
    transformer, dual decoders, gate) is unchanged.
    """

    def __init__(
        self,
        context_epochs: int = 7,
        out_channels: int = 1,
        base_channels: int = 16,
        depth: int = 4,
        epoch_samples: int = 512,
        num_heads: int = 4,
        window_size: int = 16,
    ) -> None:
        super().__init__()
        if context_epochs < 1:
            raise ValueError("context_epochs must be >= 1")
        self.context_epochs = int(context_epochs)
        self.out_channels = int(out_channels)
        self.base_channels = int(base_channels)
        self.depth = int(depth)
        self.epoch_samples = int(epoch_samples)
        self.center_index = self.context_epochs // 2

        self.stem = nn.Sequential(
            nn.Conv1d(self.context_epochs, self.base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(self.base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.base_channels, self.base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

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

        self.gate = nn.Sequential(
            nn.Conv1d(self.base_channels * 2, self.base_channels, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv1d(self.base_channels, self.out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"DHCTGanV2Generator expects (B, C, T), got shape {tuple(x.shape)}")
        outputs = self._compute_outputs(x)
        return outputs["artifact"]

    def _compute_outputs(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Center epoch is referenced by the fusion path so the gate can decide
        # between "clean head says X" and "noisy_center - artifact head says X".
        noisy_center = x[:, self.center_index : self.center_index + 1, :]

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

        clean_feat = F.interpolate(clean_feat, size=x.shape[-1], mode="linear", align_corners=False)
        artifact_feat = F.interpolate(artifact_feat, size=x.shape[-1], mode="linear", align_corners=False)

        clean_pred = self.clean_head(clean_feat)
        artifact_pred = self.artifact_head(artifact_feat)

        gate_in = torch.cat([clean_feat, artifact_feat], dim=1)
        gate = self.gate(gate_in)
        fused_clean = gate * clean_pred + (1.0 - gate) * (noisy_center - artifact_pred)

        return {
            "artifact": artifact_pred,
            "clean": clean_pred,
            "fused_clean": fused_clean,
            "gate": gate,
            "noisy_center": noisy_center,
        }


# ---------------------------------------------------------------------------
# Discriminator (unchanged from v1: single-channel artifact)
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
# Loss module
# ---------------------------------------------------------------------------


class DHCTGanV2Loss(nn.Module):
    """Reconstruction + consistency + adversarial loss for DHCT-GAN v2.

    Loss contract identical to v1: the loss receives ``pred`` of shape
    ``(B, 1, T)`` (artifact prediction for the center epoch) and ``target``
    of shape ``(B, 3, T)`` where the channel axis packs
    ``[artifact_target, clean_target, noisy_center]``.

    The discriminator and its private optimizer live inside this module so the
    standard ``facet-train`` single-optimizer contract keeps working: each
    forward call performs an internal discriminator step against
    ``pred.detach()``, then returns the generator's combined loss.
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
        if target.shape[1] != 3:
            raise ValueError(
                f"DHCTGanV2Loss expects target with 3 channels (artifact, clean, noisy_center), "
                f"got shape {tuple(target.shape)}"
            )

        self._ensure_device(pred)

        artifact_target = target[:, 0:1, :]
        clean_target = target[:, 1:2, :]
        noisy_center = target[:, 2:3, :]

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

        recon_artifact = self.recon(pred, artifact_target)
        fused_clean = noisy_center - pred
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
# Dataset wrapper (channel-wise 7-epoch context view)
# ---------------------------------------------------------------------------


class DHCTGanV2ContextDataset:
    """Per-channel multi-epoch context dataset for DHCT-GAN v2.

    Each item exposes ``(noisy_context, target_stack)`` where
    ``noisy_context`` has shape ``(context_epochs, samples)`` and
    ``target_stack`` has shape ``(3, samples)`` packing
    ``[artifact_target, clean_target, noisy_center]``. The model sees the full
    context; the loss only needs the center-epoch reconstructions.

    Wraps :class:`NPZContextArtifactDataset` and exposes the per-channel,
    per-example view used elsewhere (e.g. ``cascaded_context_dae``).
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

        # Pre-cache the clean_center array so we can build (artifact, clean,
        # noisy_center) target stacks without a second pass over the base
        # dataset. NPZContextArtifactDataset only carries the artifact target,
        # not the clean signal — load it directly from the NPZ when available.
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
        # noisy_context: (context_epochs, channels, samples) -> (context_epochs, samples)
        noisy = noisy_context[:, channel_idx, :].astype(np.float32, copy=True)
        artifact = artifact_target[channel_idx : channel_idx + 1, :].astype(np.float32, copy=True)

        noisy_center = noisy[self.center_index : self.center_index + 1, :].copy()
        if self._clean_center is not None:
            clean = self._clean_center[base_idx, channel_idx : channel_idx + 1, :].astype(np.float32, copy=True)
        else:
            # Fallback: derive clean as noisy_center - artifact (algebraic identity given the dataset
            # builder's contract). Same baseline so demeaning is consistent.
            clean = noisy_center - artifact

        if self.demean_input:
            noisy_mean = noisy.mean(axis=-1, keepdims=True)
            noisy = noisy - noisy_mean
            # Demean the center reference using the same per-epoch mean as the
            # rest of the context so the consistency arithmetic stays
            # self-consistent.
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
    def __init__(self, parent: DHCTGanV2ContextDataset, indices: list[int]) -> None:
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
    window_size: int = 16,
    **_: object,
) -> DHCTGanV2Generator:
    """Construct the DHCT-GAN v2 generator.

    ``input_shape`` may be ``(context_epochs, samples)`` (the shape exposed by
    :class:`DHCTGanV2ContextDataset`) or ``(context_epochs, 1, samples)``.
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
    return DHCTGanV2Generator(
        context_epochs=in_channels,
        out_channels=1,
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
) -> DHCTGanV2Loss:
    return DHCTGanV2Loss(
        alpha_consistency=float(alpha_consistency),
        beta_adv=float(beta_adv),
        disc_channels=int(disc_channels),
        disc_depth=int(disc_depth),
        disc_lr=float(disc_lr),
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
) -> DHCTGanV2ContextDataset:
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
    return DHCTGanV2ContextDataset(
        base,
        context_epochs=int(context_epochs),
        npz_path=dataset_path,
        demean_input=bool(demean_input),
        demean_target=bool(demean_target),
        max_examples=max_examples,
    )


__all__ = [
    "DHCTGanV2Generator",
    "PatchGANDiscriminator",
    "DHCTGanV2Loss",
    "DHCTGanV2ContextDataset",
    "build_model",
    "build_loss",
    "build_dataset",
]
