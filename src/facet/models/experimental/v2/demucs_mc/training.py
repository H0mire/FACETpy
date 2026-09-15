"""Multichannel Demucs for the Weg-A spatio-temporal contract (run_6 Phase B).

Why this model exists
---------------------
Run 6 phases C-E established that the limit is the *error floor*, not the loss:
the 248k-parameter baseline leaves a ~44 µV residual while a typical injected IED
peaks at ~23 µV, so the spike drowns in the model's own error and every
spike-local metric collapses (``docs/research/run_6_results.md``). Reweighting the
loss moved morphology correlation from -0.018 to +0.167 and no further. Only a
model that reconstructs the artifact more accurately can change that, which is
what phase B asks for.

Input format — the detail that decides whether this works
---------------------------------------------------------
The single-channel Demucs that scored +31.30 dB in run 1 does **not** see an
epoch at a time. Its dataset hands it ``(1, context_epochs * epoch_samples)``:
the seven trigger-defined epochs of one channel **concatenated into one
waveform**. This model keeps that and adds the neighbours as conv channels, so a
batch item is ``(B, 1 + k_neighbors, context_epochs * core)``.

That matters twice over. The epoch-to-epoch repetition of the artifact becomes
plain temporal structure that a wide receptive field can exploit, and the window
is long enough that ``stride**depth`` downsampling still leaves a usable
bottleneck — 3584 samples over 256x is 14 steps, where a single 512-sample epoch
would leave 2.

An earlier version of this file treated the ``context_epochs x channels`` grid as
21 independent units of 512 samples each and attended across them. It reached
4.3 dB against the baseline's 33.0 dB, and the failure was initially written up
as the U-Net being the wrong inductive bias for a high-frequency target. That
conclusion was wrong — run 1's result already contradicted it. The architecture
was fine; it was being fed the wrong shape. Cross-channel attention therefore now
runs over the ``1 + k_neighbors`` electrodes, as the phase-B sketch intended.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from facet.training.dataset import NPZSpatioTemporalDataset
from facet.training.weg_a_baseline import RecoveredCleanLoss, SpikeWeightedMSELoss

DEFAULT_DATASET = "./output/weg_a_farm_v5_512/weg_a_spatiotemporal_dataset.npz"


def _glu_channels(channels: int) -> int:
    return 2 * channels


class _EncoderBlock(torch.nn.Module):
    """Strided conv + GLU, as in Demucs."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        padding = (kernel_size - stride) // 2
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.act = torch.nn.GELU()
        self.conv_glu = torch.nn.Conv1d(out_channels, _glu_channels(out_channels), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv(x))
        return torch.nn.functional.glu(self.conv_glu(h), dim=1)


class _DecoderBlock(torch.nn.Module):
    """GLU + transposed conv, mirroring :class:`_EncoderBlock`."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, last: bool) -> None:
        super().__init__()
        self.conv_glu = torch.nn.Conv1d(in_channels, _glu_channels(in_channels), kernel_size=3, padding=1)
        padding = (kernel_size - stride) // 2
        self.deconv = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.act = None if last else torch.nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.glu(self.conv_glu(x), dim=1)
        h = self.deconv(h)
        return h if self.act is None else self.act(h)


class CrossUnitAttention(torch.nn.Module):
    """Multi-head attention across the electrodes at each time step.

    Channels are treated as a set, not a sequence: no positional encoding is
    added, so the layer is **equivariant** to permuting them. Which channel is the
    target is decided downstream by the head, not by attention order — that keeps
    the layer from silently learning a fixed slot ordering.
    """

    def __init__(self, features: int, n_heads: int = 4) -> None:
        super().__init__()
        n_heads = max(1, min(n_heads, features))
        while features % n_heads != 0:
            n_heads -= 1
        self.norm = torch.nn.LayerNorm(features)
        self.attn = torch.nn.MultiheadAttention(features, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T) -> attend over C independently per time step
        b, u, f, t = x.shape
        h = x.permute(0, 3, 1, 2).reshape(b * t, u, f)   # (B*T, U, F)
        h_norm = self.norm(h)
        attended, _ = self.attn(h_norm, h_norm, h_norm, need_weights=False)
        h = h + attended                                  # residual
        return h.reshape(b, t, u, f).permute(0, 2, 3, 1)  # back to (B, U, F, T)


class MultichannelDemucs(torch.nn.Module):
    """Shared-weight Demucs U-Net with a cross-channel attention bridge.

    Input ``(B, context_epochs, channels, T)``; the epochs are concatenated in
    time so each channel is one ``context_epochs * T`` waveform. Output
    ``(B, 1, T)``: the artifact of the centre epoch of the target channel
    (channel 0 by the builder's ordering — self first, then montage neighbours).
    """

    def __init__(
        self,
        context_epochs: int,
        n_channels: int,
        depth: int = 4,
        initial_channels: int = 32,
        kernel_size: int = 8,
        stride: int = 4,
        lstm_layers: int = 2,
        n_heads: int = 4,
        attention_levels: int | None = None,
        rescale: float = 0.1,
    ) -> None:
        super().__init__()
        self.context_epochs = context_epochs
        self.n_channels = n_channels
        self.n_units = context_epochs * n_channels
        self.depth = depth
        self.stride = stride

        # Attention only on the deeper (cheaper, more abstract) levels by default:
        # at full resolution the (B*T, U, F) attention is the dominant cost.
        self.attention_levels = depth if attention_levels is None else attention_levels

        encoder, decoder, attns = [], [], []
        in_ch, chans = 1, []
        for level in range(depth):
            out_ch = initial_channels * (2**level)
            encoder.append(_EncoderBlock(in_ch, out_ch, kernel_size, stride))
            deep_enough = level >= depth - self.attention_levels
            attns.append(CrossUnitAttention(out_ch, n_heads) if deep_enough else torch.nn.Identity())
            chans.append(out_ch)
            in_ch = out_ch
        for level in reversed(range(depth)):
            out_ch = 1 if level == 0 else initial_channels * (2 ** (level - 1))
            decoder.append(_DecoderBlock(chans[level], out_ch, kernel_size, stride, last=(level == 0)))

        self.encoder = torch.nn.ModuleList(encoder)
        self.channel_attn = torch.nn.ModuleList(attns)
        self.decoder = torch.nn.ModuleList(decoder)

        bottleneck = chans[-1]
        self.bilstm = torch.nn.LSTM(
            bottleneck, bottleneck, num_layers=lstm_layers, bidirectional=True, batch_first=True
        )
        self.lstm_proj = torch.nn.Linear(2 * bottleneck, bottleneck)
        self.head = torch.nn.Conv1d(n_channels, 1, kernel_size=1)
        self._rescale_init_weights(rescale)

    def _rescale_init_weights(self, target_std: float) -> None:
        """Demucs' weight rescaling — omitting it is not a detail.

        A deep GLU encoder/decoder is badly conditioned under default init: the
        first version of this model without rescaling reached only 4.3 dB where
        the 248k baseline reaches 33.0 dB, which looked like an architecture
        failure and was an initialisation failure.
        """
        if target_std <= 0:
            return
        for module in self.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                with torch.no_grad():
                    std = module.weight.std().clamp(min=1e-12).item()
                    module.weight.div_((std / target_std) ** 0.5)
                    if module.bias is not None:
                        module.bias.zero_()

    @staticmethod
    def _fit(x: torch.Tensor, length: int) -> torch.Tensor:
        """Crop or edge-pad the time axis to ``length`` (stride rounding)."""
        if x.shape[-1] == length:
            return x
        if x.shape[-1] > length:
            return x[..., :length]
        return torch.nn.functional.pad(x, (0, length - x.shape[-1]), mode="replicate")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected (batch, context_epochs, channels, samples), got {tuple(x.shape)}")
        b, ep, ch, t = x.shape
        if ep != self.context_epochs or ch != self.n_channels:
            raise ValueError(
                f"Model built for {self.context_epochs} epochs x {self.n_channels} channels, got {ep} x {ch}"
            )
        # Concatenate the epochs **in time**, one long waveform per channel. This is
        # the format the single-channel Demucs was trained on, and it is what makes
        # the U-Net work here: the epoch-to-epoch repetition of the artifact becomes
        # ordinary temporal structure a wide receptive field can see, and the window
        # is long enough (7 x 512 = 3584) that `stride**depth` downsampling still
        # leaves a usable bottleneck.
        full = t * ep
        seq_in = x.permute(0, 2, 1, 3).reshape(b, ch, full)     # (B, C, epochs*T)

        span = self.stride**self.depth
        t_pad = int(math.ceil(full / span) * span)
        h = self._fit(seq_in, t_pad).unsqueeze(2)               # (B, C, 1, T)

        skips: list[torch.Tensor] = []
        for enc, attn in zip(self.encoder, self.channel_attn, strict=True):
            flat = h.reshape(b * ch, h.shape[2], h.shape[3])
            enc_out = enc(flat)
            h = enc_out.reshape(b, ch, enc_out.shape[1], enc_out.shape[2])
            h = attn(h)                                          # across channels
            skips.append(h)

        # BiLSTM over time at the bottleneck, weights shared across channels.
        bc, f, tb = b * ch, h.shape[2], h.shape[3]
        seq = h.reshape(bc, f, tb).transpose(1, 2)               # (B*C, T, F)
        seq, _ = self.bilstm(seq)
        h = self.lstm_proj(seq).transpose(1, 2).reshape(b, ch, f, tb)

        for i, dec in enumerate(self.decoder):
            level = self.depth - 1 - i
            h = h + self._fit(skips[level], h.shape[-1])
            flat = h.reshape(b * ch, h.shape[2], h.shape[3])
            dec_out = dec(flat)
            h = dec_out.reshape(b, ch, dec_out.shape[1], dec_out.shape[2])

        # (B, C, 1, T) -> the centre epoch of the reconstructed waveform, then mix
        # the channels down to the target channel's artifact.
        out = self._fit(h.squeeze(2), full)                      # (B, C, epochs*T)
        centre = out[..., (ep // 2) * t : (ep // 2 + 1) * t]     # (B, C, T)
        return self.head(centre)


def build_model(
    input_shape: tuple[int, int, int],
    depth: int = 4,
    initial_channels: int = 32,
    kernel_size: int = 8,
    stride: int = 4,
    lstm_layers: int = 2,
    n_heads: int = 4,
    attention_levels: int | None = None,
    rescale: float = 0.1,
    **_: object,
) -> MultichannelDemucs:
    context_epochs, n_channels, _samples = input_shape
    return MultichannelDemucs(
        context_epochs=context_epochs,
        n_channels=n_channels,
        depth=depth,
        initial_channels=initial_channels,
        kernel_size=kernel_size,
        stride=stride,
        lstm_layers=lstm_layers,
        n_heads=n_heads,
        attention_levels=attention_levels,
        rescale=rescale,
    )


def build_dataset(
    path: str = DEFAULT_DATASET,
    max_examples: int | None = None,
    max_shift: int | None = None,
    fractional_shift: bool = False,
    background_mix_prob: float = 0.0,
    demean_input: bool = False,
    demean_target: bool = False,
    target_key: str = "artifact_center",
    target_extras: tuple[str, ...] = (),
    residual_mode: bool = False,
    seed: int = 0,
    **_: object,
) -> NPZSpatioTemporalDataset:
    """Same Weg-A dataset the baseline uses, so the comparison stays apples-to-apples."""
    return NPZSpatioTemporalDataset(
        path=Path(path).expanduser(),
        target_key=target_key,
        max_examples=max_examples,
        max_shift=max_shift,
        fractional_shift=fractional_shift,
        background_mix_prob=background_mix_prob,
        demean_input=demean_input,
        demean_target=demean_target,
        target_extras=tuple(target_extras),
        residual_mode=residual_mode,
        seed=seed,
    )


def build_loss(name: str = "mse", spike_weight: float = 20.0, mse_weight: float = 10.0, **_: object):
    normalized = name.strip().lower()
    if normalized in {"l1", "mae"}:
        return torch.nn.L1Loss()
    if normalized in {"smooth_l1", "huber"}:
        return torch.nn.SmoothL1Loss()
    if normalized == "mse":
        return torch.nn.MSELoss()
    if normalized in {"spike_mse", "spike_weighted_mse"}:
        return SpikeWeightedMSELoss(spike_weight=spike_weight)
    if normalized in {"recovered_clean", "si_sdr_clean"}:
        return RecoveredCleanLoss(mse_weight=mse_weight, spike_weight=spike_weight)
    raise ValueError(
        f"Unsupported loss '{name}'. Use one of: mse, l1, smooth_l1, spike_mse, recovered_clean."
    )
