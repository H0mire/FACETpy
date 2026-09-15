"""Paper-faithful training factories for the IC-U-Net edition.

This module re-implements the IC-U-Net of Chuang et al. 2022 (NeuroImage
263:119586; arXiv 2111.10026; repo ``roseDwayane/AIEEG``) more faithfully than
the original ``facet.models.ic_unet`` package, while staying compatible with the
FACETpy facet-train factory contract and runnable on CPU.

Key paper-accurate changes vs. the original edition
----------------------------------------------------
1. **Sensor-level network, no in-graph ICA** (Sec 2.3, Conclusion). The paper
   uses ICA + ICLabel ONLY to synthesise training pairs; the runtime network is
   a plain channel-space U-Net that takes raw EEG channels in and outputs the
   reconstructed (clean) EEG channels. We drop the frozen ``W``/``W_pinv``
   buffers from the default model. An opt-in ``use_frozen_ica`` flag is kept as
   a documented *non-paper* extension.
2. **ReLU CBR blocks** (Sec 2.1: "Convolution, Batch normalization, and ReLU").
   The original used LeakyReLU(0.1); we default to ReLU, with LeakyReLU opt-in.
3. **Transposed-convolution (deconvolution) decoder** (Sec 2.1). The original
   used parameter-free ``nn.Upsample``; we use a learned ``nn.ConvTranspose1d``.
4. **Normalised, equal-weight ensemble loss** (Eq. 2): equal weights
   ``alpha = [1, 1, 1, 1]`` divided by ``sum(alpha)``.
5. **Paper-accurate frequency term** (Eq. 4): ``L_freq`` is the MSE of the
   per-channel z-scored power spectral density ``|rfft|**2`` restricted to the
   1-50 Hz band (band hi clamped below Nyquist for arbitrary sfreq).
6. **Per-time-series z-score normalisation** before training (Sec 3.1/4.1),
   replacing the original demean-only convention. Demean-only stays opt-in.
7. **Clean-reconstruction target by default** (Eqs 1-3): the DDAE predicts the
   clean center epoch (``target_type='clean'``). The FACETpy ``artifact`` head
   is kept as a compatible option for like-for-like comparison.

Where a paper technique does not transfer to single-/few-channel EEG-fMRI
gradient-artifact removal (in-graph ICA, ICLabel mixB/mixBnB synthesis), it is
dropped and documented in ``documentation/paper_accuracy_review.md`` rather than
copied blindly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from facet.training.dataset import NPZContextArtifactDataset

# ---------------------------------------------------------------------------
# CBR building blocks (Conv1d -> BatchNorm1d -> ReLU, paper Sec 2.1)
# ---------------------------------------------------------------------------


def _make_activation(activation: str, negative_slope: float = 0.1) -> nn.Module:
    """Resolve the CBR activation. Paper default is ReLU (Sec 2.1)."""
    name = str(activation).strip().lower()
    if name in {"relu"}:
        return nn.ReLU(inplace=True)
    if name in {"leaky_relu", "leakyrelu", "lrelu"}:
        return nn.LeakyReLU(negative_slope, inplace=True)
    raise ValueError(f"Unknown activation '{activation}' (expected 'relu' or 'leaky_relu')")


class _CBRDoubleConv1d(nn.Module):
    """Two stacked CBR blocks: (Conv1d -> BatchNorm1d -> activation) x2.

    Matches the paper's CBR block (Sec 2.1), defaulting to ReLU. ``same`` padding
    keeps the temporal length unchanged so the encoder downsampling is the only
    length-changing operation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: str = "relu",
        negative_slope: float = 0.1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            _make_activation(activation, negative_slope),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            _make_activation(activation, negative_slope),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    """Max-pool downsample then a CBR double-conv. Filter count DOUBLES here."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: str = "relu",
        negative_slope: float = 0.1,
    ) -> None:
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv = _CBRDoubleConv1d(in_channels, out_channels, kernel_size, activation, negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _Up(nn.Module):
    """Learned transposed-convolution upsample + concatenative skip + CBR conv.

    The paper (Sec 2.1) decodes with a 1-D *transposed convolution*
    (deconvolution), not interpolation. Filter count HALVES across this block.
    A pad/crop guard tolerates odd context lengths.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: str = "relu",
        negative_slope: float = 0.1,
    ) -> None:
        super().__init__()
        # ConvTranspose1d with stride 2 doubles the temporal length (learned
        # deconvolution). It also reduces the feature dimension to out_channels.
        self.up = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = _CBRDoubleConv1d(out_channels + skip_channels, out_channels, kernel_size, activation, negative_slope)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Length guard for odd context lengths: crop or pad to the skip length.
        if x.shape[-1] > skip.shape[-1]:
            x = x[..., : skip.shape[-1]]
        elif x.shape[-1] < skip.shape[-1]:
            x = nn.functional.pad(x, (0, skip.shape[-1] - x.shape[-1]))
        return self.conv(torch.cat([skip, x], dim=1))


# ---------------------------------------------------------------------------
# Paper-faithful sensor-level U-Net core
# ---------------------------------------------------------------------------


class IcUNetCore(nn.Module):
    """Sensor-level 1-D U-Net with the IC-U-Net filter ladder.

    The network consumes raw EEG channels and reconstructs EEG channels in the
    same channel space (the paper's "no limitations on channel numbers"). The
    encoder doubles the filter count after each downsample; the decoder halves
    it after each transposed-conv upsample; encoder features are concatenated to
    the decoder via skip connections.

    The exact Fig 2A kernel sizes could not be transcribed from the PDF; the
    doubling/halving ladder is paper-consistent and ``depth``/``base_channels``/
    ``kernel_size`` are exposed as kwargs (default 4-level, 64-base, matching the
    documented ``roseDwayane/AIEEG`` configuration).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        base_channels: int = 64,
        depth: int = 4,
        kernel_size: int = 7,
        activation: str = "relu",
        negative_slope: float = 0.1,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels if out_channels is not None else in_channels)
        self.depth = int(depth)
        self.kernel_size = int(kernel_size)

        # Filter ladder: base, 2*base, 4*base, ... DOUBLING after each downsample.
        channels = [base_channels * (2**level) for level in range(depth)]

        self.inc = _CBRDoubleConv1d(self.in_channels, channels[0], kernel_size, activation, negative_slope)

        self.downs = nn.ModuleList()
        for level in range(depth - 1):
            self.downs.append(_Down(channels[level], channels[level + 1], kernel_size, activation, negative_slope))

        self.ups = nn.ModuleList()
        for level in range(depth - 1, 0, -1):
            # Upsample from channels[level] -> channels[level - 1], concatenating
            # the encoder skip of channels[level - 1]. Filters HALVE per step.
            self.ups.append(
                _Up(
                    in_channels=channels[level],
                    skip_channels=channels[level - 1],
                    out_channels=channels[level - 1],
                    kernel_size=kernel_size,
                    activation=activation,
                    negative_slope=negative_slope,
                )
            )

        self.outc = nn.Conv1d(channels[0], self.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        h = self.inc(x)
        skips.append(h)
        for down in self.downs:
            h = down(h)
            skips.append(h)
        # skips: [enc0, enc1, ..., bottleneck]; decode from the bottleneck up.
        h = skips[-1]
        for level, up in enumerate(self.ups):
            skip = skips[-(level + 2)]
            h = up(h, skip)
        return self.outc(h)


# ---------------------------------------------------------------------------
# Edition wrapper: sensor-level reconstruction + center extraction
# ---------------------------------------------------------------------------


class IcUNetPaperAccurate(nn.Module):
    """Paper-faithful IC-U-Net for the Niazy proof-fit context.

    Pipeline (sensor space, NO in-graph ICA by default):

    1. Input ``(B, n_channels, context_epochs * epoch_samples)`` (flattened
       context, already produced by the dataset).
    2. Optional per-channel z-score / demean is handled in the dataset, so the
       model receives a normalised signal and only reconstructs it.
    3. :class:`IcUNetCore` reconstructs the full context in channel space.
    4. Extract the center epoch.
    5. Return the center epoch either as the predicted CLEAN signal
       (``output_type='clean'``, paper-faithful default) or as the predicted
       ARTIFACT ``noisy_center - clean_center`` (``output_type='artifact'``,
       FACETpy-compatible option).

    The optional ``use_frozen_ica`` flag restores the original frozen ICA
    sandwich (``W`` then U-Net then ``W_pinv``) as a documented non-paper
    extension; it is OFF by default.
    """

    def __init__(
        self,
        n_channels: int,
        context_epochs: int,
        epoch_samples: int,
        *,
        base_channels: int = 64,
        depth: int = 4,
        kernel_size: int = 7,
        activation: str = "relu",
        negative_slope: float = 0.1,
        output_type: str = "clean",
        use_frozen_ica: bool = False,
        ica_init: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        self.context_epochs = int(context_epochs)
        self.epoch_samples = int(epoch_samples)
        self.full_samples = self.context_epochs * self.epoch_samples
        self.center_index = self.context_epochs // 2
        self.center_start = self.center_index * self.epoch_samples
        self.center_stop = self.center_start + self.epoch_samples

        normalized_output = str(output_type).strip().lower()
        if normalized_output not in {"clean", "artifact"}:
            raise ValueError(f"output_type must be 'clean' or 'artifact', got '{output_type}'")
        self.output_type = normalized_output
        self.use_frozen_ica = bool(use_frozen_ica)

        self.unet = IcUNetCore(
            in_channels=self.n_channels,
            out_channels=self.n_channels,
            base_channels=int(base_channels),
            depth=int(depth),
            kernel_size=int(kernel_size),
            activation=activation,
            negative_slope=negative_slope,
        )

        # Non-paper extension: optional frozen ICA sandwich. OFF by default.
        if self.use_frozen_ica:
            if ica_init is None:
                ica_init = np.eye(self.n_channels, dtype=np.float32)
            else:
                ica_init = np.asarray(ica_init, dtype=np.float32)
                if ica_init.shape != (self.n_channels, self.n_channels):
                    raise ValueError(
                        f"ica_init must have shape ({self.n_channels}, {self.n_channels}), got {ica_init.shape}"
                    )
            ica_inv = np.linalg.pinv(ica_init).astype(np.float32)
            self.register_buffer("ica_W", torch.from_numpy(ica_init))
            self.register_buffer("ica_W_pinv", torch.from_numpy(ica_inv))

    def _apply_ica(self, x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ij,bjt->bit", matrix, x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_frozen_ica:
            ic = self._apply_ica(x, self.ica_W)
            reconstructed_ic = self.unet(ic)
            reconstructed_full = self._apply_ica(reconstructed_ic, self.ica_W_pinv)
        else:
            reconstructed_full = self.unet(x)

        clean_center = reconstructed_full[..., self.center_start : self.center_stop]
        if self.output_type == "clean":
            return clean_center
        # artifact head: predicted artifact = noisy_center - reconstructed_clean_center
        noisy_center = x[..., self.center_start : self.center_stop]
        return noisy_center - clean_center


# ---------------------------------------------------------------------------
# Paper-faithful ensemble loss (Chuang et al. 2022, Eqs 2-4)
# ---------------------------------------------------------------------------


class IcUNetEnsembleLoss(nn.Module):
    """Normalised four-term ensemble loss matching Eqs 2-4.

    ``L_ens = (1 / sum(alpha)) * (a1*L_amp + a2*L_vel + a3*L_acc + a4*L_freq)``

    with the paper's best configuration ``alpha = [1, 1, 1, 1]`` (Eq. 2,
    Fig 4A, Sec 4.2).

    - ``L_amp`` = MSE over channels x time.
    - ``L_vel`` = MSE of first-order temporal differences.
    - ``L_acc`` = MSE of second-order temporal differences.
    - ``L_freq`` = MSE of the per-channel **z-scored power spectral density**
      ``|rfft|**2``, restricted to the **1-50 Hz** band (Eq. 4). ``sfreq`` is
      used to resolve the band; the hi edge is clamped below Nyquist so the term
      is safe for arbitrary sample rates, and short signals fall back to the full
      positive-frequency band.
    """

    def __init__(
        self,
        amplitude_weight: float = 1.0,
        velocity_weight: float = 1.0,
        acceleration_weight: float = 1.0,
        frequency_weight: float = 1.0,
        *,
        sfreq: float | None = None,
        freq_band: tuple[float, float] = (1.0, 50.0),
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.amplitude_weight = float(amplitude_weight)
        self.velocity_weight = float(velocity_weight)
        self.acceleration_weight = float(acceleration_weight)
        self.frequency_weight = float(frequency_weight)
        self.sfreq = float(sfreq) if sfreq is not None and np.isfinite(sfreq) else None
        self.freq_lo = float(freq_band[0])
        self.freq_hi = float(freq_band[1])
        self.eps = float(eps)
        self._mse = nn.MSELoss()
        self._weight_sum = (
            self.amplitude_weight + self.velocity_weight + self.acceleration_weight + self.frequency_weight
        )
        if self._weight_sum <= 0:
            raise ValueError("At least one ensemble weight must be positive")

    @staticmethod
    def _diff(x: torch.Tensor) -> torch.Tensor:
        return x[..., 1:] - x[..., :-1]

    def _band_indices(self, n_samples: int, device: torch.device) -> torch.Tensor | None:
        """Resolve rfft bin indices in the 1-50 Hz band, or None for full band."""
        if self.sfreq is None or n_samples < 4:
            return None
        freqs = torch.fft.rfftfreq(n_samples, d=1.0 / self.sfreq).to(device)
        nyquist = self.sfreq / 2.0
        hi = min(self.freq_hi, nyquist - (self.sfreq / n_samples))  # keep strictly below Nyquist
        lo = max(self.freq_lo, 0.0)
        mask = (freqs >= lo) & (freqs <= hi)
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        if idx.numel() < 2:
            return None
        return idx

    def _psd_zscore(self, x: torch.Tensor, band_idx: torch.Tensor | None) -> torch.Tensor:
        """Per-channel z-scored power spectral density over the chosen band."""
        spectrum = torch.fft.rfft(x, dim=-1)
        psd = spectrum.real**2 + spectrum.imag**2  # |rfft|**2 power spectral density
        if band_idx is not None:
            psd = psd.index_select(-1, band_idx)
        mean = psd.mean(dim=-1, keepdim=True)
        std = psd.std(dim=-1, keepdim=True)
        return (psd - mean) / (std + self.eps)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weighted = self.amplitude_weight * self._mse(prediction, target)

        if self.velocity_weight > 0:
            weighted = weighted + self.velocity_weight * self._mse(self._diff(prediction), self._diff(target))

        if self.acceleration_weight > 0:
            weighted = weighted + self.acceleration_weight * self._mse(
                self._diff(self._diff(prediction)), self._diff(self._diff(target))
            )

        if self.frequency_weight > 0:
            band_idx = self._band_indices(prediction.shape[-1], prediction.device)
            pred_psd = self._psd_zscore(prediction, band_idx)
            target_psd = self._psd_zscore(target, band_idx)
            weighted = weighted + self.frequency_weight * self._mse(pred_psd, target_psd)

        # Eq. 2 normalisation: divide the weighted sum by sum(alpha).
        return weighted / self._weight_sum


def build_loss(name: str = "ensemble", **kwargs: Any) -> nn.Module:
    """facet-train loss factory.

    The paper-faithful default is the normalised four-term ensemble loss. Pass
    ``sfreq`` (forwarded by facet-train) so the frequency term can resolve the
    1-50 Hz band; without it the term degrades gracefully to the full
    positive-frequency band.
    """
    normalized = name.strip().lower()
    if normalized in {"mse", "amplitude"}:
        return nn.MSELoss()
    if normalized in {"l1", "mae"}:
        return nn.L1Loss()
    if normalized in {"smooth_l1", "huber"}:
        return nn.SmoothL1Loss()
    if normalized in {"ensemble", "ic_unet_ensemble"}:
        return IcUNetEnsembleLoss(
            amplitude_weight=float(kwargs.get("amplitude_weight", 1.0)),
            velocity_weight=float(kwargs.get("velocity_weight", 1.0)),
            acceleration_weight=float(kwargs.get("acceleration_weight", 1.0)),
            frequency_weight=float(kwargs.get("frequency_weight", 1.0)),
            sfreq=kwargs.get("sfreq"),
            freq_band=tuple(kwargs.get("freq_band", (1.0, 50.0))),
        )
    raise ValueError(f"Unknown loss '{name}'")


# ---------------------------------------------------------------------------
# Dataset wrapper: flatten context + per-channel z-score normalisation
# ---------------------------------------------------------------------------


class _SubsetDataset:
    def __init__(self, parent: "NiazyContextIcUNetDataset", indices: list[int]) -> None:
        self._parent = parent
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._parent[self._indices[idx]]


class NiazyContextIcUNetDataset:
    """Flattens the 7-epoch context into one channel-space time series.

    Wraps :class:`facet.training.dataset.NPZContextArtifactDataset`. The base
    dataset returns ``(noisy_context, target)`` with shapes
    ``(context_epochs, channels, epoch_samples)`` and ``(channels,
    epoch_samples)``. This wrapper reshapes the input to ``(channels,
    context_epochs * epoch_samples)`` for the sensor-level 1-D U-Net.

    Normalisation default is **per-channel z-score** (subtract mean, divide by
    std + eps), matching Sec 3.1/4.1 of the paper. Demean-only (the original
    behaviour) remains opt-in via ``normalize='demean'``.

    The target key is selected by ``target_type``:
    ``'clean'`` -> ``clean_center`` (paper-faithful), ``'artifact'`` ->
    ``artifact_center`` (FACETpy-compatible). When the input is z-scored, the
    target is normalised by the same per-channel input statistics so the network
    learns a self-consistent mapping; the per-example scale is exposed via
    :meth:`scale_for` for denormalisation at inference.
    """

    def __init__(
        self,
        base_dataset: Any,
        *,
        target_type: str = "clean",
        normalize: str = "zscore",
        max_examples: int | None = None,
        eps: float = 1e-8,
    ) -> None:
        self.base_dataset = base_dataset
        normalized_target = str(target_type).strip().lower()
        if normalized_target not in {"clean", "artifact"}:
            raise ValueError(f"target_type must be 'clean' or 'artifact', got '{target_type}'")
        self.target_type = normalized_target
        normalized_mode = str(normalize).strip().lower()
        if normalized_mode not in {"zscore", "demean", "none"}:
            raise ValueError(f"normalize must be 'zscore', 'demean' or 'none', got '{normalize}'")
        self.normalize = normalized_mode
        self.eps = float(eps)

        n_base = len(base_dataset)
        if n_base == 0:
            raise ValueError("base dataset must contain at least one example")
        first_noisy, first_target = base_dataset[0]
        if first_noisy.ndim != 3:
            raise ValueError("base dataset input must have shape (context_epochs, channels, samples)")
        if first_target.ndim != 2:
            raise ValueError("base dataset target must have shape (channels, samples)")

        self.context_epochs = int(first_noisy.shape[0])
        self.n_channels = int(first_noisy.shape[1])
        self.epoch_samples = int(first_noisy.shape[2])
        self.full_samples = self.context_epochs * self.epoch_samples
        self.center_index = self.context_epochs // 2
        self.center_start = self.center_index * self.epoch_samples
        self.center_stop = self.center_start + self.epoch_samples
        self.chunk_size = self.epoch_samples
        self.trigger_aligned = True
        self.sfreq = float(getattr(base_dataset, "sfreq", float("nan")))

        self._length = n_base if max_examples is None else max(0, min(int(max_examples), n_base))

    def __len__(self) -> int:
        return self._length

    def _flatten_input(self, noisy_context: np.ndarray) -> np.ndarray:
        return noisy_context.transpose(1, 0, 2).reshape(self.n_channels, self.full_samples).astype(np.float32, copy=True)

    def _channel_stats(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return mean.astype(np.float32), std.astype(np.float32)

    def scale_for(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the per-channel (mean, std) used to normalise example ``idx``.

        Useful for denormalising the network output at inference. ``std`` is 1
        when ``normalize != 'zscore'``; ``mean`` is 0 when ``normalize ==
        'none'``.
        """
        noisy_context, _ = self.base_dataset[idx]
        noisy_flat = self._flatten_input(noisy_context)
        if self.normalize == "none":
            mean = np.zeros((self.n_channels, 1), dtype=np.float32)
            std = np.ones((self.n_channels, 1), dtype=np.float32)
            return mean, std
        mean, std = self._channel_stats(noisy_flat)
        if self.normalize == "demean":
            std = np.ones_like(std)
        return mean, std

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        noisy_context, target = self.base_dataset[idx]
        noisy_flat = self._flatten_input(noisy_context)
        target_out = target.astype(np.float32, copy=True)

        if self.normalize == "none":
            return noisy_flat, target_out

        mean, std = self._channel_stats(noisy_flat)
        noisy_flat = noisy_flat - mean
        if self.normalize == "zscore":
            denom = std + self.eps
            noisy_flat = noisy_flat / denom
            # Apply the SAME per-channel input statistics to the target so the
            # network learns a self-consistent (normalised) mapping.
            target_out = (target_out - mean) / denom if self.target_type == "clean" else target_out / denom
        else:  # demean
            target_out = target_out - mean if self.target_type == "clean" else target_out
        return noisy_flat, target_out

    @property
    def input_shape(self) -> tuple[int, int]:
        return (self.n_channels, self.full_samples)

    @property
    def target_shape(self) -> tuple[int, int]:
        return (self.n_channels, self.epoch_samples)

    @property
    def n_chunks(self) -> int:
        return len(self)

    def train_val_split(self, val_ratio: float = 0.2, seed: int = 42):
        n = len(self)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n).tolist()
        n_val = max(1, int(n * val_ratio))
        val_idx_list = indices[:n_val]
        train_idx = indices[n_val:]
        if not train_idx:  # tiny datasets: never leave train empty
            train_idx = val_idx_list
        return _SubsetDataset(self, sorted(train_idx)), _SubsetDataset(self, sorted(val_idx_list))


# ---------------------------------------------------------------------------
# Optional frozen-ICA fitting helper (non-paper extension)
# ---------------------------------------------------------------------------


def _fit_ica_matrix(
    dataset_path: str | Path,
    n_channels: int,
    n_samples: int = 100_000,
    random_state: int = 0,
) -> np.ndarray:
    """Fit a frozen ICA unmixing matrix (non-paper extension; see README).

    Returns an identity matrix if FastICA is unavailable, the path is missing,
    or the fit fails. Only used when ``use_frozen_ica=True``.
    """
    try:
        from sklearn.decomposition import FastICA
    except ImportError:
        return np.eye(n_channels, dtype=np.float32)

    path = Path(dataset_path).expanduser()
    if not path.exists():
        return np.eye(n_channels, dtype=np.float32)

    with np.load(path, allow_pickle=False) as bundle:
        noisy = np.asarray(bundle["noisy_context"], dtype=np.float32)

    if noisy.ndim != 4 or noisy.shape[2] != n_channels:
        return np.eye(n_channels, dtype=np.float32)

    flat = noisy.transpose(0, 2, 1, 3).reshape(-1, n_channels, noisy.shape[1] * noisy.shape[3])
    flat = flat.transpose(0, 2, 1).reshape(-1, n_channels)
    rng = np.random.default_rng(random_state)
    if flat.shape[0] > n_samples:
        idx = rng.choice(flat.shape[0], size=n_samples, replace=False)
        flat = flat[idx]

    ica = FastICA(
        n_components=n_channels,
        whiten="unit-variance",
        max_iter=500,
        tol=1e-4,
        random_state=random_state,
    )
    try:
        ica.fit(flat)
    except Exception:
        return np.eye(n_channels, dtype=np.float32)
    return ica.components_.astype(np.float32)


# ---------------------------------------------------------------------------
# facet-train factories
# ---------------------------------------------------------------------------


def _resolve_context_dims(
    input_shape: tuple[int, ...] | None,
    n_channels: int | None,
    context_epochs: int | None,
    epoch_samples: int | None,
) -> tuple[int, int, int]:
    """Resolve (n_channels, context_epochs, epoch_samples) from injected kwargs."""
    if input_shape is not None and len(input_shape) == 2:
        resolved_channels = int(input_shape[0])
        full_samples = int(input_shape[1])
        if context_epochs and epoch_samples:
            if context_epochs * epoch_samples != full_samples:
                raise ValueError(
                    f"context_epochs * epoch_samples ({context_epochs}*{epoch_samples}) does not "
                    f"match input_shape[1]={full_samples}"
                )
            return resolved_channels, int(context_epochs), int(epoch_samples)
        resolved_epochs = int(context_epochs) if context_epochs else 7
        if full_samples % resolved_epochs != 0:
            raise ValueError(f"Cannot split full_samples={full_samples} into {resolved_epochs} epochs")
        return resolved_channels, resolved_epochs, full_samples // resolved_epochs

    if input_shape is not None and len(input_shape) == 3:
        # (context_epochs, channels, epoch_samples) layout from the base dataset.
        return int(input_shape[1]), int(input_shape[0]), int(input_shape[2])

    if n_channels is None or context_epochs is None or epoch_samples is None:
        raise ValueError(
            "build_model requires input_shape or (n_channels, context_epochs, epoch_samples)"
        )
    return int(n_channels), int(context_epochs), int(epoch_samples)


def build_model(
    input_shape: tuple[int, ...] | None = None,
    target_shape: tuple[int, int] | None = None,
    context_epochs: int | None = None,
    epoch_samples: int | None = None,
    n_channels: int | None = None,
    target_type: str = "clean",
    base_channels: int = 64,
    depth: int = 4,
    kernel_size: int = 7,
    activation: str = "relu",
    use_frozen_ica: bool = False,
    dataset_path: str | None = None,
    fit_ica: bool = False,
    ica_random_state: int = 0,
    **_: object,
) -> IcUNetPaperAccurate:
    """facet-train model factory (paper-faithful sensor-level U-Net).

    Accepts the kwargs facet-train injects (``n_channels``, ``chunk_size``,
    ``sfreq``, ``target_type``, ``input_shape``, ``target_shape``,
    ``context_epochs``, ``epoch_samples``, ...) via explicit args or ``**_``;
    none are required-positional. ``target_type`` defaults to the paper-faithful
    clean-reconstruction objective.
    """
    resolved_channels, resolved_epochs, resolved_epoch_samples = _resolve_context_dims(
        input_shape, n_channels, context_epochs, epoch_samples
    )

    output_type = str(target_type).strip().lower() if target_type else "clean"
    if output_type not in {"clean", "artifact"}:
        output_type = "clean"

    ica_init = None
    if use_frozen_ica and fit_ica and dataset_path is not None:
        ica_init = _fit_ica_matrix(
            dataset_path=dataset_path,
            n_channels=resolved_channels,
            random_state=ica_random_state,
        )

    return IcUNetPaperAccurate(
        n_channels=resolved_channels,
        context_epochs=resolved_epochs,
        epoch_samples=resolved_epoch_samples,
        base_channels=int(base_channels),
        depth=int(depth),
        kernel_size=int(kernel_size),
        activation=activation,
        output_type=output_type,
        use_frozen_ica=bool(use_frozen_ica),
        ica_init=ica_init,
    )


def build_dataset(
    path: str | None = None,
    context_path: str | None = None,
    max_examples: int | None = None,
    target_type: str = "clean",
    normalize: str = "zscore",
    **_: object,
) -> NiazyContextIcUNetDataset:
    """facet-train dataset factory.

    Selects the target key by ``target_type`` (``clean`` -> ``clean_center``,
    ``artifact`` -> ``artifact_center``) and applies per-channel z-score
    normalisation by default (Sec 3.1/4.1). The base
    :class:`NPZContextArtifactDataset` returns un-normalised arrays; this wrapper
    owns the normalisation so the per-example scale is recoverable.
    """
    dataset_path = Path(path or context_path or "").expanduser()
    if not str(dataset_path) or str(dataset_path) == ".":
        raise ValueError("build_dataset requires path or context_path")

    normalized_target = str(target_type).strip().lower()
    if normalized_target not in {"clean", "artifact"}:
        raise ValueError(f"target_type must be 'clean' or 'artifact', got '{target_type}'")
    target_key = "clean_center" if normalized_target == "clean" else "artifact_center"

    base = NPZContextArtifactDataset(
        path=dataset_path,
        input_key="noisy_context",
        target_key=target_key,
        demean_input=False,
        demean_target=False,
    )
    return NiazyContextIcUNetDataset(
        base,
        target_type=normalized_target,
        normalize=normalize,
        max_examples=max_examples,
    )
