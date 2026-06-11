"""Training factories for the paper-accurate Conv-TasNet source separator.

This edition is a more faithful re-implementation of

    Y. Luo and N. Mesgarani, "Conv-TasNet: Surpassing Ideal Time-Frequency
    Magnitude Masking for Speech Separation," IEEE/ACM Transactions on Audio,
    Speech, and Language Processing, vol. 27, no. 8, pp. 1256-1266, 2019.
    (arXiv:1809.07454v3).

vs. the original ``facet.models.conv_tasnet`` package. The high-value
faithfulness fixes are:

1.  LINEAR encoder by default (no ReLU non-negativity) paired with a Sigmoid
    mask -- this is the paper's BEST published configuration (Table III), not
    the original code's ReLU encoder which corresponds to a lower-scoring row.
2.  An EXPLICIT skip-connection width ``skip_channels`` (Sc) that is distinct
    from the bottleneck width ``bottleneck_channels`` (B), as defined in Fig.1C
    and Table I. The mask-prediction 1x1-conv reads from the Sc-wide skip space.

EEG-fMRI-appropriate deviations (documented, NOT blindly copied from the paper):

*   Loss defaults to MSE / weighted MSE on ORDERED sources, not SI-SNR + uPIT.
    The two sources (clean EEG, gradient artifact) are known and ordered, so
    permutation-invariant training is unnecessary, and SI-SNR's scale invariance
    would discard the amplitude information that matters for a deterministic
    AAS-derived dataset where the artifact must be subtracted at true scale.
    SI-SNR is offered only as an ablation option.
*   An optional source-additivity ("consistency") penalty that exploits the
    EXACT property noisy = clean + artifact of the FACETpy dataset. This is a
    FACETpy-appropriate enhancement that goes beyond the paper.
*   Input-length-matched architecture defaults (N=256, H=256, R=2) for the
    ~512-sample EEG epochs, rather than the paper's 4-second/8 kHz speech dims
    (N=512, H=512, R=3). The paper reference values are documented in the
    README and ``documentation/paper_accuracy_review.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


class _GlobalLayerNorm(torch.nn.Module):
    """Global layer norm over the channel and time dimensions (gLN).

    Matches eq. 9-11 of the paper: features are normalised by the mean and
    variance taken jointly over BOTH the channel axis and the time axis, with a
    learned per-channel scale (gamma) and bias (beta) in R^{N x 1}. This is the
    non-causal normalisation the paper shows is ~2.5 dB better than the causal
    cLN, and it is the correct choice for offline EEG-fMRI correction.
    """

    def __init__(self, n_features: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(1, n_features, 1))
        self.beta = torch.nn.Parameter(torch.zeros(1, n_features, 1))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class _TemporalBlock(torch.nn.Module):
    """One dilated 1-D conv block of the TCN separation module (Fig. 1C).

    Faithful ordering ``conv -> PReLU -> Norm`` for both the expansion 1x1-conv
    and the depthwise D-conv. The block returns two separate paths:

    *   a residual path (H -> B) added back to the block input, and
    *   a skip-connection path (H -> Sc) summed across all blocks.

    The skip width ``skip_channels`` (Sc) is now an explicit, independently
    tuned hyperparameter distinct from the bottleneck width ``bottleneck_channels``
    (B), matching Table I of the paper. The depthwise-separable convolution is
    realised as a grouped depthwise D-conv whose paired pointwise (1x1) mixing is
    folded into the residual/skip convs -- the canonical form used by mainstream
    reference repos (asteroid, kaituoxu/Conv-TasNet); see the accuracy review for
    the rationale.
    """

    def __init__(
        self,
        bottleneck_channels: int,
        hidden_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.expand = torch.nn.Conv1d(bottleneck_channels, hidden_channels, kernel_size=1)
        self.expand_act = torch.nn.PReLU()
        self.expand_norm = _GlobalLayerNorm(hidden_channels)
        self.dconv = torch.nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            groups=hidden_channels,
        )
        self.dconv_act = torch.nn.PReLU()
        self.dconv_norm = _GlobalLayerNorm(hidden_channels)
        self.residual = torch.nn.Conv1d(hidden_channels, bottleneck_channels, kernel_size=1)
        self.skip = torch.nn.Conv1d(hidden_channels, skip_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.expand_norm(self.expand_act(self.expand(x)))
        h = self.dconv_norm(self.dconv_act(self.dconv(h)))
        return x + self.residual(h), self.skip(h)


class ConvTasNetSeparator(torch.nn.Module):
    """Paper-accurate 1-D Conv-TasNet for single-channel artifact separation.

    Input shape: ``(batch, 1, samples)``. Output shape:
    ``(batch, n_sources, samples)``. With ``n_sources=2`` the first source is
    the clean EEG estimate (index 0) and the second is the gradient artifact
    estimate (index 1) -- ordered, so no permutation handling is needed.

    Pipeline (Fig. 1A/B):
        encoder (1-D conv, optional linear/ReLU activation)
        -> gLN -> bottleneck 1x1-conv (N -> B)
        -> TCN: R repeats x X dilated blocks, dilation 2^x
        -> PReLU -> mask 1x1-conv (Sc -> C*N) -> mask activation
        -> elementwise mask the SHARED encoder latent
        -> ONE shared transposed-conv decoder per source -> overlap-add.
    """

    def __init__(
        self,
        *,
        n_sources: int = 2,
        encoder_filters: int = 256,
        encoder_kernel: int = 16,
        encoder_activation: str = "linear",
        bottleneck_channels: int = 128,
        hidden_channels: int = 256,
        skip_channels: int = 128,
        block_kernel: int = 3,
        n_blocks: int = 8,
        n_repeats: int = 2,
        mask_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        if encoder_kernel < 2 or encoder_kernel % 2 != 0:
            raise ValueError("encoder_kernel must be an even integer >= 2")
        self.n_sources = int(n_sources)
        self.encoder_filters = int(encoder_filters)
        self.encoder_kernel = int(encoder_kernel)
        self.encoder_activation = str(encoder_activation).strip().lower()
        self.bottleneck_channels = int(bottleneck_channels)
        self.hidden_channels = int(hidden_channels)
        self.skip_channels = int(skip_channels)
        self.block_kernel = int(block_kernel)
        self.n_blocks = int(n_blocks)
        self.n_repeats = int(n_repeats)
        self.mask_activation = mask_activation

        if self.encoder_activation not in {"linear", "identity", "none", "relu"}:
            raise ValueError(
                f"Unsupported encoder_activation '{encoder_activation}'. Expected 'linear' or 'relu'."
            )

        self.encoder_stride = self.encoder_kernel // 2
        self.encoder = torch.nn.Conv1d(
            1,
            self.encoder_filters,
            kernel_size=self.encoder_kernel,
            stride=self.encoder_stride,
            bias=False,
        )
        self.pre_norm = _GlobalLayerNorm(self.encoder_filters)
        self.bottleneck = torch.nn.Conv1d(self.encoder_filters, self.bottleneck_channels, kernel_size=1)
        self.tcn_blocks = torch.nn.ModuleList(
            [
                _TemporalBlock(
                    bottleneck_channels=self.bottleneck_channels,
                    hidden_channels=self.hidden_channels,
                    skip_channels=self.skip_channels,
                    kernel_size=self.block_kernel,
                    dilation=2**block_idx,
                )
                for _ in range(self.n_repeats)
                for block_idx in range(self.n_blocks)
            ]
        )
        self.mask_act = torch.nn.PReLU()
        # Mask predictor reads from the Sc-wide skip space (paper Fig. 1B/C).
        self.mask_conv = torch.nn.Conv1d(
            self.skip_channels,
            self.n_sources * self.encoder_filters,
            kernel_size=1,
        )
        self.decoder = torch.nn.ConvTranspose1d(
            self.encoder_filters,
            1,
            kernel_size=self.encoder_kernel,
            stride=self.encoder_stride,
            bias=False,
        )

    def _encode(self, mixture: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(mixture)
        # Paper's best config (Table III) uses a LINEAR encoder. ReLU is kept
        # only as the original-TasNet option.
        if self.encoder_activation == "relu":
            latent = torch.relu(latent)
        return latent

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        if mixture.dim() != 3 or mixture.shape[1] != 1:
            raise ValueError(
                f"ConvTasNetSeparator expects shape (batch, 1, samples), got {tuple(mixture.shape)}"
            )
        n_samples = mixture.shape[-1]
        latent = self._encode(mixture)
        bottleneck = self.bottleneck(self.pre_norm(latent))

        skip_sum = torch.zeros(
            bottleneck.shape[0],
            self.skip_channels,
            bottleneck.shape[-1],
            dtype=bottleneck.dtype,
            device=bottleneck.device,
        )
        h = bottleneck
        for block in self.tcn_blocks:
            h, skip = block(h)
            skip_sum = skip_sum + skip

        mask_logits = self.mask_conv(self.mask_act(skip_sum))
        masks = self._apply_mask_activation(mask_logits)
        masks = masks.view(mixture.shape[0], self.n_sources, self.encoder_filters, latent.shape[-1])

        sources: list[torch.Tensor] = []
        for src_idx in range(self.n_sources):
            masked = latent * masks[:, src_idx]
            decoded = self.decoder(masked)
            sources.append(decoded[..., :n_samples])
        return torch.cat(sources, dim=1)

    def _apply_mask_activation(self, mask_logits: torch.Tensor) -> torch.Tensor:
        normalized = self.mask_activation.strip().lower()
        if normalized == "sigmoid":
            return torch.sigmoid(mask_logits)
        if normalized == "relu":
            return torch.relu(mask_logits)
        if normalized == "softmax":
            batch, _, frames = mask_logits.shape
            reshaped = mask_logits.view(batch, self.n_sources, self.encoder_filters, frames)
            return torch.softmax(reshaped, dim=1).view(batch, -1, frames)
        raise ValueError(
            f"Unsupported mask_activation '{self.mask_activation}'. Expected one of: sigmoid, relu, softmax."
        )


# ---------------------------------------------------------------------------
# Channel-wise dataset wrapper
# ---------------------------------------------------------------------------


class ChannelWiseSourceSeparationDataset:
    """Yield ``(noisy, sources)`` per channel from an NPZ context bundle.

    Each item returns ``(mixture, sources)`` where ``mixture`` has shape
    ``(1, samples)`` and ``sources`` has shape ``(2, samples)`` with index
    ``0`` = clean EEG and index ``1`` = gradient artifact. Reuses the
    channel-wise flattening and optional demeaning of the original edition.

    Note: raw-input demeaning is an EEG-specific addition (DC-offset removal)
    NOT present in the paper, which only normalises the encoder latent via gLN.
    It is kept as a sensible default and applied identically at train and
    inference time so the two never diverge.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_examples: int | None = None,
        demean_input: bool = True,
        demean_target: bool = True,
    ) -> None:
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        with np.load(self.path, allow_pickle=True) as bundle:
            noisy_center = bundle["noisy_center"].astype(np.float32, copy=False)
            clean_center = bundle["clean_center"].astype(np.float32, copy=False)
            artifact_center = bundle["artifact_center"].astype(np.float32, copy=False)
            self.sfreq = float(bundle["sfreq"][0]) if "sfreq" in bundle else float("nan")

        if noisy_center.shape != clean_center.shape or noisy_center.shape != artifact_center.shape:
            raise ValueError("noisy_center, clean_center, and artifact_center must have identical shapes")
        if noisy_center.ndim != 3:
            raise ValueError("noisy_center must have shape (examples, channels, samples)")

        self.noisy_center = noisy_center
        self.clean_center = clean_center
        self.artifact_center = artifact_center
        self.n_examples = int(noisy_center.shape[0])
        self.n_channels = int(noisy_center.shape[1])
        self.epoch_samples = int(noisy_center.shape[2])
        self.chunk_size = self.epoch_samples
        self.target_type = "artifact"
        self.trigger_aligned = True
        self.demean_input = bool(demean_input)
        self.demean_target = bool(demean_target)

        total = self.n_examples * self.n_channels
        self._length = total if max_examples is None else max(0, min(int(max_examples), total))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        example_idx = int(idx) // self.n_channels
        channel_idx = int(idx) % self.n_channels

        mixture = self.noisy_center[example_idx, channel_idx : channel_idx + 1, :].astype(np.float32, copy=True)
        clean = self.clean_center[example_idx, channel_idx, :].astype(np.float32, copy=True)
        artifact = self.artifact_center[example_idx, channel_idx, :].astype(np.float32, copy=True)

        if self.demean_input:
            mixture -= mixture.mean(axis=-1, keepdims=True)
        if self.demean_target:
            clean -= clean.mean()
            artifact -= artifact.mean()

        sources = np.stack([clean, artifact], axis=0).astype(np.float32, copy=False)
        return mixture, sources

    @property
    def input_shape(self) -> tuple[int, int]:
        return (1, self.epoch_samples)

    @property
    def target_shape(self) -> tuple[int, int]:
        return (2, self.epoch_samples)

    @property
    def n_chunks(self) -> int:
        return len(self)

    def train_val_split(self, val_ratio: float = 0.2, seed: int = 42):
        n = len(self)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n).tolist()
        n_val = max(1, int(n * val_ratio))
        val_set = set(indices[:n_val])
        train_idx = [i for i in range(n) if i not in val_set]
        val_idx = [i for i in range(n) if i in val_set]
        return _SubsetDataset(self, train_idx), _SubsetDataset(self, val_idx)


class _SubsetDataset:
    def __init__(self, parent: ChannelWiseSourceSeparationDataset, indices: list[int]) -> None:
        self._parent = parent
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._parent[self._indices[idx]]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class _NegSISDR(torch.nn.Module):
    """Negative SI-SDR averaged over the source axis (no permutation).

    Offered only as an ABLATION. SI-SNR (eq. 15 of the paper) is scale-invariant,
    which is desirable for speech (arbitrary absolute amplitude) but actively
    discards meaningful amplitude information for our deterministic AAS-derived
    artifact, where the artifact must be subtracted at true scale. uPIT is NOT
    implemented because the (clean, artifact) sources are known and ORDERED.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Negative SI-SDR requires matching shapes, got {tuple(prediction.shape)} vs {tuple(target.shape)}"
            )
        prediction = prediction - prediction.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        scale = (prediction * target).sum(dim=-1, keepdim=True) / (target.pow(2).sum(dim=-1, keepdim=True) + self.eps)
        projection = scale * target
        noise = prediction - projection
        sdr = 10.0 * torch.log10((projection.pow(2).sum(dim=-1) + self.eps) / (noise.pow(2).sum(dim=-1) + self.eps))
        return -sdr.mean()


class _WeightedSourceMSE(torch.nn.Module):
    """MSE per source with separate weights for clean and artifact."""

    def __init__(self, clean_weight: float = 1.0, artifact_weight: float = 1.0) -> None:
        super().__init__()
        self.clean_weight = float(clean_weight)
        self.artifact_weight = float(artifact_weight)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        clean_mse = torch.nn.functional.mse_loss(prediction[:, 0], target[:, 0])
        artifact_mse = torch.nn.functional.mse_loss(prediction[:, 1], target[:, 1])
        return self.clean_weight * clean_mse + self.artifact_weight * artifact_mse


class _ConsistencyMSE(torch.nn.Module):
    """MSE on the sources plus a source-additivity (consistency) penalty.

    A FACETpy-appropriate enhancement BEYOND the paper. Because the dataset
    obeys the exact identity ``noisy = clean + artifact``, the sum of the two
    predicted sources should reconstruct the mixture. We add a penalty on the
    reconstruction of the target mixture (clean + artifact targets) so the two
    predicted sources stay consistent with one another:

        loss = mse(pred, target)
             + consistency_weight * mse(pred[:,0] + pred[:,1], target[:,0] + target[:,1])

    The target mixture ``target[:,0] + target[:,1]`` equals the (demeaned)
    noisy signal under the dataset's additivity, so this needs no extra inputs.
    The paper relaxes the unit-summation mask constraint; here we instead lean
    on the stronger, exactly-true additivity available for EEG-fMRI data.
    """

    def __init__(
        self,
        clean_weight: float = 1.0,
        artifact_weight: float = 1.0,
        consistency_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.clean_weight = float(clean_weight)
        self.artifact_weight = float(artifact_weight)
        self.consistency_weight = float(consistency_weight)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        clean_mse = torch.nn.functional.mse_loss(prediction[:, 0], target[:, 0])
        artifact_mse = torch.nn.functional.mse_loss(prediction[:, 1], target[:, 1])
        pred_mix = prediction[:, 0] + prediction[:, 1]
        target_mix = target[:, 0] + target[:, 1]
        consistency = torch.nn.functional.mse_loss(pred_mix, target_mix)
        return (
            self.clean_weight * clean_mse
            + self.artifact_weight * artifact_mse
            + self.consistency_weight * consistency
        )


# ---------------------------------------------------------------------------
# Factories used by facet-train
# ---------------------------------------------------------------------------


def build_model(
    *,
    n_sources: int = 2,
    encoder_filters: int = 256,
    encoder_kernel: int = 16,
    encoder_activation: str = "linear",
    bottleneck_channels: int = 128,
    hidden_channels: int = 256,
    skip_channels: int = 128,
    block_kernel: int = 3,
    n_blocks: int = 8,
    n_repeats: int = 2,
    mask_activation: str = "sigmoid",
    **_: object,
) -> ConvTasNetSeparator:
    """Build the paper-accurate Conv-TasNet separator.

    Accepts (and ignores via ``**_``) the kwargs facet-train injects:
    ``n_channels, chunk_size, sfreq, target_type, training_config,
    input_shape, target_shape, context_epochs, epoch_samples``. Explicit YAML
    ``model.kwargs`` override the defaults below.

    Defaults are input-length-matched to ~512-sample EEG epochs. The paper's
    headline non-causal config (5.1M params) is N=512, L=16, B=128, H=512,
    Sc=128, P=3, X=8, R=3 -- see the README / accuracy review.
    """
    return ConvTasNetSeparator(
        n_sources=n_sources,
        encoder_filters=encoder_filters,
        encoder_kernel=encoder_kernel,
        encoder_activation=encoder_activation,
        bottleneck_channels=bottleneck_channels,
        hidden_channels=hidden_channels,
        skip_channels=skip_channels,
        block_kernel=block_kernel,
        n_blocks=n_blocks,
        n_repeats=n_repeats,
        mask_activation=mask_activation,
    )


def build_loss(
    name: str = "mse",
    *,
    clean_weight: float = 1.0,
    artifact_weight: float = 1.0,
    consistency_weight: float = 0.5,
    **_: object,
) -> torch.nn.Module:
    """Build the training loss.

    Defaults to ``mse`` on ordered sources -- the EEG-appropriate choice. No
    permutation-invariant training (sources are known and ordered). SI-SNR
    (``si_sdr_neg``) is available only as an ablation. ``consistency_mse`` adds
    the FACETpy-specific source-additivity penalty.
    """
    normalized = name.strip().lower()
    if normalized == "mse":
        return torch.nn.MSELoss()
    if normalized in {"l1", "mae"}:
        return torch.nn.L1Loss()
    if normalized in {"weighted_mse", "source_mse"}:
        return _WeightedSourceMSE(clean_weight=clean_weight, artifact_weight=artifact_weight)
    if normalized in {"consistency_mse", "additivity_mse", "consistency"}:
        return _ConsistencyMSE(
            clean_weight=clean_weight,
            artifact_weight=artifact_weight,
            consistency_weight=consistency_weight,
        )
    if normalized in {"si_sdr_neg", "neg_si_sdr", "si_sdr"}:
        return _NegSISDR()
    raise ValueError(
        f"Unsupported loss name '{name}'. Use one of: mse, l1, weighted_mse, consistency_mse, si_sdr_neg."
    )


def build_dataset(
    path: str | None = None,
    context_path: str | None = None,
    max_examples: int | None = None,
    demean_input: bool = True,
    demean_target: bool = True,
    **_: object,
) -> ChannelWiseSourceSeparationDataset:
    dataset_path = path or context_path
    if not dataset_path:
        raise ValueError("build_dataset requires path or context_path")
    return ChannelWiseSourceSeparationDataset(
        path=dataset_path,
        max_examples=max_examples,
        demean_input=demean_input,
        demean_target=demean_target,
    )


__all__ = [
    "ConvTasNetSeparator",
    "ChannelWiseSourceSeparationDataset",
    "build_model",
    "build_loss",
    "build_dataset",
]
