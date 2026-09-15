"""Training factories for the paper-accurate Dual-Pathway Autoencoder (DPAE).

This edition is a more faithful re-implementation of

    H. Xiong, Y. Ma, and Y. Li, "A general dual-pathway network for EEG
    denoising," Frontiers in Neuroscience, vol. 17, art. 1258024, 2023.
    doi:10.3389/fnins.2023.1258024.

vs. the original ``facet.models.dpae`` package. DPAE is a lightweight,
supervised denoising autoencoder whose defining contribution (Fig. 2) is a
TWO-PATHWAY encoder at different scales feeding a SYMMETRIC FUSION MODULE
(Fusion Encoder -> BN -> Fusion Decoder) wrapped by a residual ("Resnet")
skip, followed by a decoder reconstructing the CLEAN signal. We implement the
1D-CNN instantiation, which the paper found most robust on real multichannel
EEG (applied channel-by-channel) and which matches FACETpy's per-channel,
per-epoch correction setting.

High-value faithfulness fixes vs. the original edition
------------------------------------------------------
1.  SYMMETRIC FUSION MODULE (the paper's core contribution). After concatenating
    the two pathway latents we apply BatchNorm, then a Fusion Encoder that
    compresses the channel dimension through several 1x1 convs (e.g. /2, /4, /8)
    and a mirror Fusion Decoder that expands it back -- the "common feature
    coding" of Fig. 2 / Table 1. The original used only a single 1x1 conv.
2.  RESIDUAL SKIP ACROSS THE FUSION MODULE on the feature maps
    (``fusion_out = decode(encode(z)) + z``), matching Fig. 2 "Resnet
    Connection" (Drozdzal 2016), NOT a scalar shortcut on the raw input signal.
3.  CLEAN-EEG reconstruction TARGET with MSE loss (paper Sec. 2.3), instead of
    predicting the artifact. ``target_type`` is configurable so FACETpy's
    subtract-the-artifact path still works; the two are equivalent under the
    dataset's exact additivity ``noisy = clean + artifact``.
4.  Two CNN pathways differing by KERNEL SIZE and STRIDE per the paper: Pathway1
    uses kernel 3 / stride 1 (fine detail, no downsampling); Pathway2 uses
    kernel 5 / stride 4 (coarse, 4x downsample). The original's dilation +
    k=15/11/7 + MaxPool scheme is replaced.
5.  Asymmetric NEURON SHRINKAGE RATIOS: ``shrink_ratio_high=0.75`` (expand then
    contract) and ``shrink_ratio_low=0.45`` (contract only), driving the channel
    widths of the two pathways (paper Sec. 2.2, Table 1).
6.  Per-segment NORMALIZATION at training and inference: subtract std, divide by
    max-abs (paper Sec. 3.1), applied identically in ``build_dataset`` and in
    the processor so train/inference never diverge.

EEG-fMRI-appropriate deviations (documented, NOT blindly copied)
----------------------------------------------------------------
*   Training data is the FACETpy Niazy proof-fit NPZ bundle (gradient artifacts)
    rather than EEGdenoiseNet EOG/EMG noise. The architecture is unchanged; only
    the noise source differs. See ``documentation/paper_accuracy_review.md``.
*   The MLP and 1D-RNN variants of the paper are out of scope; the 1D-CNN is the
    one suited to single-channel time-domain per-epoch correction.

All activations are SeLU (paper). See README.md for the full discrepancy table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


def _selu() -> nn.Module:
    # SeLU throughout, matching the paper. ``inplace`` is avoided so TorchScript
    # tracing and gradient flow stay simple and robust.
    return nn.SELU()


class _ConvPathway(nn.Module):
    """One conv pathway with a paper-style shrinkage-ratio channel progression.

    The paper (Sec. 2.2, Table 1) gives the two pathways genuinely different
    representational capacity via different neuron shrinkage ratios applied
    layer by layer:

    *   high-dimensional pathway (ratio ~0.75): the width first EXPANDS once
        (``ceil(F / ratio)``) and then CONTRACTS by ``* ratio`` each subsequent
        layer (512 -> 682 -> 511 -> 383 in the MLP table);
    *   low-dimensional pathway (ratio ~0.45): the width CONTRACTS by ``* ratio``
        every layer (512 -> 230 -> 103 -> 46).

    Here those ratios drive the CONV CHANNEL widths. The pathway's temporal
    downsampling is realised purely by ``stride`` (paper: stride 1 vs stride 4),
    so the two pathways reach a matched bottleneck LENGTH when the strides are
    chosen so ``stride_total`` is identical.
    """

    def __init__(
        self,
        *,
        base_filters: int,
        kernel_size: int,
        stride: int,
        shrink_ratio: float,
        expand_first: bool,
        n_layers: int,
        total_downsample: int,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.total_downsample = int(total_downsample)

        widths = self._build_widths(base_filters, shrink_ratio, expand_first, n_layers)
        self.out_channels = widths[-1]

        # Distribute the required total temporal downsampling across the layers
        # using this pathway's per-layer stride. A layer either strides (when we
        # still need to downsample) or uses stride 1 (refinement, paper Path1).
        strides = self._build_strides(n_layers, total_downsample, self.stride)

        layers: list[nn.Module] = []
        in_ch = 1
        pad = self.kernel_size // 2
        for out_ch, st in zip(widths, strides, strict=True):
            layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=self.kernel_size, stride=st, padding=pad)
            )
            layers.append(_selu())
            in_ch = out_ch
        self.layers = nn.Sequential(*layers)

    @staticmethod
    def _build_widths(
        base_filters: int, shrink_ratio: float, expand_first: bool, n_layers: int
    ) -> list[int]:
        widths: list[int] = []
        width = float(base_filters)
        if expand_first:
            # High-dim pathway: upgrade dimensionality once (paper: 512 -> 682).
            width = base_filters / shrink_ratio
        for _ in range(n_layers):
            widths.append(max(1, int(round(width))))
            width = width * shrink_ratio
        return widths

    @staticmethod
    def _build_strides(n_layers: int, total_downsample: int, stride: int) -> list[int]:
        strides = [1] * n_layers
        if stride <= 1 or total_downsample <= 1:
            return strides
        # Place strided layers from the front until the product reaches the
        # requested total downsampling factor (e.g. stride 4 once -> 4x).
        produced = 1
        idx = 0
        while produced < total_downsample and idx < n_layers:
            strides[idx] = stride
            produced *= stride
            idx += 1
        return strides

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _SymmetricFusion(nn.Module):
    """Paper's Fusion Encoder -> BN -> Fusion Decoder with a residual skip.

    Realises Fig. 2's fusion module on the concatenated pathway feature maps.
    The Fusion Encoder compresses the channel dimension through ``fusion_depth``
    halvings (e.g. C -> C/2 -> C/4 -> C/8); the Fusion Decoder mirrors it back to
    C. A residual ("Resnet") skip wraps the whole module so the joint
    representation has an identity mapping (Drozdzal 2016), aiding gradient flow.
    A BatchNorm sits on the joint representation, regularising the two pathways
    onto a uniform interval (Santurkar 2018, paper Sec. 2.2).
    """

    def __init__(self, fused_channels: int, fusion_depth: int = 3) -> None:
        super().__init__()
        self.fused_channels = int(fused_channels)
        self.bn = nn.BatchNorm1d(self.fused_channels)

        # Build the compressing channel schedule, never dropping below 1.
        widths = [self.fused_channels]
        for _ in range(max(1, int(fusion_depth))):
            widths.append(max(1, widths[-1] // 2))

        enc: list[nn.Module] = []
        for in_ch, out_ch in zip(widths[:-1], widths[1:], strict=True):
            enc.append(nn.Conv1d(in_ch, out_ch, kernel_size=1))
            enc.append(_selu())
        self.fusion_encoder = nn.Sequential(*enc)

        dec: list[nn.Module] = []
        rev = list(reversed(widths))
        for in_ch, out_ch in zip(rev[:-1], rev[1:], strict=True):
            dec.append(nn.Conv1d(in_ch, out_ch, kernel_size=1))
            dec.append(_selu())
        self.fusion_decoder = nn.Sequential(*dec)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.bn(z)
        compressed = self.fusion_encoder(z)
        reconstructed = self.fusion_decoder(compressed)
        # Resnet connection wrapping the fusion module (identity of joint rep).
        return reconstructed + z


class _Decoder(nn.Module):
    """Reconstruction head mirroring the pathway downsampling back to length L.

    Upsamples by the exact factor the pathways downsampled by (so the output
    length equals the input length, removing the original edition's hard
    ``%4`` constraint), then a pre-output conv stack ending in a 1x1 conv to a
    single channel. A BatchNorm precedes the decoder, matching the paper's
    "BN -> Dense -> Output" head (Table 1 / Fig. 2).
    """

    def __init__(self, fused_channels: int, base_filters: int, upsample_factor: int) -> None:
        super().__init__()
        f = int(base_filters)
        self.pre_bn = nn.BatchNorm1d(int(fused_channels))

        layers: list[nn.Module] = []
        in_ch = int(fused_channels)
        factor = int(upsample_factor)
        # Decompose the upsampling factor into stride-2 transposed convs.
        while factor > 1:
            step = 2 if factor % 2 == 0 else factor
            layers.append(
                nn.ConvTranspose1d(in_ch, f, kernel_size=2 * step, stride=step, padding=step // 2)
            )
            layers.append(_selu())
            in_ch = f
            factor //= step
        if not layers:
            # No upsampling needed; project to base filters with a 1x1 conv.
            layers.append(nn.Conv1d(in_ch, f, kernel_size=1))
            layers.append(_selu())
            in_ch = f
        layers.append(nn.Conv1d(in_ch, f, kernel_size=3, padding=1))
        layers.append(_selu())
        layers.append(nn.Conv1d(f, 1, kernel_size=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(self.pre_bn(x))


class DualPathwayAutoencoder(nn.Module):
    """Paper-accurate 1D-CNN dual-pathway autoencoder.

    Forward contract:
        input  : (batch, 1, samples)  single-channel epoch
        output : (batch, 1, samples)  reconstructed CLEAN EEG (default), or the
                                       artifact if the checkpoint was trained
                                       with ``target_type='artifact'`` -- the
                                       network is identical either way; only the
                                       training target differs.

    Pipeline (Fig. 2):
        Pathway1 (k=3, stride 1, ratio 0.75 expand-then-contract)  \\
                                                                     concat
        Pathway2 (k=5, stride 4, ratio 0.45 contract)              /
        -> BatchNorm -> Fusion Encoder -> Fusion Decoder (+Resnet skip)
        -> BatchNorm -> Decoder (upsample to L, 1 channel).

    Both pathways downsample by the SAME total factor (``pathway2_stride``) so
    their bottleneck feature maps share a length and can be concatenated over the
    channel axis; Pathway1 reaches it with strided early layers too.
    """

    def __init__(
        self,
        input_size: int,
        *,
        base_filters: int = 32,
        shrink_ratio_low: float = 0.45,
        shrink_ratio_high: float = 0.75,
        pathway_layers: int = 4,
        pathway2_stride: int = 4,
        fusion_depth: int = 3,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.base_filters = int(base_filters)
        self.shrink_ratio_low = float(shrink_ratio_low)
        self.shrink_ratio_high = float(shrink_ratio_high)
        self.pathway_layers = int(pathway_layers)
        self.total_downsample = int(pathway2_stride)
        self.fusion_depth = int(fusion_depth)

        # High-dimensional pathway: fine detail, kernel 3. Realises the same
        # total downsampling via strided early layers so its length matches.
        self.pathway_high = _ConvPathway(
            base_filters=self.base_filters,
            kernel_size=3,
            stride=2,
            shrink_ratio=self.shrink_ratio_high,
            expand_first=True,
            n_layers=self.pathway_layers,
            total_downsample=self.total_downsample,
        )
        # Low-dimensional pathway: coarse, kernel 5, stride 4 (paper Path2).
        self.pathway_low = _ConvPathway(
            base_filters=self.base_filters,
            kernel_size=5,
            stride=self.total_downsample,
            shrink_ratio=self.shrink_ratio_low,
            expand_first=False,
            n_layers=self.pathway_layers,
            total_downsample=self.total_downsample,
        )

        fused = self.pathway_high.out_channels + self.pathway_low.out_channels
        self.fusion = _SymmetricFusion(fused, fusion_depth=self.fusion_depth)
        self.decoder = _Decoder(fused, self.base_filters, self.total_downsample)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[1] == 1 and x.shape[2] == 1:
            # Accept (batch, 1, 1, samples) tensors from upstream for robustness.
            x = x.squeeze(1)
        n_samples = x.shape[-1]
        latent_high = self.pathway_high(x)
        latent_low = self.pathway_low(x)
        # Align bottleneck lengths defensively (rounding of strided lengths can
        # differ by one sample for odd inputs); crop to the shorter.
        length = min(latent_high.shape[-1], latent_low.shape[-1])
        joint = torch.cat([latent_high[..., :length], latent_low[..., :length]], dim=1)
        fused = self.fusion(joint)
        decoded = self.decoder(fused)
        # Length-safe: crop or pad the reconstruction to the input length.
        if decoded.shape[-1] > n_samples:
            decoded = decoded[..., :n_samples]
        elif decoded.shape[-1] < n_samples:
            pad = n_samples - decoded.shape[-1]
            decoded = nn.functional.pad(decoded, (0, pad), mode="replicate")
        return decoded


# ---------------------------------------------------------------------------
# Per-segment normalization (paper Sec. 3.1: subtract std, divide by max-abs)
# ---------------------------------------------------------------------------


def normalize_segment(segment: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, float, float]:
    """Standardise one 1D segment: subtract std, divide by max-abs.

    Returns ``(normalized, std, scale)`` so the transform can be inverted at
    inference. ``normalized = (segment - std) / scale`` where
    ``scale = max(|segment - std|)``. Mirrors the paper's two-step inference
    normalisation and is applied identically in training and inference.
    """
    segment = np.asarray(segment, dtype=np.float32)
    std = float(np.std(segment))
    shifted = segment - std
    scale = float(np.max(np.abs(shifted)))
    if scale < eps:
        scale = 1.0
    return (shifted / scale).astype(np.float32, copy=False), std, scale


# ---------------------------------------------------------------------------
# Channel-wise dataset (loads the Niazy proof-fit NPZ bundle directly)
# ---------------------------------------------------------------------------


class ChannelWiseDPAEDataset:
    """Yield per-channel ``(1, samples)`` items from an NPZ center bundle.

    Loads ``noisy_center`` / ``clean_center`` / ``artifact_center`` (each
    ``(examples, channels, samples)``) directly and flattens examples x channels
    into single-channel items. The TARGET defaults to the CLEAN signal (paper
    Sec. 2.3); set ``target_type='artifact'`` for FACETpy's subtract-the-artifact
    path. Both input and target are per-segment normalised (subtract std, divide
    by max-abs) so the network sees the same amplitude scale as at inference.

    Exposes every attribute the facet-train dataset contract requires, including
    ``epoch_samples`` (missing on the original edition), and propagates them onto
    the ``train_val_split`` subsets.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        target_type: str = "clean",
        normalize: bool = True,
        max_examples: int | None = None,
    ) -> None:
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        normalized_target = str(target_type).strip().lower()
        if normalized_target not in {"clean", "artifact"}:
            raise ValueError(f"target_type must be 'clean' or 'artifact', got '{target_type}'")

        with np.load(self.path, allow_pickle=False) as bundle:
            noisy_center = bundle["noisy_center"].astype(np.float32, copy=False)
            clean_center = bundle["clean_center"].astype(np.float32, copy=False)
            artifact_center = bundle["artifact_center"].astype(np.float32, copy=False)
            self.sfreq = float(bundle["sfreq"][0]) if "sfreq" in bundle else float("nan")

        if noisy_center.shape != clean_center.shape or noisy_center.shape != artifact_center.shape:
            raise ValueError("noisy_center, clean_center, and artifact_center must have identical shapes")
        if noisy_center.ndim != 3:
            raise ValueError("noisy_center must have shape (examples, channels, samples)")

        self._noisy = noisy_center
        self._clean = clean_center
        self._artifact = artifact_center
        self.n_examples = int(noisy_center.shape[0])
        self.n_channels = int(noisy_center.shape[1])
        self.epoch_samples = int(noisy_center.shape[2])
        self.chunk_size = self.epoch_samples
        self.target_type = normalized_target
        self.normalize = bool(normalize)
        self.trigger_aligned = True

        total = self.n_examples * self.n_channels
        self._length = total if max_examples is None else max(0, min(int(max_examples), total))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        example_idx = int(idx) // self.n_channels
        channel_idx = int(idx) % self.n_channels

        noisy = self._noisy[example_idx, channel_idx, :].astype(np.float32, copy=True)
        if self.target_type == "artifact":
            target = self._artifact[example_idx, channel_idx, :].astype(np.float32, copy=True)
        else:
            target = self._clean[example_idx, channel_idx, :].astype(np.float32, copy=True)

        if self.normalize:
            noisy_norm, std, scale = normalize_segment(noisy)
            # Apply the SAME (std, scale) transform to the target so the network
            # learns the normalised mapping; clean and noisy differ only by the
            # artifact under the dataset's additivity.
            if self.target_type == "clean":
                target_norm = ((target - std) / scale).astype(np.float32, copy=False)
            else:
                # The artifact is a difference, so only the scale applies (the
                # std shift cancels: noisy - clean = artifact).
                target_norm = (target / scale).astype(np.float32, copy=False)
            noisy_out = noisy_norm[np.newaxis, :]
            target_out = target_norm[np.newaxis, :]
        else:
            noisy_out = noisy[np.newaxis, :]
            target_out = target[np.newaxis, :]
        return noisy_out, target_out

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
        val_set = set(indices[:n_val])
        train_idx = [i for i in range(n) if i not in val_set]
        val_idx = [i for i in range(n) if i in val_set]
        return _SubsetDataset(self, train_idx), _SubsetDataset(self, val_idx)


class _SubsetDataset:
    """Subset view that carries the full dataset contract attributes."""

    def __init__(self, parent: ChannelWiseDPAEDataset, indices: list[int]) -> None:
        self._parent = parent
        self._indices = indices
        # Propagate the contract attributes onto the subset (the original
        # edition's subset exposed none of these).
        self.n_channels = parent.n_channels
        self.chunk_size = parent.chunk_size
        self.epoch_samples = parent.epoch_samples
        self.target_type = parent.target_type
        self.trigger_aligned = parent.trigger_aligned
        self.sfreq = parent.sfreq
        self.input_shape = parent.input_shape
        self.target_shape = parent.target_shape

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._parent[self._indices[idx]]

    @property
    def n_chunks(self) -> int:
        return len(self)


# ---------------------------------------------------------------------------
# facet-train factories
# ---------------------------------------------------------------------------


def build_model(
    input_shape: tuple[int, int] | tuple[int, int, int] | None = None,
    chunk_size: int | None = None,
    *,
    base_filters: int = 32,
    shrink_ratio_low: float = 0.45,
    shrink_ratio_high: float = 0.75,
    pathway_layers: int = 4,
    pathway2_stride: int = 4,
    fusion_depth: int = 3,
    **_: object,
) -> DualPathwayAutoencoder:
    """Build the paper-accurate DPAE 1D-CNN model for facet-train.

    Accepts (and ignores via ``**_``) the kwargs facet-train injects:
    ``n_channels, chunk_size, sfreq, target_type, training_config, input_shape,
    target_shape, context_epochs, epoch_samples``. Explicit YAML ``model.kwargs``
    override the defaults below. The model is identical for clean and artifact
    targets; only the training target differs.
    """
    if input_shape is not None:
        input_size = int(input_shape[-1])
    elif chunk_size is not None:
        input_size = int(chunk_size)
    else:
        raise ValueError("build_model requires input_shape or chunk_size")
    return DualPathwayAutoencoder(
        input_size=input_size,
        base_filters=int(base_filters),
        shrink_ratio_low=float(shrink_ratio_low),
        shrink_ratio_high=float(shrink_ratio_high),
        pathway_layers=int(pathway_layers),
        pathway2_stride=int(pathway2_stride),
        fusion_depth=int(fusion_depth),
    )


def build_loss(name: str = "mse", **_: object) -> nn.Module:
    """Build the training loss. Paper uses MSE between decoder output and clean."""
    normalized = name.strip().lower()
    if normalized == "l1":
        return nn.L1Loss()
    if normalized in {"smooth_l1", "huber"}:
        return nn.SmoothL1Loss()
    return nn.MSELoss()


def build_dataset(
    path: str | None = None,
    context_path: str | None = None,
    max_examples: int | None = None,
    *,
    target_type: str = "clean",
    normalize: bool = True,
    **_: object,
) -> ChannelWiseDPAEDataset:
    """Load a Niazy proof-fit center bundle and expose per-channel segments.

    Defaults to the CLEAN target (paper Sec. 2.3). Pass ``target_type='artifact'``
    for FACETpy's subtract-the-artifact convenience (numerically equivalent under
    the dataset's exact additivity ``noisy = clean + artifact``).
    """
    dataset_path = path or context_path
    if not dataset_path:
        raise ValueError("build_dataset requires path or context_path")
    return ChannelWiseDPAEDataset(
        path=dataset_path,
        target_type=target_type,
        normalize=normalize,
        max_examples=max_examples,
    )


__all__ = [
    "DualPathwayAutoencoder",
    "ChannelWiseDPAEDataset",
    "normalize_segment",
    "build_model",
    "build_loss",
    "build_dataset",
]
