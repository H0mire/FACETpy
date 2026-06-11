"""Paper-accurate training factories for the Demucs gradient-artifact model.

This is the *paper-accurate edition* of the time-domain Demucs (Defossez,
Usunier, Bottou, Bach 2019, "Music Source Separation in the Waveform Domain",
arXiv:1911.13254), adapted to predict the multi-epoch fMRI gradient-artifact
context per EEG channel.

Compared with the original ``facet.models.demucs`` package this edition is more
faithful to the source paper on four points, all of which keep the model
CPU-runnable and compatible with the facet-train factory contract:

1. **Arbitrary-length handling (Sec 4, reference impl).** The encoder/decoder
   is now length-agnostic via a ``valid_length`` helper that pads the input up
   to the next multiple of ``stride ** depth`` and center-crops each encoder
   skip and the final output. The original ``forward`` summed skips with no
   length alignment and crashed on any input length not divisible by
   ``stride ** depth`` (verified: ``depth=2``, length 700 -> RuntimeError).

2. **2x resampling trick (Sec 4.1 "Resampling").** Optional ``resample`` factor
   (default 2) upsamples the waveform before the encoder and downsamples the
   output after the decoder, inside the forward pass (and therefore inside the
   end-to-end loss). Implemented with a dependency-free, TorchScript-traceable
   sinc/Kaiser FIR via ``F.conv1d`` so it runs on CPU and traces cleanly.

3. **Init weight rescaling (Sec 4.3).** Keeps the paper's ``alpha = std(w)/a``,
   ``w' = w/sqrt(alpha)`` (``a = 0.1``) but no longer zeroes conv biases, which
   the paper does not specify.

4. **Principled depth.** ``auto_depth`` caps depth at the maximum that keeps the
   (optionally upsampled) bottleneck length >= 1, so the K=8/S=4 blocks never
   collapse the short EEG context. ``initial_channels`` stays at the paper-best
   64 by default.

See ``README.md`` and ``documentation/paper_accuracy_review.md`` for the full
discrepancy table and EEG-fMRI applicability assessment.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _glu_channels(channels: int) -> int:
    return 2 * channels


def _center_crop_1d(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
    """Center-crop the last dim of ``tensor`` to ``target_length``.

    Mirrors the official Demucs reference, which center-trims encoder skips so
    they align with the (shorter) decoder feature maps. If the tensor is shorter
    than ``target_length`` it is right-padded with zeros so the sum still works
    (rare; only on degenerate tiny lengths).
    """
    length = tensor.shape[-1]
    if length == target_length:
        return tensor
    if length > target_length:
        start = (length - target_length) // 2
        return tensor[..., start : start + target_length]
    pad = target_length - length
    return F.pad(tensor, (0, pad))


def _build_kaiser_sinc_kernel(zeros: int, factor: int, rolloff: float = 0.945) -> torch.Tensor:
    """Build a windowed-sinc (Kaiser) FIR low-pass kernel for ``factor``x resampling.

    This is a small, dependency-free version of the band-limited sinc filter the
    Demucs reference uses for its 2x resampling trick. The kernel is normalised
    so that a constant signal passes through unchanged and traces cleanly to
    TorchScript (only ``torch`` ops, no Python control flow in ``forward``).

    Parameters
    ----------
    zeros : int
        Half-width of the sinc in output samples (the FIR is ``2*zeros*factor+1``
        taps long). Small values keep the convolution cheap on CPU.
    factor : int
        Integer up/down-sampling factor (2 for the paper's trick).
    rolloff : float
        Cutoff as a fraction of the Nyquist frequency to suppress ringing.
    """
    half_width = zeros * factor
    idx = torch.arange(-half_width, half_width + 1, dtype=torch.float64)
    t = idx / factor
    cutoff = rolloff / factor
    # sinc(2*cutoff*t) lowpass, scaled to unity DC gain via the (2*cutoff) factor.
    sinc = torch.where(
        t == 0,
        torch.tensor(2.0 * cutoff, dtype=torch.float64),
        torch.sin(2.0 * math.pi * cutoff * t) / (math.pi * t),
    )
    # Kaiser window (beta tied to the half-width for a reasonable transition band).
    beta = 6.0
    n = torch.arange(0, idx.numel(), dtype=torch.float64)
    alpha = (idx.numel() - 1) / 2.0
    ratio = (n - alpha) / alpha
    ratio = torch.clamp(ratio, -1.0, 1.0)
    window = torch.special.i0(beta * torch.sqrt(1.0 - ratio**2)) / torch.special.i0(torch.tensor(beta, dtype=torch.float64))
    kernel = sinc * window
    kernel = kernel / kernel.sum()
    return kernel.to(torch.float32)


class _Resampler(torch.nn.Module):
    """Band-limited integer up/down-sampler used by the Demucs resampling trick.

    ``upsample`` zero-stuffs by ``factor`` and convolves with a sinc/Kaiser FIR
    (gain corrected by ``factor`` to preserve amplitude). ``downsample`` low-pass
    filters and decimates by ``factor``. Implemented purely with ``F.conv1d`` so
    it is CPU-cheap and TorchScript-traceable; no external dependency.
    """

    def __init__(self, factor: int, zeros: int = 6) -> None:
        super().__init__()
        self.factor = int(factor)
        if self.factor < 1:
            raise ValueError("resample factor must be >= 1")
        kernel = _build_kaiser_sinc_kernel(zeros=zeros, factor=max(self.factor, 1))
        self.pad = (kernel.numel() - 1) // 2
        # (out_channels=1, in_channels=1, kernel_width)
        self.register_buffer("kernel", kernel.view(1, 1, -1))

    def upsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.factor == 1:
            return x
        batch, channels, length = x.shape
        flat = x.reshape(batch * channels, 1, length)
        stuffed = F.conv_transpose1d(flat, torch.ones(1, 1, 1, device=x.device, dtype=x.dtype), stride=self.factor)
        # conv_transpose with a 1-tap unit kernel zero-stuffs by stride.
        filtered = F.conv1d(stuffed, self.kernel.to(x.dtype) * self.factor, padding=self.pad)
        out_length = length * self.factor
        filtered = _center_crop_1d(filtered, out_length)
        return filtered.reshape(batch, channels, out_length)

    def downsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.factor == 1:
            return x
        batch, channels, length = x.shape
        flat = x.reshape(batch * channels, 1, length)
        filtered = F.conv1d(flat, self.kernel.to(x.dtype), padding=self.pad)
        filtered = _center_crop_1d(filtered, length)
        decimated = filtered[..., :: self.factor]
        out_length = decimated.shape[-1]
        return decimated.reshape(batch, channels, out_length)


class _EncoderBlock(torch.nn.Module):
    """Demucs encoder block: Conv1d(K=8,S=4)+ReLU then Conv1d(K=1)+GLU.

    Faithful to Sec 4 / Fig. 2 of arXiv:1911.13254. Uses ``padding=0`` (the
    reference performs no input-aligning padding inside the block; the model-level
    ``valid_length`` padding handles arbitrary lengths instead).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride)
        self.relu = torch.nn.ReLU()
        self.conv_glu = torch.nn.Conv1d(out_channels, _glu_channels(out_channels), kernel_size=1)
        self.glu = torch.nn.GLU(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv(x))
        x = self.glu(self.conv_glu(x))
        return x


class _DecoderBlock(torch.nn.Module):
    """Demucs decoder block: Conv1d(K=3)+GLU then ConvTranspose1d(K=8,S=4)+ReLU.

    The final block omits the trailing ReLU so the output can be signed (the
    gradient-artifact waveform can have either polarity). Faithful to the
    paper's final-block linearity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        is_last: bool = False,
    ) -> None:
        super().__init__()
        self.conv_glu = torch.nn.Conv1d(in_channels, _glu_channels(in_channels), kernel_size=3, padding=1)
        self.glu = torch.nn.GLU(dim=1)
        self.deconv = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)
        self.is_last = is_last
        self.activation = torch.nn.ReLU() if not is_last else torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.glu(self.conv_glu(x))
        x = self.deconv(x)
        x = self.activation(x)
        return x


class Demucs(torch.nn.Module):
    """Paper-accurate time-domain Demucs U-Net for waveform-to-waveform regression.

    Faithful to arXiv:1911.13254 in encoder/decoder structure, GLU gating,
    BiLSTM bottleneck, summed U-Net skips, init weight-rescaling (no batch
    norm), and the optional 2x resampling trick (Sec 4.1). It is length-agnostic
    via ``valid_length`` padding + center-trim, so it never crashes on input
    lengths that are not a multiple of ``stride ** depth`` (the bug in the
    original edition).

    Parameters
    ----------
    in_channels : int
        Number of input channels (default: 1, channel-wise inference).
    out_channels : int
        Number of output channels per source (default: 1, predicts the artifact).
    n_sources : int
        Number of output sources (default: 1, artifact-only — the EEG adaptation
        of the paper's S-source sum).
    depth : int
        Number of encoder/decoder blocks. Capped by ``auto_depth`` to a safe
        value for the (optionally upsampled) input length.
    initial_channels : int
        Channels of the first encoder block C_1 (default: 64, the paper best).
    kernel_size : int
        Convolution kernel size in the encoder/decoder (default: 8).
    stride : int
        Convolution stride in the encoder/decoder (default: 4).
    lstm_layers : int
        Number of bidirectional LSTM layers at the bottleneck (default: 2).
    rescale : float
        Target standard-deviation ratio ``a`` for the init weight-rescaling trick
        (default: 0.1, matching the paper's reference value).
    resample : int
        Integer 2x-style resampling factor for the Sec 4.1 trick (default: 2;
        1 disables). The input is upsampled by this factor before the encoder and
        downsampled by it after the decoder, restoring the original length.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_sources: int = 1,
        depth: int = 4,
        initial_channels: int = 64,
        kernel_size: int = 8,
        stride: int = 4,
        lstm_layers: int = 2,
        rescale: float = 0.1,
        resample: int = 2,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if kernel_size <= stride:
            raise ValueError("kernel_size must be greater than stride for the U-Net to work")
        if resample < 1:
            raise ValueError("resample must be >= 1")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_sources = int(n_sources)
        self.depth = int(depth)
        self.initial_channels = int(initial_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.lstm_layers = int(lstm_layers)
        self.resample = int(resample)

        self.resampler = _Resampler(self.resample) if self.resample > 1 else None

        encoder_blocks: list[_EncoderBlock] = []
        decoder_blocks: list[_DecoderBlock] = []
        channels = [self.in_channels]
        for level in range(self.depth):
            in_c = channels[-1]
            out_c = self.initial_channels if level == 0 else channels[-1] * 2
            channels.append(out_c)
            encoder_blocks.append(_EncoderBlock(in_c, out_c, self.kernel_size, self.stride))

        for level in range(self.depth):
            in_c = channels[-1 - level]
            out_c = channels[-2 - level]
            is_last = level == self.depth - 1
            if is_last:
                out_c = self.out_channels * self.n_sources
            decoder_blocks.append(_DecoderBlock(in_c, out_c, self.kernel_size, self.stride, is_last=is_last))

        self.encoder = torch.nn.ModuleList(encoder_blocks)
        self.decoder = torch.nn.ModuleList(decoder_blocks)

        bottleneck_channels = channels[-1]
        self.lstm = torch.nn.LSTM(
            input_size=bottleneck_channels,
            hidden_size=bottleneck_channels,
            num_layers=self.lstm_layers,
            bidirectional=True,
            batch_first=False,
        )
        self.lstm_projection = torch.nn.Linear(2 * bottleneck_channels, bottleneck_channels)

        self._rescale_init_weights(rescale)

    def valid_length(self, length: int) -> int:
        """Round ``length`` up to the next multiple of ``stride ** depth``.

        Mirrors the official Demucs reference's ``valid_length`` so encoder
        downsamplings always divide evenly and decoder upsamplings reconstruct a
        length that is >= the original (the model then center-crops back).
        """
        divisor = self.stride**self.depth
        return int(math.ceil(length / divisor) * divisor)

    def _rescale_init_weights(self, target_std: float) -> None:
        """Sec 4.3 weight rescaling: w' = w / sqrt(std(w) / a). No bias zeroing."""
        for module in self.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                with torch.no_grad():
                    std = module.weight.std().clamp(min=1e-12).item()
                    scale = (std / target_std) ** 0.5
                    module.weight.div_(scale)
                    # NB: the paper's scheme (Sec 4.3) rescales weights only and
                    # is silent about biases, so we leave PyTorch's default bias
                    # init untouched (the original edition zeroed it).

    def _run_unet(self, x: torch.Tensor) -> torch.Tensor:
        original_length = x.shape[-1]
        padded_length = self.valid_length(original_length)
        if padded_length != original_length:
            x = F.pad(x, (0, padded_length - original_length))

        skips: list[torch.Tensor] = []
        out = x
        for block in self.encoder:
            out = block(out)
            skips.append(out)

        bottleneck = out.permute(2, 0, 1).contiguous()
        bottleneck, _ = self.lstm(bottleneck)
        bottleneck = self.lstm_projection(bottleneck)
        out = bottleneck.permute(1, 2, 0).contiguous()
        # Deepest decoder input = e_L + LSTM output (Fig. 2b). Center-crop the
        # skip to the (possibly different) bottleneck length first.
        out = out + _center_crop_1d(skips[-1], out.shape[-1])

        for level, block in enumerate(self.decoder):
            if level > 0:
                skip = skips[-1 - level]
                out = out + _center_crop_1d(skip, out.shape[-1])
            out = block(out)

        # Restore the exact original length (decoder output >= original_length).
        return _center_crop_1d(out, original_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.resampler is not None:
            input_length = x.shape[-1]
            x = self.resampler.upsample(x)
            x = self._run_unet(x)
            x = self.resampler.downsample(x)
            x = _center_crop_1d(x, input_length)
            return x
        return self._run_unet(x)


class FlatContextArtifactDataset:
    """Per-channel flattened multi-epoch context dataset for waveform models.

    Loads the Niazy proof-fit ``.npz`` bundle directly. Each item is
    ``(noisy, artifact)`` where both arrays have shape
    ``(1, context_epochs * epoch_samples)`` — the trigger-defined epochs of a
    single channel concatenated into one waveform. Semantics match the original
    edition (per-channel flatten, demean) so the facet-train contract and NPZ
    format are unchanged.

    A FACETpy-appropriate, off-by-default augmentation substitutes for the
    paper's music-specific augmentations: an optional small random gain and small
    circular time-shift of the concatenated window (consistent with the
    shift-trick intuition; the music pitch/tempo/remix/channel-swap augmentations
    do not transfer to single-channel gradient artifacts — see the review doc).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        context_epochs: int = 7,
        demean_input: bool = True,
        demean_target: bool = True,
        max_examples: int | None = None,
        augment: bool = False,
        augment_gain_range: tuple[float, float] = (0.9, 1.1),
        augment_max_shift: int = 0,
        augment_seed: int = 0,
    ) -> None:
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.context_epochs = int(context_epochs)
        self.demean_input = bool(demean_input)
        self.demean_target = bool(demean_target)
        if self.context_epochs < 1 or self.context_epochs % 2 == 0:
            raise ValueError("context_epochs must be a positive odd integer")

        self.augment = bool(augment)
        self.augment_gain_range = (float(augment_gain_range[0]), float(augment_gain_range[1]))
        self.augment_max_shift = int(augment_max_shift)
        self._rng = np.random.default_rng(int(augment_seed))

        with np.load(self.path, allow_pickle=True) as bundle:
            noisy = bundle["noisy_context"].astype(np.float32, copy=False)
            artifact = bundle["artifact_context"].astype(np.float32, copy=False)
            self.sfreq = float(bundle["sfreq"][0]) if "sfreq" in bundle else float("nan")

        if noisy.shape != artifact.shape:
            raise ValueError("noisy_context and artifact_context must share the same shape")
        if noisy.ndim != 4:
            raise ValueError("noisy_context must have shape (examples, context_epochs, channels, samples)")
        if noisy.shape[1] != self.context_epochs:
            raise ValueError(f"expected {self.context_epochs} context epochs in the bundle, got {noisy.shape[1]}")

        self._noisy = noisy
        self._artifact = artifact
        n_examples = int(noisy.shape[0])
        self.n_channels = int(noisy.shape[2])
        self.epoch_samples = int(noisy.shape[3])
        self.total_samples = self.context_epochs * self.epoch_samples
        self.chunk_size = self.total_samples
        self.target_type = "artifact"
        self.trigger_aligned = True

        total = n_examples * self.n_channels
        self._length = total if max_examples is None else max(0, min(int(max_examples), total))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        example_idx = int(idx) // self.n_channels
        channel_idx = int(idx) % self.n_channels
        noisy_flat = (
            self._noisy[example_idx, :, channel_idx, :].reshape(1, self.total_samples).astype(np.float32, copy=True)
        )
        target_flat = (
            self._artifact[example_idx, :, channel_idx, :].reshape(1, self.total_samples).astype(np.float32, copy=True)
        )
        if self.augment:
            gain = float(self._rng.uniform(*self.augment_gain_range))
            noisy_flat *= gain
            target_flat *= gain
            if self.augment_max_shift > 0:
                shift = int(self._rng.integers(-self.augment_max_shift, self.augment_max_shift + 1))
                if shift != 0:
                    noisy_flat = np.roll(noisy_flat, shift, axis=-1)
                    target_flat = np.roll(target_flat, shift, axis=-1)
        if self.demean_input:
            noisy_flat -= noisy_flat.mean(axis=-1, keepdims=True)
        if self.demean_target:
            target_flat -= target_flat.mean(axis=-1, keepdims=True)
        return noisy_flat, target_flat

    @property
    def input_shape(self) -> tuple[int, int]:
        return (1, self.total_samples)

    @property
    def target_shape(self) -> tuple[int, int]:
        return (1, self.total_samples)

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
    def __init__(self, parent: FlatContextArtifactDataset, indices: list[int]) -> None:
        self._parent = parent
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._parent[self._indices[idx]]


def _resolve_total_samples(
    input_shape: tuple[int, ...] | None,
    epoch_samples: int | None,
    context_epochs: int,
) -> int:
    if input_shape is not None:
        if len(input_shape) == 2:
            return int(input_shape[1])
        if len(input_shape) >= 3:
            return int(input_shape[-1]) * int(input_shape[-3] if len(input_shape) >= 3 else 1)
    if epoch_samples is None:
        raise ValueError("Demucs build_model requires input_shape or epoch_samples to infer the input length")
    return int(epoch_samples) * int(context_epochs)


def _max_safe_depth(total_samples: int, kernel_size: int, stride: int) -> int:
    """Largest depth keeping the encoder bottleneck length >= 1.

    Each encoder block applies ``Conv1d(K, S, padding=0)``:
    ``L_out = floor((L_in - K) / S) + 1``. We keep adding levels while the
    bottleneck stays >= 1 sample.
    """
    length = int(total_samples)
    depth = 0
    while True:
        next_length = (length - kernel_size) // stride + 1
        if next_length < 1:
            break
        length = next_length
        depth += 1
    return max(depth, 1)


def build_model(
    input_shape: tuple[int, ...] | None = None,
    target_shape: tuple[int, ...] | None = None,
    context_epochs: int | None = None,
    epoch_samples: int | None = None,
    in_channels: int = 1,
    out_channels: int = 1,
    n_sources: int = 1,
    depth: int = 4,
    initial_channels: int = 64,
    kernel_size: int = 8,
    stride: int = 4,
    lstm_layers: int = 2,
    rescale: float = 0.1,
    resample: int = 2,
    auto_depth: bool = True,
    **_: object,
) -> Demucs:
    """Factory consumed by ``facet-train``.

    ``input_shape`` is injected by the CLI from the dataset's ``input_shape``
    property: for :class:`FlatContextArtifactDataset` it is ``(1, total_samples)``.

    With ``auto_depth=True`` (default) ``depth`` is capped at the maximum that
    keeps the (optionally upsampled) bottleneck length >= 1 sample, so the K=8/S=4
    blocks never collapse the short EEG context. Accepts (and ignores) the
    facet-train-injected ``n_channels``, ``chunk_size``, ``sfreq``,
    ``target_type``, ``training_config`` via ``**_``.
    """
    total_samples = _resolve_total_samples(input_shape, epoch_samples, context_epochs or 7)
    # The resampling trick runs the U-Net on the upsampled signal, so depth is
    # bounded by the upsampled length (one extra usable level, as the paper notes).
    effective_samples = total_samples * max(int(resample), 1)
    if auto_depth:
        depth = min(int(depth), _max_safe_depth(effective_samples, kernel_size, stride))
    else:
        safe = _max_safe_depth(effective_samples, kernel_size, stride)
        if depth > safe:
            raise ValueError(
                f"depth={depth} collapses input length {effective_samples} below 1 sample at the bottleneck "
                f"(max safe depth is {safe}); enable auto_depth or reduce depth."
            )
    return Demucs(
        in_channels=in_channels,
        out_channels=out_channels,
        n_sources=n_sources,
        depth=depth,
        initial_channels=initial_channels,
        kernel_size=kernel_size,
        stride=stride,
        lstm_layers=lstm_layers,
        rescale=rescale,
        resample=resample,
    )


def build_loss(name: str = "l1", **_: object) -> torch.nn.Module:
    """Reconstruction loss. L1 (Sec 4.2 default) unless overridden.

    The paper sums L1 over its S sources (Eq. 2); with the EEG adaptation S=1 the
    source-sum is trivially satisfied. ``mse`` and ``smooth_l1``/``huber`` are
    retained as the paper-mentioned ablation alternatives.
    """
    normalized = name.strip().lower()
    if normalized == "mse":
        return torch.nn.MSELoss()
    if normalized in {"smooth_l1", "huber"}:
        return torch.nn.SmoothL1Loss()
    return torch.nn.L1Loss()


def build_dataset(
    path: str | None = None,
    context_path: str | None = None,
    context_epochs: int = 7,
    max_examples: int | None = None,
    demean_input: bool = True,
    demean_target: bool = True,
    augment: bool = False,
    augment_gain_range: tuple[float, float] = (0.9, 1.1),
    augment_max_shift: int = 0,
    augment_seed: int = 0,
    **_: object,
) -> FlatContextArtifactDataset:
    dataset_path = path or context_path
    if not dataset_path:
        raise ValueError("build_dataset requires path or context_path")
    return FlatContextArtifactDataset(
        path=dataset_path,
        context_epochs=context_epochs,
        demean_input=demean_input,
        demean_target=demean_target,
        max_examples=max_examples,
        augment=augment,
        augment_gain_range=augment_gain_range,
        augment_max_shift=augment_max_shift,
        augment_seed=augment_seed,
    )
