"""Training factories for the Demucs gradient-artifact model.

Implements the original time-domain Demucs (Defossez et al. 2019, arXiv:1911.13254)
adapted to predict the 7-epoch fMRI gradient artifact context per EEG channel.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch



def _glu_channels(channels: int) -> int:
    return 2 * channels


class _EncoderBlock(torch.nn.Module):
    """Demucs encoder block: Conv1d(K=8,S=4)+ReLU then Conv1d(K=1)+GLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        padding = (kernel_size - stride) // 2
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
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
    artifact waveform can have either polarity).
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
        padding = (kernel_size - stride) // 2
        self.deconv = torch.nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.is_last = is_last
        self.activation = torch.nn.ReLU() if not is_last else torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.glu(self.conv_glu(x))
        x = self.deconv(x)
        x = self.activation(x)
        return x


class Demucs(torch.nn.Module):
    """Original time-domain Demucs U-Net for waveform-to-waveform regression.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default: 1, channel-wise inference).
    out_channels : int
        Number of output channels per source (default: 1, predicts the artifact).
    n_sources : int
        Number of output sources (default: 1, artifact-only).
    depth : int
        Number of encoder/decoder blocks (default: 4 for our 3584-sample input).
    initial_channels : int
        Channels of the first encoder block C_1 (default: 64).
    kernel_size : int
        Convolution kernel size in the encoder/decoder (default: 8).
    stride : int
        Convolution stride in the encoder/decoder (default: 4).
    lstm_layers : int
        Number of bidirectional LSTM layers at the bottleneck (default: 2).
    rescale : float
        Target standard-deviation ratio for the init weight-rescaling trick
        (default: 0.1, matching the paper's reference value).
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
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if kernel_size <= stride:
            raise ValueError("kernel_size must be greater than stride for symmetric padding")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_sources = int(n_sources)
        self.depth = int(depth)
        self.initial_channels = int(initial_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.lstm_layers = int(lstm_layers)

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

    def _rescale_init_weights(self, target_std: float) -> None:
        for module in self.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                with torch.no_grad():
                    std = module.weight.std().clamp(min=1e-12).item()
                    scale = (std / target_std) ** 0.5
                    module.weight.div_(scale)
                    if module.bias is not None:
                        module.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        out = x
        for block in self.encoder:
            out = block(out)
            skips.append(out)

        bottleneck = out.permute(2, 0, 1).contiguous()
        bottleneck, _ = self.lstm(bottleneck)
        bottleneck = self.lstm_projection(bottleneck)
        out = bottleneck.permute(1, 2, 0).contiguous() + skips[-1]

        for level, block in enumerate(self.decoder):
            if level > 0:
                skip = skips[-1 - level]
                out = out + skip
            out = block(out)
        return out


class FlatContextArtifactDataset:
    """Per-channel flattened 7-epoch context dataset for waveform-domain models.

    Loads the Niazy proof-fit ``.npz`` bundle directly because the standard
    :class:`facet.training.dataset.NPZContextArtifactDataset` expects a
    ``(examples, channels, samples)`` target while we need the full 4D
    ``artifact_context`` array as the target.

    Each item is ``(noisy, artifact)`` where both arrays have shape
    ``(1, context_epochs * epoch_samples)`` — the seven trigger-defined epochs
    of a single channel concatenated into one waveform.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        context_epochs: int = 7,
        demean_input: bool = True,
        demean_target: bool = True,
        max_examples: int | None = None,
    ) -> None:
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.context_epochs = int(context_epochs)
        self.demean_input = bool(demean_input)
        self.demean_target = bool(demean_target)
        if self.context_epochs < 1 or self.context_epochs % 2 == 0:
            raise ValueError("context_epochs must be a positive odd integer")

        with np.load(self.path, allow_pickle=True) as bundle:
            noisy = bundle["noisy_context"].astype(np.float32, copy=False)
            artifact = bundle["artifact_context"].astype(np.float32, copy=False)
            self.sfreq = float(bundle["sfreq"][0]) if "sfreq" in bundle else float("nan")

        if noisy.shape != artifact.shape:
            raise ValueError("noisy_context and artifact_context must share the same shape")
        if noisy.ndim != 4:
            raise ValueError(
                "noisy_context must have shape (examples, context_epochs, channels, samples)"
            )
        if noisy.shape[1] != self.context_epochs:
            raise ValueError(
                f"expected {self.context_epochs} context epochs in the bundle, got {noisy.shape[1]}"
            )

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
        noisy_flat = self._noisy[example_idx, :, channel_idx, :].reshape(1, self.total_samples).astype(np.float32, copy=True)
        target_flat = self._artifact[example_idx, :, channel_idx, :].reshape(1, self.total_samples).astype(np.float32, copy=True)
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


def _validate_depth(total_samples: int, depth: int, kernel_size: int, stride: int) -> None:
    length = total_samples
    for level in range(depth):
        next_length = math.floor((length + (kernel_size - stride) - kernel_size) / stride) + 1
        if next_length < 1:
            raise ValueError(
                f"Demucs depth={depth} collapses input length {total_samples} below 1 sample "
                f"at level {level + 1} (current length {length})."
            )
        length = next_length


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
    **_: object,
) -> Demucs:
    """Factory consumed by ``facet-train``.

    ``input_shape`` is injected by the CLI from the dataset's ``input_shape``
    property: for :class:`FlatContextArtifactDataset` it is ``(1, total_samples)``.
    """
    total_samples = _resolve_total_samples(input_shape, epoch_samples, context_epochs or 7)
    _validate_depth(total_samples, depth, kernel_size, stride)
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
    )


def build_loss(name: str = "l1") -> torch.nn.Module:
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
    )
