"""Training factories for the DenoiseMamba (ConvSSD) gradient artifact denoiser.

DenoiseMamba is a Section 6.2 architecture from
``docs/research/dl_eeg_gradient_artifacts.pdf``. It stacks ConvSSD blocks that
combine a local 1D convolution with a Mamba-style selective state space layer.
The selective scan is implemented in pure PyTorch so the model is portable to
CPU for tests and trains on the GPU fleet without depending on the
``mamba-ssm`` CUDA package. See ``documentation/research_notes.md`` for the
underlying motivation and references.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from facet.training.dataset import NPZContextArtifactDataset


# ---------------------------------------------------------------------------
# Selective state space (Mamba-1 style) implemented in pure PyTorch.
# ---------------------------------------------------------------------------


class SelectiveSSM(nn.Module):
    """Mamba-1 style selective state space layer in pure PyTorch.

    Implements the discretised recurrence

        h_t = exp(delta_t * A) * h_{t-1} + (delta_t * B_t) * x_t
        y_t = C_t * h_t + D * x_t

    with input-dependent ``B``, ``C`` and ``delta``. The scan is sequential in
    time. Sequence lengths in this model are short (default 512), so the
    pure-PyTorch loop is fast enough on the RTX 5090 and avoids depending on
    the ``mamba-ssm`` CUDA kernel.

    Parameters
    ----------
    d_inner : int
        Number of channels carried through the SSM (= ``expand * d_model``).
    d_state : int
        Hidden state dimension per channel.
    dt_rank : int or None
        Rank of the projection that produces the ``delta`` time-step. ``None``
        falls back to ``ceil(d_inner / 16)`` as in the original Mamba code.
    """

    def __init__(self, d_inner: int, d_state: int = 16, dt_rank: int | None = None) -> None:
        super().__init__()
        self.d_inner = int(d_inner)
        self.d_state = int(d_state)
        self.dt_rank = int(dt_rank) if dt_rank is not None else max(1, math.ceil(self.d_inner / 16))

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        a_init = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, self.d_state)
        self.A_log = nn.Parameter(torch.log(a_init.clone()))
        self.D = nn.Parameter(torch.ones(self.d_inner, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the selective scan.

        Parameters
        ----------
        x : torch.Tensor, shape (B, L, d_inner)
            Pre-conv inner activations.

        Returns
        -------
        torch.Tensor, shape (B, L, d_inner)
        """
        batch, length, d_inner = x.shape
        assert d_inner == self.d_inner

        x_dbl = self.x_proj(x)
        delta_unproj, B, C = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        delta = nn.functional.softplus(self.dt_proj(delta_unproj))

        A = -torch.exp(self.A_log.float())

        delta_a = torch.einsum("bld,dn->bldn", delta, A)
        delta_b_x = torch.einsum("bld,bln,bld->bldn", delta, B, x)
        delta_a_exp = torch.exp(delta_a)

        state = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(length):
            state = delta_a_exp[:, t] * state + delta_b_x[:, t]
            y_t = torch.einsum("bdn,bn->bd", state, C[:, t])
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)

        return y + x * self.D


# ---------------------------------------------------------------------------
# ConvSSD block: 1D conv + selective SSM with pre-norm residual connections.
# ---------------------------------------------------------------------------


class MambaBlock(nn.Module):
    """A Mamba block with input projection, depthwise conv, SSM, and gate."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_inner = int(expand) * self.d_model
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=int(d_conv),
            padding=int(d_conv) - 1,
            groups=self.d_inner,
            bias=True,
        )
        self.ssm = SelectiveSSM(d_inner=self.d_inner, d_state=int(d_state))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.in_proj(x)
        x_in, gate = x_proj.chunk(2, dim=-1)
        length = x_in.shape[1]
        x_conv = self.conv1d(x_in.transpose(1, 2))[..., :length].transpose(1, 2)
        x_conv = nn.functional.silu(x_conv)
        y = self.ssm(x_conv)
        y = y * nn.functional.silu(gate)
        y = self.dropout(y)
        return self.out_proj(y)


class ConvSSDBlock(nn.Module):
    """Pre-norm ConvSSD block: depthwise conv + selective SSM with residuals."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


# ---------------------------------------------------------------------------
# Full DenoiseMamba model: predicts the artifact for a single-channel epoch.
# ---------------------------------------------------------------------------


class DenoiseMamba(nn.Module):
    """ConvSSD-based artifact predictor for single-channel EEG segments.

    Input shape  : ``(batch, 1, samples)``
    Output shape : ``(batch, 1, samples)`` — predicted gradient artifact.
    """

    def __init__(
        self,
        epoch_samples: int = 512,
        d_model: int = 64,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        n_blocks: int = 4,
        dropout: float = 0.1,
        input_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        if epoch_samples <= 0:
            raise ValueError("epoch_samples must be positive")
        if n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")
        if input_kernel_size < 1 or input_kernel_size % 2 == 0:
            raise ValueError("input_kernel_size must be a positive odd integer")

        self.epoch_samples = int(epoch_samples)
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.expand = int(expand)
        self.d_conv = int(d_conv)
        self.n_blocks = int(n_blocks)
        self.input_kernel_size = int(input_kernel_size)

        self.input_proj = nn.Conv1d(
            in_channels=1,
            out_channels=self.d_model,
            kernel_size=self.input_kernel_size,
            padding=self.input_kernel_size // 2,
        )
        self.blocks = nn.ModuleList(
            [
                ConvSSDBlock(
                    d_model=self.d_model,
                    d_state=self.d_state,
                    expand=self.expand,
                    d_conv=self.d_conv,
                    dropout=float(dropout),
                )
                for _ in range(self.n_blocks)
            ]
        )
        self.norm = nn.LayerNorm(self.d_model)
        self.output_proj = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=1,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != 1:
            raise ValueError(
                f"DenoiseMamba expects shape (batch, 1, samples); got {tuple(x.shape)}"
            )
        h = self.input_proj(x)
        h = h.transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        h = h.transpose(1, 2)
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# Dataset wrapper: single-epoch channel-wise denoising over the Niazy bundle.
# ---------------------------------------------------------------------------


class _SubsetDataset:
    def __init__(self, parent: "ChannelWiseSingleEpochArtifactDataset", indices: list[int]) -> None:
        self._parent = parent
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._parent[self._indices[idx]]


class ChannelWiseSingleEpochArtifactDataset:
    """Per-channel single-epoch dataset wrapping the Niazy proof-fit NPZ bundle.

    Yields ``(noisy, target)`` pairs of shape ``(1, samples)`` each. Internally
    iterates over ``noisy_center`` / ``artifact_center`` arrays of shape
    ``(N, n_channels, samples)``, expanding each example into ``n_channels``
    individual channel-wise items.
    """

    def __init__(
        self,
        base_dataset: Any,
        *,
        demean_input: bool = True,
        demean_target: bool = True,
        max_examples: int | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.demean_input = bool(demean_input)
        self.demean_target = bool(demean_target)

        n_base = len(base_dataset)
        if n_base == 0:
            raise ValueError("base dataset must contain at least one example")
        first_noisy, first_target = base_dataset[0]
        if first_noisy.ndim != 2:
            raise ValueError("base dataset input must have shape (channels, samples)")
        if first_target.ndim != 2:
            raise ValueError("base dataset target must have shape (channels, samples)")

        self.n_channels = int(first_noisy.shape[0])
        self.epoch_samples = int(first_noisy.shape[1])
        self.chunk_size = self.epoch_samples
        self.target_type = "artifact"
        self.trigger_aligned = True
        self.sfreq = float(getattr(base_dataset, "sfreq", float("nan")))

        total = n_base * self.n_channels
        self._length = total if max_examples is None else max(0, min(int(max_examples), total))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        base_idx = int(idx) // self.n_channels
        channel_idx = int(idx) % self.n_channels
        noisy, target = self.base_dataset[base_idx]
        noisy_out = noisy[channel_idx : channel_idx + 1, :].astype(np.float32, copy=True)
        target_out = target[channel_idx : channel_idx + 1, :].astype(np.float32, copy=True)
        if self.demean_input:
            noisy_out -= noisy_out.mean(axis=-1, keepdims=True)
        if self.demean_target:
            target_out -= target_out.mean(axis=-1, keepdims=True)
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
        val_idx = set(indices[:n_val])
        train_idx = [i for i in range(n) if i not in val_idx]
        val_idx_list = [i for i in range(n) if i in val_idx]
        return _SubsetDataset(self, train_idx), _SubsetDataset(self, val_idx_list)


class _SingleEpochCenterAdapter:
    """Adapter that exposes ``noisy_center`` / ``artifact_center`` as (channels, samples)."""

    def __init__(self, npz_dataset: NPZContextArtifactDataset) -> None:
        self._npz = npz_dataset
        self.sfreq = float(getattr(npz_dataset, "sfreq", float("nan")))

    def __len__(self) -> int:
        return len(self._npz)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        noisy_context, artifact_center = self._npz[idx]
        center_idx = noisy_context.shape[0] // 2
        noisy_center = noisy_context[center_idx]
        return noisy_center.astype(np.float32, copy=False), artifact_center.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Public factories consumed by facet-train.
# ---------------------------------------------------------------------------


def build_model(
    input_shape: tuple[int, int] | tuple[int, int, int] | None = None,
    target_shape: tuple[int, int] | None = None,
    epoch_samples: int | None = None,
    d_model: int = 64,
    d_state: int = 16,
    expand: int = 2,
    d_conv: int = 4,
    n_blocks: int = 4,
    dropout: float = 0.1,
    input_kernel_size: int = 7,
    **_: object,
) -> DenoiseMamba:
    """Construct a DenoiseMamba module from CLI / dataset metadata."""
    if epoch_samples is None and input_shape is not None:
        epoch_samples = int(input_shape[-1])
    if epoch_samples is None:
        raise ValueError("build_model requires epoch_samples or input_shape")
    return DenoiseMamba(
        epoch_samples=int(epoch_samples),
        d_model=d_model,
        d_state=d_state,
        expand=expand,
        d_conv=d_conv,
        n_blocks=n_blocks,
        dropout=dropout,
        input_kernel_size=input_kernel_size,
    )


def build_loss(name: str = "mse"):
    """Loss factory matching the cascaded_context_dae contract."""
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
    demean_input: bool = True,
    demean_target: bool = True,
    **_: object,
) -> ChannelWiseSingleEpochArtifactDataset:
    """Construct the channel-wise single-epoch dataset from a Niazy NPZ bundle."""
    dataset_path = Path(path or context_path or "").expanduser()
    if not str(dataset_path):
        raise ValueError("build_dataset requires path or context_path")
    base = NPZContextArtifactDataset(
        path=dataset_path,
        input_key="noisy_context",
        target_key="artifact_center",
        demean_input=False,
        demean_target=False,
    )
    center_adapter = _SingleEpochCenterAdapter(base)
    return ChannelWiseSingleEpochArtifactDataset(
        center_adapter,
        demean_input=demean_input,
        demean_target=demean_target,
        max_examples=max_examples,
    )
