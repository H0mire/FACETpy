"""Training factories and architecture for the spatiotemporal GNN model.

The architecture follows Yu/Yin/Zhu (2018) "Spatio-Temporal Graph
Convolutional Networks" with simplifications appropriate for a small
fixed scalp-electrode graph and a per-trigger artifact regression task.
See ``documentation/research_notes.md`` for the design rationale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np
import torch
from torch import nn

from facet.training.dataset import NPZContextArtifactDataset

# 30-channel order in the Niazy proof-fit bundle. Hard-coded so the
# Chebyshev Laplacian baked into the model is reproducible across runs.
NIAZY_PROOF_FIT_CHANNELS: tuple[str, ...] = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2", "AF4",
    "AF3", "FC2", "FC1", "CP1", "CP2", "PO3", "PO4", "FC6", "FC5", "CP5",
)

# Niazy uses old 10-20 names; map to modern 10-05 montage equivalents.
_LEGACY_NAME_ALIAS: dict[str, str] = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}


def _channel_positions(ch_names: tuple[str, ...]) -> np.ndarray:
    montage = mne.channels.make_standard_montage("standard_1005")
    positions = montage.get_positions()["ch_pos"]
    out = np.zeros((len(ch_names), 3), dtype=np.float64)
    for idx, name in enumerate(ch_names):
        lookup = _LEGACY_NAME_ALIAS.get(name, name)
        if lookup not in positions:
            raise KeyError(f"Channel '{name}' (alias '{lookup}') not in standard_1005 montage")
        out[idx] = positions[lookup]
    return out


def _knn_adjacency(positions: np.ndarray, k: int) -> np.ndarray:
    n = positions.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be smaller than the number of nodes ({n})")
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argsort(distances, axis=1)[:, :k]

    edges: set[tuple[int, int]] = set()
    edge_distances: list[float] = []
    for i in range(n):
        for j in nearest[i]:
            a, b = int(i), int(j)
            if a == b:
                continue
            edge = (a, b) if a < b else (b, a)
            if edge in edges:
                continue
            edges.add(edge)
            edge_distances.append(float(distances[a, b]))

    sigma = float(np.median(edge_distances)) if edge_distances else 1.0
    if sigma <= 0:
        sigma = 1.0

    adjacency = np.zeros((n, n), dtype=np.float64)
    for (a, b), d in zip(edges, edge_distances, strict=True):
        weight = float(np.exp(-(d ** 2) / (sigma ** 2)))
        adjacency[a, b] = weight
        adjacency[b, a] = weight
    np.fill_diagonal(adjacency, 1.0)
    return adjacency


def _normalised_laplacian(adjacency: np.ndarray) -> np.ndarray:
    n = adjacency.shape[0]
    degree = adjacency.sum(axis=1)
    with np.errstate(divide="ignore"):
        degree_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    normalised = adjacency * degree_inv_sqrt[:, np.newaxis] * degree_inv_sqrt[np.newaxis, :]
    return np.eye(n) - normalised


def build_chebyshev_laplacian(ch_names: tuple[str, ...], k: int = 4) -> torch.Tensor:
    """Compute ``L_norm - I`` for use in the Chebyshev recursion.

    The rescaling assumes ``lambda_max == 2`` for the symmetric
    normalised Laplacian (true for a connected graph and a tight bound
    in general), so ``L_tilde = 2L/lambda_max - I = L - I``.
    """
    positions = _channel_positions(ch_names)
    adjacency = _knn_adjacency(positions, k=k)
    laplacian = _normalised_laplacian(adjacency)
    return torch.from_numpy(laplacian - np.eye(laplacian.shape[0])).float()


class ChebConv(nn.Module):
    """Dense Chebyshev spectral graph convolution of order ``K``.

    Operates on a tensor of shape ``(B, C_in, N, T)`` where the third
    dimension is the graph-node dimension. The rescaled Laplacian
    ``L_tilde`` is supplied as a non-trainable buffer of shape
    ``(N, N)``.
    """

    def __init__(self, in_channels: int, out_channels: int, k_order: int = 3) -> None:
        super().__init__()
        if k_order < 1:
            raise ValueError("k_order must be >= 1")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.k_order = int(k_order)
        self.weight = nn.Parameter(torch.empty(self.k_order, self.in_channels, self.out_channels))
        self.bias = nn.Parameter(torch.zeros(self.out_channels))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor, l_tilde: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, N, T). Move N to the position multiplied by L_tilde.
        x_perm = x.permute(0, 3, 2, 1)  # (B, T, N, C_in)
        t_prev = x_perm
        out = torch.einsum("io,btni->btno", self.weight[0], t_prev)
        if self.k_order > 1:
            t_curr = torch.einsum("nm,btmi->btni", l_tilde, x_perm)
            out = out + torch.einsum("io,btni->btno", self.weight[1], t_curr)
            for k in range(2, self.k_order):
                t_next = 2.0 * torch.einsum("nm,btmi->btni", l_tilde, t_curr) - t_prev
                out = out + torch.einsum("io,btni->btno", self.weight[k], t_next)
                t_prev = t_curr
                t_curr = t_next
        out = out + self.bias
        return out.permute(0, 3, 2, 1)  # (B, C_out, N, T)


class TemporalGLU(nn.Module):
    """Gated 1-D convolution along the time axis of an ``(B, C, N, T)`` tensor."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to keep T constant with symmetric padding")
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=int(in_channels),
            out_channels=2 * int(out_channels),
            kernel_size=(1, self.kernel_size),
            padding=(0, self.padding),
        )
        self.out_channels = int(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = self.conv(x)
        a, b = gated.split(self.out_channels, dim=1)
        return torch.tanh(a) * torch.sigmoid(b)


class STConvBlock(nn.Module):
    """Yu/Yin/Zhu ST-Conv block: TGLU -> ChebConv -> TGLU + residual + LN."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        time_kernel: int = 3,
        k_order: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tglu1 = TemporalGLU(in_channels, hidden_channels, kernel_size=time_kernel)
        self.cheb = ChebConv(hidden_channels, hidden_channels, k_order=k_order)
        self.tglu2 = TemporalGLU(hidden_channels, out_channels, kernel_size=time_kernel)
        self.dropout = nn.Dropout(p=dropout)
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)

    def forward(self, x: torch.Tensor, l_tilde: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        h = self.tglu1(x)
        h = torch.relu(self.cheb(h, l_tilde))
        h = self.dropout(h)
        h = self.tglu2(h)
        return self.norm(h + residual)


class SpatiotemporalGNN(nn.Module):
    """Two ST-Conv blocks operating on the 30-electrode graph.

    Input shape:  ``(batch, context_epochs, n_channels, samples)``
    Output shape: ``(batch, n_channels, samples)`` — the predicted
    artifact at the center context epoch.
    """

    def __init__(
        self,
        context_epochs: int,
        n_channels: int,
        samples: int,
        l_tilde: torch.Tensor,
        hidden_channels: int = 16,
        time_kernel: int = 3,
        k_order: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if context_epochs < 1 or context_epochs % 2 == 0:
            raise ValueError("context_epochs must be a positive odd integer")
        if l_tilde.shape != (n_channels, n_channels):
            raise ValueError(
                f"l_tilde shape {tuple(l_tilde.shape)} does not match n_channels={n_channels}"
            )
        self.context_epochs = int(context_epochs)
        self.n_channels = int(n_channels)
        self.samples = int(samples)
        self.center_idx = self.context_epochs // 2
        self.register_buffer("l_tilde", l_tilde.clone().detach().float(), persistent=True)

        self.block1 = STConvBlock(
            in_channels=1,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            time_kernel=time_kernel,
            k_order=k_order,
            dropout=dropout,
        )
        self.block2 = STConvBlock(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            time_kernel=time_kernel,
            k_order=k_order,
            dropout=dropout,
        )
        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input layout: (B, context, N, T). Eager-mode validation is in
        # the caller (build_dataset enforces shape). We skip a runtime
        # check here so the model traces cleanly to TorchScript.
        batch = x.shape[0]
        full_time = self.context_epochs * self.samples
        x_flat = x.reshape(batch, 1, self.context_epochs, self.n_channels, self.samples)
        x_flat = x_flat.permute(0, 1, 3, 2, 4).reshape(batch, 1, self.n_channels, full_time)

        h = self.block1(x_flat, self.l_tilde)
        h = self.block2(h, self.l_tilde)
        out = self.head(h)  # (B, 1, N, T_full)

        center_start = self.center_idx * self.samples
        center_stop = center_start + self.samples
        center = out[:, 0, :, center_start:center_stop]
        return center  # (B, N, samples)


# ---------------------------------------------------------------------------
# facet-train factory functions
# ---------------------------------------------------------------------------


def build_model(
    input_shape: tuple[int, int, int] | None = None,
    target_shape: tuple[int, int] | None = None,
    context_epochs: int | None = None,
    epoch_samples: int | None = None,
    n_channels: int | None = None,
    hidden_channels: int = 16,
    time_kernel: int = 3,
    k_order: int = 3,
    dropout: float = 0.1,
    knn_k: int = 4,
    channel_names: list[str] | tuple[str, ...] | None = None,
    **_: object,
) -> SpatiotemporalGNN:
    if input_shape is not None:
        ctx, n_ch, samples = input_shape
    else:
        if context_epochs is None or epoch_samples is None or n_channels is None:
            raise ValueError(
                "build_model needs input_shape, or all of context_epochs / epoch_samples / n_channels"
            )
        ctx, n_ch, samples = int(context_epochs), int(n_channels), int(epoch_samples)

    names = tuple(channel_names) if channel_names is not None else NIAZY_PROOF_FIT_CHANNELS
    if len(names) != n_ch:
        raise ValueError(
            f"channel_names length ({len(names)}) must match n_channels from dataset ({n_ch})"
        )

    l_tilde = build_chebyshev_laplacian(names, k=knn_k)
    return SpatiotemporalGNN(
        context_epochs=int(ctx),
        n_channels=int(n_ch),
        samples=int(samples),
        l_tilde=l_tilde,
        hidden_channels=int(hidden_channels),
        time_kernel=int(time_kernel),
        k_order=int(k_order),
        dropout=float(dropout),
    )


def build_loss(name: str = "mse") -> nn.Module:
    normalised = name.strip().lower()
    if normalised == "l1":
        return nn.L1Loss()
    if normalised in {"smooth_l1", "huber"}:
        return nn.SmoothL1Loss()
    if normalised == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unsupported loss '{name}'. Use 'mse', 'l1', or 'huber'.")


class _DemeanedNPZContextArtifactDataset(NPZContextArtifactDataset):
    """NPZ dataset variant that demeans per-window before returning."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_examples: int | None = None,
        demean_input: bool = True,
        demean_target: bool = True,
    ) -> None:
        super().__init__(
            path=path,
            input_key="noisy_context",
            target_key="artifact_center",
            max_examples=max_examples,
            demean_input=demean_input,
            demean_target=demean_target,
        )


def build_dataset(
    path: str | None = None,
    max_examples: int | None = None,
    demean_input: bool = True,
    demean_target: bool = True,
    **_: object,
) -> _DemeanedNPZContextArtifactDataset:
    if path is None:
        raise ValueError("build_dataset requires path to the .npz bundle")
    return _DemeanedNPZContextArtifactDataset(
        path=Path(path).expanduser(),
        max_examples=max_examples,
        demean_input=demean_input,
        demean_target=demean_target,
    )
