"""Paper-accurate training factories and architecture for the ST-GNN model.

This is the *paper-accurate edition* of ``st_gnn``. The architecture is a
more faithful reproduction of Yu, Yin & Zhu (2018) "Spatio-Temporal Graph
Convolutional Networks" (arXiv:1709.04875, STGCN) for the spatiotemporal
block, combined with the EEG-specific "domain-guided" adjacency of Wagh &
Varatharajah (2020) "EEG-GCNN" (arXiv:2011.12107). The Chebyshev spectral
graph convolution follows Defferrard, Bresson & Vandergheynst (2016)
(arXiv:1606.09375).

What is corrected here relative to ``st_gnn`` (see
``documentation/paper_accuracy_review.md`` for the full table with severities):

1. GLU gating (STGCN Eq. 7). The temporal gated conv now computes the
   Dauphin-style GLU ``P (linear) * sigmoid(Q)``, NOT the GTU-style
   ``tanh(P) * sigmoid(Q)`` used by the original.
2. Spatial bottleneck (STGCN Fig. 2 / Sec. 4, 64->16->64). The ST-Conv
   block is now a true sandwich whose middle (spatial Cheb) layer squeezes
   the channel dimension to ``bottleneck_channels < hidden_channels`` and
   the second temporal layer widens it back.
3. Layer Normalization (STGCN Sec. 3.4). The block now ends with a real
   ``nn.LayerNorm`` over the feature (channel) axis applied per
   (node, time) position, replacing the original ``GroupNorm(num_groups=1)``.
4. Paper-style output layer (STGCN Eq. 9 region). After the two ST-Conv
   blocks the model applies a final temporal conv (width ``K_t``) before
   the 1x1 linear projection, mirroring the paper's "extra temporal conv
   then fully-connected output".
5. EEG-GCNN geodesic adjacency (EEG-GCNN Eq. 1 region). Electrode
   positions are projected to a unit sphere and the spatial distance is
   the GEODESIC (great-circle) distance ``arccos(<u, v>)`` normalised to
   ``[0, 1]``, instead of raw 3-D Euclidean (chord) distance.

Deliberate, documented deviations (kept for EEG-fMRI full-waveform artifact
regression -- these are NOT bugs):

* Length-preserving temporal convolution. The paper uses causal, NON-padded
  temporal convs that shrink the time axis toward a single forecast step.
  FACETpy must reconstruct a FULL 512-sample artifact waveform per epoch,
  not forecast one future value, so we use a length-preserving conv. A
  ``causal`` flag (default ``False``) left-pads instead of symmetric-pads to
  honour the causal intent without breaking the output shape.
* The EEG-GCNN functional-coherence adjacency branch is dropped: the Niazy
  proof-fit recording is single-subject and gradient-artifact-dominated, so
  the spatial (geodesic) branch alone is the appropriate domain prior.
* Per-window demean instead of global Z-score normalisation, matching the
  cascaded-context baselines for like-for-like comparability.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import torch
from torch import nn

from facet.training.dataset import NPZContextArtifactDataset

# 30-channel order in the Niazy proof-fit bundle. Hard-coded so the
# Chebyshev Laplacian baked into the model is reproducible across runs.
NIAZY_PROOF_FIT_CHANNELS: tuple[str, ...] = (
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T3",
    "C3",
    "Cz",
    "C4",
    "T4",
    "T5",
    "P3",
    "Pz",
    "P4",
    "T6",
    "O1",
    "O2",
    "AF4",
    "AF3",
    "FC2",
    "FC1",
    "CP1",
    "CP2",
    "PO3",
    "PO4",
    "FC6",
    "FC5",
    "CP5",
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


def _geodesic_distance_matrix(positions: np.ndarray) -> np.ndarray:
    """Great-circle distance between electrodes projected to the unit sphere.

    EEG-GCNN (Wagh & Varatharajah 2020, arXiv:2011.12107) builds the
    spatial branch of its domain-guided adjacency from the GEODESIC distance
    on a unit sphere: each electrode position is normalised to unit norm and
    the pairwise distance is ``arccos(clip(<u, v>, -1, 1))`` (radians on the
    sphere). This respects scalp curvature, unlike the raw 3-D Euclidean
    *chord* distance used by the original ``st_gnn``.
    """
    norms = np.linalg.norm(positions, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    unit = positions / norms
    cosine = np.clip(unit @ unit.T, -1.0, 1.0)
    geodesic = np.arccos(cosine)  # (N, N) in [0, pi]
    np.fill_diagonal(geodesic, 0.0)
    return geodesic


def _geodesic_adjacency(positions: np.ndarray, k: int, epsilon: float | None = None) -> np.ndarray:
    """Domain-guided adjacency from EEG-GCNN-style geodesic distance.

    1. Compute the geodesic distance matrix on the unit sphere.
    2. Standardise distances to ``[0, 1]`` (EEG-GCNN normalises both the
       spatial and functional branches to ``[0, 1]`` before averaging).
    3. Keep either a symmetric k-NN graph (default) or an epsilon-threshold
       graph, then apply Gaussian edge weights ``exp(-d^2 / sigma^2)`` with
       ``sigma`` the median retained (normalised) edge distance.
    4. Add self-loops with weight 1.0.

    The functional-coherence branch of EEG-GCNN is intentionally dropped
    (single-subject, gradient-artifact-dominated data); see the module
    docstring and ``documentation/paper_accuracy_review.md``.
    """
    n = positions.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be smaller than the number of nodes ({n})")

    geodesic = _geodesic_distance_matrix(positions)
    max_d = float(geodesic.max())
    if max_d <= 0:
        max_d = 1.0
    dist = geodesic / max_d  # normalised distances in [0, 1]

    masked = dist.copy()
    np.fill_diagonal(masked, np.inf)

    edges: set[tuple[int, int]] = set()
    if epsilon is not None:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if masked[i, j] <= epsilon:
                    edge = (i, j) if i < j else (j, i)
                    edges.add(edge)
    else:
        nearest = np.argsort(masked, axis=1)[:, :k]
        for i in range(n):
            for j in nearest[i]:
                a, b = int(i), int(j)
                if a == b:
                    continue
                edges.add((a, b) if a < b else (b, a))

    edge_distances = [float(dist[a, b]) for (a, b) in edges]
    sigma = float(np.median(edge_distances)) if edge_distances else 1.0
    if sigma <= 0:
        sigma = 1.0

    adjacency = np.zeros((n, n), dtype=np.float64)
    for (a, b), d in zip(edges, edge_distances, strict=True):
        weight = float(np.exp(-(d**2) / (sigma**2)))
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


def build_chebyshev_laplacian(
    ch_names: tuple[str, ...],
    k: int = 4,
    epsilon: float | None = None,
) -> torch.Tensor:
    """Compute the rescaled Laplacian ``L_tilde = L_norm - I`` for ChebNet.

    Uses the EEG-GCNN geodesic-on-sphere adjacency (paper-accurate edition)
    rather than the original Euclidean-chord k-NN graph. The rescaling
    assumes ``lambda_max == 2`` for the symmetric normalised Laplacian
    (STGCN's stated approximation; the bound is exactly 2 for a connected
    graph), so ``L_tilde = 2L/lambda_max - I = L - I``.
    """
    positions = _channel_positions(ch_names)
    adjacency = _geodesic_adjacency(positions, k=k, epsilon=epsilon)
    laplacian = _normalised_laplacian(adjacency)
    return torch.from_numpy(laplacian - np.eye(laplacian.shape[0])).float()


class ChebConv(nn.Module):
    """Dense Chebyshev spectral graph convolution of order ``K`` (Defferrard 2016).

    Operates on a tensor of shape ``(B, C_in, N, T)`` where the third
    dimension is the graph-node dimension. The rescaled Laplacian
    ``L_tilde`` is supplied as a non-trainable buffer of shape ``(N, N)``.
    ``k_order == 1`` collapses to the Kipf (2017) 1st-order local-averaging
    variant that STGCN also offers (Eq. 4-5).
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
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

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
    """Gated 1-D temporal conv with the paper-accurate STGCN GLU (Eq. 7).

    The ``2 * C_out`` conv output is split into ``[P, Q]`` and the gated
    output is ``P (linear) * sigmoid(Q)`` -- the Dauphin-style GLU used by
    STGCN. This is the key fix versus the original ``st_gnn`` TGLU, which
    used the GTU-style ``tanh(P) * sigmoid(Q)``.

    Length handling is a deliberate, documented EEG-fMRI deviation from the
    paper's non-padded causal conv (which shrinks ``T`` toward a single
    forecast step):

    * ``causal=False`` (default): symmetric padding keeps ``T`` constant so
      the full-length artifact-regression head works.
    * ``causal=True``: left-padding by ``kernel_size - 1`` keeps ``T``
      constant while honouring the paper's causal intent (output at ``t``
      depends only on inputs ``<= t``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        causal: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if not causal and kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric (non-causal) padding")
        self.kernel_size = int(kernel_size)
        self.causal = bool(causal)
        # In causal mode we left-pad manually in forward; in symmetric mode
        # we let Conv2d apply the (0, K//2) padding.
        conv_padding = (0, 0) if self.causal else (0, self.kernel_size // 2)
        self.conv = nn.Conv2d(
            in_channels=int(in_channels),
            out_channels=2 * int(out_channels),
            kernel_size=(1, self.kernel_size),
            padding=conv_padding,
        )
        self.out_channels = int(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal:
            pad = self.kernel_size - 1
            # Left-pad the time axis only: (left, right, top, bottom).
            x = nn.functional.pad(x, (pad, 0, 0, 0))
        gated = self.conv(x)
        p, q = gated.split(self.out_channels, dim=1)
        # STGCN Eq. 7: linear value branch P times sigmoid gate Q.
        return p * torch.sigmoid(q)


class _ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel (feature) axis of an ``(B, C, N, T)`` tensor.

    STGCN (Sec. 3.4) applies Layer Normalization at the end of every
    ST-Conv block. Here normalisation is performed independently for each
    ``(node, time)`` position over the feature/channel dimension, which is
    the natural LayerNorm analogue for the per-node feature vector. This
    replaces the original ``GroupNorm(num_groups=1)`` (which normalised
    jointly over ``(C, N, T)``).
    """

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(num_channels), eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N, T) -> (B, N, T, C) -> LN over C -> back.
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class STConvBlock(nn.Module):
    """Paper-accurate STGCN ST-Conv block (sandwich with spatial bottleneck).

    ``TGLU(in -> hidden) -> ReLU(ChebConv(hidden -> bottleneck)) ->
    TGLU(bottleneck -> out)`` with a 1x1 residual and a final LayerNorm,
    matching STGCN Eq. 8 and the bottleneck topology of Fig. 2
    (64 -> 16 -> 64 in the paper's experiments).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        bottleneck_channels: int,
        out_channels: int,
        time_kernel: int = 3,
        k_order: int = 3,
        dropout: float = 0.1,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.tglu1 = TemporalGLU(in_channels, hidden_channels, kernel_size=time_kernel, causal=causal)
        self.cheb = ChebConv(hidden_channels, bottleneck_channels, k_order=k_order)
        self.tglu2 = TemporalGLU(bottleneck_channels, out_channels, kernel_size=time_kernel, causal=causal)
        self.dropout = nn.Dropout(p=dropout)
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()
        self.norm = _ChannelLayerNorm(out_channels)

    def forward(self, x: torch.Tensor, l_tilde: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        h = self.tglu1(x)
        # STGCN Eq. 8: ReLU wraps the spatial graph conv inside the sandwich.
        h = torch.relu(self.cheb(h, l_tilde))
        h = self.dropout(h)
        h = self.tglu2(h)
        return self.norm(h + residual)


class SpatiotemporalGNN(nn.Module):
    """Paper-accurate two-block STGCN over the 30-electrode graph.

    Input shape:  ``(batch, context_epochs, n_channels, samples)``
    Output shape: ``(batch, n_channels, samples)`` -- the predicted artifact
    at the center context epoch.
    """

    def __init__(
        self,
        context_epochs: int,
        n_channels: int,
        samples: int,
        l_tilde: torch.Tensor,
        hidden_channels: int = 64,
        bottleneck_channels: int = 16,
        time_kernel: int = 3,
        k_order: int = 3,
        dropout: float = 0.1,
        causal: bool = False,
    ) -> None:
        super().__init__()
        if context_epochs < 1 or context_epochs % 2 == 0:
            raise ValueError("context_epochs must be a positive odd integer")
        if l_tilde.shape != (n_channels, n_channels):
            raise ValueError(f"l_tilde shape {tuple(l_tilde.shape)} does not match n_channels={n_channels}")
        if bottleneck_channels < 1 or hidden_channels < 1:
            raise ValueError("hidden_channels and bottleneck_channels must be positive")
        self.context_epochs = int(context_epochs)
        self.n_channels = int(n_channels)
        self.samples = int(samples)
        self.center_idx = self.context_epochs // 2
        self.register_buffer("l_tilde", l_tilde.clone().detach().float(), persistent=True)

        # Paper sandwich with spatial bottleneck (STGCN Fig. 2: hidden ->
        # bottleneck -> hidden). Block 1 lifts 1 feature to hidden width.
        self.block1 = STConvBlock(
            in_channels=1,
            hidden_channels=hidden_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=hidden_channels,
            time_kernel=time_kernel,
            k_order=k_order,
            dropout=dropout,
            causal=causal,
        )
        self.block2 = STConvBlock(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=hidden_channels,
            time_kernel=time_kernel,
            k_order=k_order,
            dropout=dropout,
            causal=causal,
        )
        # Paper-style output layer (STGCN Eq. 9 region): an extra temporal
        # conv over the post-block features followed by a 1x1 linear
        # projection. The paper collapses time to a single forecast step;
        # here the head is length-preserving because the target is a full
        # 512-sample waveform (documented EEG-fMRI deviation).
        if causal:
            out_pad = (0, 0)
        else:
            out_pad = (0, time_kernel // 2)
        self.out_temporal = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=(1, time_kernel),
            padding=out_pad,
        )
        self.out_causal = bool(causal)
        self.out_kernel = int(time_kernel)
        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input layout: (B, context, N, T). Eager-mode validation is in the
        # caller (build_dataset enforces shape). We skip a runtime check
        # here so the model traces cleanly to TorchScript.
        batch = x.shape[0]
        full_time = self.context_epochs * self.samples
        x_flat = x.reshape(batch, 1, self.context_epochs, self.n_channels, self.samples)
        x_flat = x_flat.permute(0, 1, 3, 2, 4).reshape(batch, 1, self.n_channels, full_time)

        h = self.block1(x_flat, self.l_tilde)
        h = self.block2(h, self.l_tilde)

        # Output layer: extra temporal conv (Eq. 9 region) then linear 1x1.
        if self.out_causal:
            h = nn.functional.pad(h, (self.out_kernel - 1, 0, 0, 0))
        h = torch.relu(self.out_temporal(h))
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
    hidden_channels: int = 64,
    bottleneck_channels: int = 16,
    time_kernel: int = 3,
    k_order: int = 3,
    dropout: float = 0.1,
    knn_k: int = 4,
    epsilon: float | None = None,
    causal: bool = False,
    channel_names: list[str] | tuple[str, ...] | None = None,
    **_: object,
) -> SpatiotemporalGNN:
    if input_shape is not None:
        ctx, n_ch, samples = input_shape
    else:
        if context_epochs is None or epoch_samples is None or n_channels is None:
            raise ValueError("build_model needs input_shape, or all of context_epochs / epoch_samples / n_channels")
        ctx, n_ch, samples = int(context_epochs), int(n_channels), int(epoch_samples)

    names = tuple(channel_names) if channel_names is not None else NIAZY_PROOF_FIT_CHANNELS
    if len(names) != n_ch:
        raise ValueError(f"channel_names length ({len(names)}) must match n_channels from dataset ({n_ch})")

    l_tilde = build_chebyshev_laplacian(names, k=knn_k, epsilon=epsilon)
    return SpatiotemporalGNN(
        context_epochs=int(ctx),
        n_channels=int(n_ch),
        samples=int(samples),
        l_tilde=l_tilde,
        hidden_channels=int(hidden_channels),
        bottleneck_channels=int(bottleneck_channels),
        time_kernel=int(time_kernel),
        k_order=int(k_order),
        dropout=float(dropout),
        causal=bool(causal),
    )


def build_loss(name: str = "mse", **_: object) -> nn.Module:
    # STGCN uses an L2 / MSE objective (Eq. 9); MSE is the paper-accurate
    # default. l1/huber are exposed for cheap ablation.
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
    context_path: str | None = None,
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
