# ST-GNN Research Notes

This document records the research underlying the FACETpy `st_gnn`
implementation for fMRI gradient-artifact removal in EEG. It is the
reasoning artefact required by the model-agent workflow before any
implementation work begins.

## Source Papers

1. **Yu, B., Yin, H., Zhu, Z. (2018).** *Spatio-Temporal Graph
   Convolutional Networks: A Deep Learning Framework for Traffic
   Forecasting.* arXiv:1709.04875.
   <https://arxiv.org/abs/1709.04875>
2. **Wagh, N., Varatharajah, Y. (2020).** *EEG-GCNN: Augmenting
   Electroencephalogram-based Neurological Disease Diagnosis using a
   Domain-guided Graph Convolutional Neural Network.* arXiv:2011.12107.
   <https://arxiv.org/abs/2011.12107>
3. **Defferrard, M., Bresson, X., Vandergheynst, P. (2016).**
   *Convolutional Neural Networks on Graphs with Fast Localized Spectral
   Filtering.* arXiv:1606.09375 — defines the Chebyshev spectral graph
   convolution that ST-GCN uses for the spatial block.
4. **Kipf, T., Welling, M. (2017).** *Semi-Supervised Classification with
   Graph Convolutional Networks.* arXiv:1609.02907 — the K=1 special
   case used widely in EEG-GNN follow-ups.

The thesis report points at this family in
`docs/research/dl_eeg_gradient_artifacts.pdf` Section 7.3 (Graph Neural
Networks: Topographic Consistency).

## Plain-Language Description

A spatiotemporal graph network treats the 30-channel EEG montage as a
fixed graph: each electrode is a node, edges encode scalp neighbourhood,
and the time axis is processed by a 1-D temporal convolution. The
gradient artifact has strong topographic structure (it looks similar on
neighbouring electrodes because the gradient coil produces a smooth
spatial pattern). A graph conv lets the network borrow information from
neighbouring electrodes when predicting the artifact at one electrode.
Stacking spatial graph convolutions with temporal convolutions captures
both periodic structure inside one TR and topographic consistency
across electrodes.

## Architecture (As Adopted Here)

The ST-GCN paper proposes ST-Conv blocks with the layered structure
`TGLU → ChebConv → TGLU → residual`. We adopt this with simplifications
appropriate for our small fixed graph and our regression task:

- Input tensor: `(batch, context_epochs=7, n_channels=30, samples=512)`.
- We treat the 7 context epochs as a single concatenated time axis
  `T = 7 * 512 = 3584` so a single spatiotemporal model sees the full
  context.
- Tensor layout inside the model: `(batch, features=1, nodes=30, time)`.
- Two stacked ST-Conv blocks operate on that tensor.
- A final temporal 1×1 conv projects features back to one channel.
- The center epoch (samples `3*512 : 4*512`) of the output is returned
  as the predicted artifact for the center trigger, matching the
  cascaded_context_dae output contract: `(batch, 30, 512)`.

### ST-Conv Block

Following Yu/Yin/Zhu:

1. Temporal Gated Linear Unit (TGLU). 2-D conv with kernel `(1, K_t)`,
   `K_t=3`, padding `(0, 1)` to keep `T` constant. Output is split in
   half along channels; first half is `tanh`, second half is `sigmoid`,
   product is the gated output.
2. Spatial Chebyshev graph convolution of order `K_s=3`, applied at
   each time step using the precomputed normalised Laplacian of the
   electrode graph.
3. Second TGLU.
4. Residual: input projected with a 1×1 conv when channel count
   changes, added to the output.
5. LayerNorm across the time axis at the end.

Channel progression: `1 → 16 → 16 → 16 → 16 → 1`. Two blocks use
hidden width 16 to stay well within 24 GB VRAM.

### Chebyshev Convolution

We do **not** depend on `torch_geometric.nn.ChebConv` at runtime even
though `torch_geometric` is added to `pyproject.toml`. The reason is
TorchScript export: PyG layers consume `edge_index` graph data that
does not trace cleanly with `torch.jit.trace`, and the FACETpy
`facet-train` exporter calls `torch.jit.trace` on the model with a
single dense tensor input. Implementing the Chebyshev recursion
directly on a dense `(N, N)` Laplacian is trivial (the graph has only
30 nodes), traces cleanly, and is mathematically identical. The
`torch_geometric` dependency is still added so that future hybrids
(e.g. dynamic-graph variants) can use its higher-level utilities, and
so the per-job `uv sync` on the pod has it available.

The Chebyshev recursion:

- `T_0(L̃) X = X`
- `T_1(L̃) X = L̃ X`
- `T_k(L̃) X = 2 L̃ T_{k-1}(L̃) X - T_{k-2}(L̃) X`

with rescaled Laplacian `L̃ = 2 L_norm / lambda_max - I` and
`L_norm = I - D^{-1/2} A D^{-1/2}`. We approximate `lambda_max` with
`2.0` (standard for the rescaled symmetric Laplacian; bound is exactly
2 for the normalised Laplacian) so `L̃ = L_norm - I`. This avoids one
eigenvalue computation per training start without observable accuracy
cost on a 30-node graph.

### Adjacency Construction

The Niazy proof-fit dataset uses 30 channels in this fixed order
(taken from
`output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz`):

```
Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz,
P4, T6, O1, O2, AF4, AF3, FC2, FC1, CP1, CP2, PO3, PO4, FC6, FC5, CP5
```

Niazy uses the older 10-20 nomenclature `T3/T4/T5/T6`, which corresponds
to `T7/T8/P7/P8` in the modern 10-05 montage. The channel-name to
montage-position lookup applies that translation.

Adjacency strategy:

1. Look up each channel's 3-D Cartesian position from MNE's
   `standard_1005` montage with the `T3→T7, T4→T8, T5→P7, T6→P8`
   alias.
2. Build a **k-NN graph with `k=4`** based on Euclidean distance on
   the scalp surface.
3. Make the graph symmetric (undirected) by taking the union of the
   k-NN edges in both directions.
4. Apply Gaussian edge weights:
   `A_ij = exp(-d_ij^2 / sigma^2)` with `sigma` set to the median
   non-zero Euclidean distance among the k-NN edges.
5. Add self-loops with weight 1.0.
6. Compute the symmetric normalised Laplacian
   `L = I - D^{-1/2} A D^{-1/2}` and store it as a non-trainable
   `(30, 30)` buffer on the model.

This is the "domain-guided" path also used in EEG-GCNN: a fixed
geometry-driven adjacency rather than a learned dynamic graph. We do
not attempt the EEG-GCNN functional-connectivity branch, because the
Niazy proof-fit recording is a single subject and the gradient
artifact dominates resting-state coherence anyway.

### Loss

The original STGCN paper uses MAE for traffic forecasting. The
cascaded baselines on the same dataset use MSE (`torch.nn.MSELoss`).
We adopt MSE for like-for-like comparability with
`cascaded_context_dae` and `cascaded_dae` — switching loss in addition
to switching architecture would entangle two variables. The training
factory exposes `name: mse|l1|huber` to make follow-up ablation cheap.

## Mapping To The Niazy Proof-Fit Dataset

Niazy proof-fit context bundle
(`output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz`):

| Field               | Shape           | Meaning                                       |
|---------------------|-----------------|-----------------------------------------------|
| `noisy_context`     | `(833, 7, 30, 512)` | Input EEG + AAS-residual artifact          |
| `artifact_center`   | `(833, 30, 512)`    | AAS-estimated artifact for the centre epoch |
| `clean_context`     | `(833, 7, 30, 512)` | AAS-corrected EEG (training surrogate)     |
| `sfreq`             | `(1,)`              | 4096.0 Hz                                  |

We supervise on `(noisy_context, artifact_center)` so the model is
directly comparable to the cascaded context DAE.

## Training Tricks From The Source Paper

The STGCN paper relies on:

- LayerNorm at the end of each ST-Conv block.
- Dropout after each TGLU output (we use `p=0.1`).
- Adam optimiser with `lr=1e-3` and standard decay.

For our shorter regression task we keep:

- Adam, `lr=1e-3`, `weight_decay=1e-4`.
- Gradient norm clip 1.0.
- Early stopping on validation MSE with patience 10.
- Demean the input context per-window before forward pass to focus the
  model on the AC component (matches the cascaded_context_dae
  convention).

## Hardware Feasibility

Parameter count estimate for the chosen widths:

- Block 1 TGLU (2D conv 1×3, in=1, out=2·16): 96
- Block 1 Cheb conv (K=3, in=16, out=16): 768
- Block 1 TGLU (1×3, in=16, out=2·16): 1 536
- Block 1 residual 1×1 (in=1, out=16): 16
- Block 2 TGLU (1×3, in=16, out=2·16): 1 536
- Block 2 Cheb conv (K=3, in=16, out=16): 768
- Block 2 TGLU (1×3, in=16, out=2·16): 1 536
- Final 1×1 conv (in=16, out=1): 16
- LayerNorms and biases: a few hundred more

That is ~7 k trainable parameters — orders of magnitude below
the 24 GB VRAM budget. The forward pass on a single example is
`B × 16 × 30 × 3584 × 4 bytes ≈ 6.6 MB`, so a batch of 64 fits
comfortably with activation memory under 1 GB. One epoch over the
833 examples × 30 channels-as-graph = 833 batches with `batch_size=64`
runs in single-digit seconds on an RTX 5090.

We do **not** need to reduce the published architecture to fit the
hardware envelope.

## Open Questions

1. Does the AAS-derived training target's smooth spatial structure
   actually exercise the graph convolution, or will an MLP per
   channel match performance? Worth ablating with `K_s=1` (which
   collapses to per-channel temporal conv) if the headline result is
   marginal.
2. The dataset's 30-channel order is fixed at training time. If a
   downstream user re-orders or drops electrodes, the adjacency baked
   into the TorchScript checkpoint becomes invalid. The processor
   adapter validates the channel set; a future improvement would be
   to recompute adjacency at inference and pass it as a second model
   argument, but the current `facet-train` export traces with a
   single tensor input.
3. T3/T4/T5/T6 vs T7/T8/P7/P8 naming: we apply a one-direction alias.
   If a future dataset uses the modern names directly, the adjacency
   builder accepts both via the alias map.
