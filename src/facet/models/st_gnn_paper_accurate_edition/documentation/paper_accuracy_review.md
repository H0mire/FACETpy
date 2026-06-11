# ST-GNN Paper-Accuracy Review

Reviewer assessment of the original FACETpy `st_gnn` against its source
papers, and the fixes applied in this paper-accurate edition.

## Source papers

1. **Yu, B., Yin, H., Zhu, Z. (2018).** *Spatio-Temporal Graph
   Convolutional Networks: A Deep Learning Framework for Traffic
   Forecasting.* arXiv:1709.04875. ("STGCN")
2. **Wagh, N., Varatharajah, Y. (2020).** *EEG-GCNN: Augmenting
   EEG-based Neurological Disease Diagnosis using a Domain-guided Graph
   Convolutional Neural Network.* arXiv:2011.12107. ("EEG-GCNN")
3. **Defferrard, M., Bresson, X., Vandergheynst, P. (2016).** *CNNs on
   Graphs with Fast Localized Spectral Filtering.* arXiv:1606.09375.
   ("ChebNet")
4. **Kipf, T., Welling, M. (2017).** *Semi-Supervised Classification with
   Graph Convolutional Networks.* arXiv:1609.02907. (1st-order variant.)

## Discrepancy table

| # | Aspect | Paper specifies | Original `st_gnn` | Severity | Status in this edition |
|---|--------|-----------------|-------------------|----------|------------------------|
| 1 | GLU activation | STGCN Eq. 7: split conv output `[P, Q]`; output = `P` (linear) ⊙ `sigmoid(Q)`. Only the gate is nonlinear. | `TemporalGLU` computes `tanh(P) * sigmoid(Q)` (a GTU gate, not a GLU). | **High** | **Fixed.** `TemporalGLU.forward` returns `p * sigmoid(q)`. |
| 2 | ST-Conv block channel topology | STGCN Fig. 2 / Sec. 4: temporal-spatial-temporal sandwich with a spatial **bottleneck** (experiment 64→16→64). | Constant hidden width 16 (`1→16→16→16→16→1`); Cheb keeps `16→16`. No squeeze. | Medium | **Fixed.** Block is `TGLU(in→hidden) → ReLU(Cheb(hidden→bottleneck)) → TGLU(bottleneck→out)`, `bottleneck < hidden`. Production widths 64/16. |
| 3 | Normalization | STGCN Sec. 3.4: Layer Normalization at the end of each ST-Conv block. | `GroupNorm(num_groups=1)` (normalises jointly over `(C, N, T)`). | Medium | **Fixed.** `_ChannelLayerNorm` = `nn.LayerNorm` over the channel axis per `(node, time)`. |
| 4 | Output layer | STGCN Eq. 9 region: extra temporal conv → single time step → FC linear map. | Single `1×1 Conv2d` head over full time, then center crop. No dedicated temporal-conv collapse / FC. | Low | **Adapted (faithful in structure).** Added a final temporal conv (`1×K_t`) before the `1×1` projection; head stays length-preserving because the target is a full waveform. See deviation D2. |
| 5 | Graph adjacency | EEG-GCNN: `A = 0.5·(A_spatial + A_functional)`; `A_spatial` = **geodesic-on-sphere** distance, normalised to `[0,1]`. STGCN Eq. 10 uses thresholded `exp(-d²/σ²)`. | k-NN (k=4) on **raw 3-D Euclidean chord** distance; no functional branch; no threshold. | Medium | **Fixed (spatial branch).** Geodesic `arccos(<u,v>)` on the unit sphere, normalised to `[0,1]`, Gaussian weights, optional `epsilon` threshold. Functional branch dropped — deviation D3. |
| 6 | Spatial-conv nonlinearity placement | STGCN Eq. 8: block is TGLU → ReLU(spatial graph-conv) → TGLU. | `relu(cheb(...))` then dropout then `tglu2` — consistent. Dropout not in paper. | Low | **Kept.** ReLU on the spatial output (Eq. 8). Dropout retained as a documented, configurable regulariser. |
| 7 | `lambda_max` rescaling | ChebNet/STGCN rescale `L̃ = 2L/λ_max − I`; STGCN notes `λ_max ≈ 2`. | Hard-codes `λ_max = 2` (`L̃ = L_norm − I`). Faithful. | Low | **Kept** (paper-sanctioned approximation). |
| 8 | Optimizer / LR schedule | RMSprop/Adam, lr=1e-3, 0.7 decay every 5 epochs, batch 50, 50 epochs, Z-score input. | Adam lr=1e-3, wd=1e-4, grad-clip, early stop, per-window demean. No 0.7/5-epoch step decay. (Owned by facet-train.) | Low | **Advisory.** YAML documents the step-decay recommendation; per-window demean is a deliberate baseline-matching choice (D4). The factory cannot control the schedule. |

## EEG-fMRI applicability assessment

| Paper method | Keep? | Rationale |
|--------------|-------|-----------|
| Chebyshev K-localized spectral graph conv on a fixed scalp graph | **Yes** | The gradient artifact has strong topographic structure across neighbouring electrodes; a K-localized spatial filter is the right inductive bias and is cheap on a 30-node graph. |
| Temporal gated conv with proper GLU gating (`P * sigmoid(Q)`) | **Yes** | Gated temporal convs model the periodic intra-TR artifact well and are CPU-cheap. Gating fixed to the paper's GLU. |
| ST-Conv sandwich with spatial bottleneck + LayerNorm + residual | **Yes** | The defining STGCN unit; transfers directly. Bottleneck + LayerNorm adopted; widths kept tiny for CPU at smoke time. |
| Causal non-padded temporal conv shrinking T toward one forecast step | **No** | FACETpy reconstructs a full 512-sample artifact per epoch, not a one-step forecast. Length-preserving padded convs are the correct adaptation (deviation D2). |
| Output layer = temporal-conv-to-single-step + FC scalar regression | **No** | Single-step FC suits scalar traffic-speed forecasting; here the target is a multichannel time series, so a full-length conv head is required. Final temporal conv retained to echo the paper structure. |
| EEG-GCNN geodesic-on-sphere spatial adjacency | **Yes** | The EEG-correct way to build the electrode graph; trivially cheap. Adopted in place of raw Euclidean k-NN. |
| EEG-GCNN functional-coherence adjacency branch | **No** | Needs multi-window coherence estimation; single-subject, gradient-artifact-dominated data makes resting-state coherence uninformative (deviation D3). |
| STGCN 1st-order (Kipf) graph-conv variant | **Yes (configurable)** | Useful cheap ablation (`k_order=1` collapses to local averaging) and what most EEG-GNN follow-ups use. Exposed; default `k_order=3`. |
| Graclus coarsening + binary-tree max pooling (ChebNet) | **No** | Pooling/coarsening matters for large graphs and classification; a fixed 30-node regression graph needs none and node identity must be preserved to output per-electrode artifacts. |
| Z-score input norm + RMSprop with 0.7/5-epoch step decay, batch 50, 50 epochs | **Advisory** | Reasonable recipe; step-decay recommended in the YAML. Per-window demean kept to match cascaded-context baselines (D4). Training schedule is owned by facet-train. |

## Documented deviations (summary)

- **D1 — Dense TorchScript-friendly ChebConv.** No runtime `torch_geometric`
  dependency (PyG `edge_index` does not `torch.jit.trace` cleanly). The dense
  recursion on a 30-node `(N, N)` Laplacian is mathematically identical.
- **D2 — Length-preserving temporal convs + full-length output head.**
  Required for full-waveform artifact regression; a `causal=True` flag
  left-pads to honour the paper's causal intent without changing output
  shape.
- **D3 — Functional-coherence adjacency branch dropped.** Single-subject,
  artifact-dominated data; only the geodesic spatial branch is used.
- **D4 — Per-window demean instead of global Z-score.** Matches the
  cascaded-context baselines for like-for-like comparison.
