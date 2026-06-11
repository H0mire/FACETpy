# Spatiotemporal Graph Neural Network — Paper-Accurate Edition

A more faithful reimplementation of the ST-GNN scalp-electrode graph
network for fMRI gradient-artifact prediction in EEG. The spatiotemporal
block now follows **Yu, Yin & Zhu 2018, "Spatio-Temporal Graph
Convolutional Networks"** (`arXiv:1709.04875`, "STGCN") much more closely,
and the electrode adjacency adopts the **EEG-GCNN** domain-guided geodesic
spatial graph (Wagh & Varatharajah 2020, `arXiv:2011.12107`). The Chebyshev
spectral convolution follows Defferrard et al. 2016 (`arXiv:1606.09375`).

This package lives **alongside** the original `st_gnn` and does not modify
it. It registers a globally-unique processor name
`st_gnn_paper_accurate_correction`.

## Scope

- Input shape: `(batch, context_epochs=7, n_channels=30, samples=512)`.
- Output shape: `(batch, n_channels=30, samples=512)` — predicted artifact
  at the centre context epoch.
- Multichannel: the entire 30-electrode graph is processed at once.
- Requires trigger metadata at inference (context epochs are
  trigger-aligned).
- The trained TorchScript checkpoint bakes in the 30-channel adjacency in a
  fixed order; downstream pipelines must present those channels by name.

## What changed vs. the original `st_gnn` (and why)

| # | Change | Paper basis |
|---|--------|-------------|
| 1 | **GLU gating fixed.** `TemporalGLU` now returns `P (linear) * sigmoid(Q)` instead of the GTU-style `tanh(P) * sigmoid(Q)`. | STGCN **Eq. 7** (Dauphin-style GLU: only the gate gets a nonlinearity). |
| 2 | **Spatial bottleneck.** The ST-Conv block is now a real sandwich `TGLU(in→hidden) → ReLU(Cheb(hidden→bottleneck)) → TGLU(bottleneck→out)` with `bottleneck < hidden`. The original used a constant width 16 with no squeeze. | STGCN **Fig. 2 / Sec. 4** (channel bottleneck 64→16→64). |
| 3 | **LayerNorm.** Each block ends with a real `nn.LayerNorm` over the feature/channel axis (per node, per time), replacing `GroupNorm(num_groups=1)` which normalised jointly over `(C, N, T)`. | STGCN **Sec. 3.4** (Layer Normalization within every ST-Conv block). |
| 4 | **Paper output layer.** After the two blocks, a final temporal conv (width `K_t`) precedes the 1×1 linear projection, echoing the paper's "extra temporal conv then fully-connected output". | STGCN **Eq. 9** region. |
| 5 | **Geodesic adjacency.** Electrode positions are projected to a unit sphere and the spatial distance is the **great-circle** distance `arccos(<u, v>)` normalised to `[0, 1]`, instead of raw 3-D Euclidean *chord* distance. | EEG-GCNN **Eq. 1** region (domain-guided geodesic-on-sphere spatial graph). |
| 6 | **Configurable graph order / threshold.** `k_order=1` gives the Kipf (2017) 1st-order variant STGCN also offers; an optional `epsilon` builds a distance-threshold graph instead of k-NN. | STGCN **Eq. 4–5** (1st-order), **Eq. 10** (thresholded weighted graph). |

The dense TorchScript-friendly Chebyshev recursion, the `lambda_max ≈ 2`
rescaling (`L̃ = L_norm − I`), and the MSE/L2 loss were already faithful and
are retained.

## Deliberate, documented deviations (kept for EEG-fMRI)

These differ from the paper **on purpose** because FACETpy reconstructs a
full artifact waveform rather than forecasting a scalar:

- **Length-preserving temporal convs.** The paper uses causal, non-padded
  convs that shrink time toward a single forecast step. We pad to keep
  `T` constant so the model emits a full 512-sample artifact per epoch.
  A `causal=True` flag left-pads (instead of symmetric-pads) to honour the
  causal intent without breaking the output shape.
- **Full-length output head**, not the paper's single-step FC scalar.
- **EEG-GCNN functional-coherence branch dropped** (single subject,
  artifact-dominated data); only the geodesic spatial branch is used.
- **Per-window demean** instead of global Z-score, matching the
  cascaded-context baselines for like-for-like comparison.
- **Graclus coarsening / binary-tree pooling skipped** — irrelevant for a
  fixed 30-node regression graph where node identity must be preserved.

See `documentation/paper_accuracy_review.md` for the full discrepancy table
with severities and the per-method EEG-fMRI applicability assessment.

## Architecture summary

```
Input  : (B, 7, 30, 512)
        └─ reshape / permute → (B, 1, 30, 7*512=3584)
ST-Conv block 1: TGLU(1→H) → ReLU(Cheb K=3, H→Bneck) → TGLU(Bneck→H)
                 + 1×1 residual + LayerNorm(over channels)
ST-Conv block 2: TGLU(H→H) → ReLU(Cheb K=3, H→Bneck) → TGLU(Bneck→H)
                 + residual + LayerNorm
Output layer   : temporal conv(H→H, 1×K_t) → ReLU → 1×1 conv(H→1)
Center crop on time (samples 3*512 : 4*512)
Output : (B, 30, 512)
```

(Production widths: `hidden_channels=64`, `bottleneck_channels=16`, matching
the paper's 64→16→64. Smoke widths are far smaller.)

## Adjacency

Built once at training time:

1. Channel positions looked up in MNE's `standard_1005` montage
   (`T3→T7, T4→T8, T5→P7, T6→P8` aliasing for Niazy's older nomenclature).
2. Positions projected to the **unit sphere**; pairwise **geodesic**
   distance `arccos(clip(<u, v>, -1, 1))`.
3. Distances normalised to `[0, 1]` (EEG-GCNN convention).
4. Symmetric k-NN graph (`k=4`, default) or `epsilon`-threshold graph, with
   Gaussian edge weights `exp(-d²/σ²)`, `σ` = median normalised edge
   distance.
5. Self-loops weight 1.0.
6. Symmetric normalised Laplacian `L = I − D^{-1/2} A D^{-1/2}`, rescaled to
   `L̃ = L − I` (lambda_max ≈ 2, STGCN's stated approximation).

## Training

Build the Niazy proof-fit dataset (same as the original `st_gnn`), then:

```bash
uv run facet-train fit \
  --config src/facet/models/st_gnn_paper_accurate_edition/training_niazy_proof_fit_smoke.yaml
```

The smoke YAML uses `device: cpu`, tiny dims, `max_epochs: 1`. For a real
run, scale `hidden_channels` to 64 / `bottleneck_channels` to 16 and enable
a 0.7-per-5-epoch step LR scheduler (STGCN Sec. 4) in facet-train.

## Inference

```python
from facet.models.st_gnn_paper_accurate_edition import (
    PaperAccurateSpatiotemporalGNNCorrection,
)

context = context | PaperAccurateSpatiotemporalGNNCorrection(
    checkpoint_path="training_output/<run>/exports/st_gnn_paper_accurate.ts",
    context_epochs=7,
    epoch_samples=512,
)
```

The processor validates that the EEG context contains the 30-channel set the
model was trained on and resamples native trigger-to-trigger epochs to the
model's 512-sample input.
