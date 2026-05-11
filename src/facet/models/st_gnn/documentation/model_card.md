# Spatiotemporal Graph Neural Network (ST-GNN) Model Card

## Summary

`st_gnn` is a multichannel graph network that predicts the
fMRI-gradient artifact for the centre trigger of a 7-epoch context
window. The model treats the 30-channel scalp montage as a fixed graph
and stacks two ST-Conv blocks (gated 1-D temporal conv → Chebyshev
spectral graph conv → gated 1-D temporal conv) followed by a 1×1
projection. The output is the predicted artifact at the centre epoch.

## Intended Role

The first graph-based baseline in the FACETpy deep-learning catalog,
covering family 7 (Graph / GNN) of the architecture catalog. It tests
whether explicit topographic adjacency improves artifact reconstruction
relative to the channel-wise `cascaded_dae` and `cascaded_context_dae`
baselines on the same dataset.

## Input And Output

- Input: `(batch, 7, 30, 512)`. The 30 channels follow the Niazy
  proof-fit order baked into `NIAZY_PROOF_FIT_CHANNELS`.
- Output: `(batch, 30, 512)` artifact estimate at the centre epoch.
- Correction: predicted artifact is subtracted by
  `DeepLearningCorrection`.

## Compatibility Notes

- The trained checkpoint embeds the 30-channel Laplacian. Different
  channel sets or orderings invalidate the model; the adapter raises
  `ProcessorValidationError` if any expected channel is missing.
- Native trigger-to-trigger artifact lengths can differ from the
  512-sample model length. The adapter resamples per epoch in and the
  predicted centre artifact back to the native length before
  subtraction.
- Requires trigger metadata (`uses_triggers=True`).

## Hyperparameters

| Name              | Value | Notes |
|-------------------|-------|-------|
| `context_epochs`  | 7     | Matches `cascaded_context_dae`. |
| `epoch_samples`   | 512   | Matches `cascaded_context_dae`. |
| `hidden_channels` | 16    | Width inside ST-Conv blocks. |
| `time_kernel`     | 3     | Per the original ST-GCN paper. |
| `k_order`         | 3     | Chebyshev order. |
| `dropout`         | 0.1   | After each TGLU. |
| `knn_k`           | 4     | k-NN neighbours per electrode. |
| Optimiser         | Adam  | `lr=1e-3, weight_decay=1e-4`. |
| Loss              | MSE   | Same as cascaded baselines for fair comparison. |

## Evaluation Notes

Use the standard evaluation structure described in
`src/facet/models/evaluation_standard.md`. Compare against
`cascaded_context_dae` and `cascaded_dae` using the same Niazy
proof-fit dataset.
