# Spatiotemporal Graph Neural Network (ST-GNN)

A scalp-electrode graph network for fMRI gradient-artifact prediction
in EEG. Implements the ST-Conv block from Yu/Yin/Zhu 2018
(`arXiv:1709.04875`) with a fixed-geometry electrode adjacency in the
spirit of EEG-GCNN (`arXiv:2011.12107`).

## Scope

- Input shape: `(batch, context_epochs=7, n_channels=30, samples=512)`.
- Output shape: `(batch, n_channels=30, samples=512)` — predicted
  artifact at the centre context epoch.
- Multichannel: the entire 30-electrode graph is processed at once,
  unlike `cascaded_dae` and `cascaded_context_dae` which run
  channel-wise.
- Requires trigger metadata at inference because context epochs are
  trigger-aligned.
- The trained TorchScript checkpoint bakes in the 30-channel adjacency
  in a fixed order; downstream pipelines must present those channels
  by name.

## Architecture Summary

```
Input  : (B, 7, 30, 512)
        └─ reshape / permute → (B, 1, 30, 7*512)
ST-Conv block 1 (TGLU → ChebConv K=3 → TGLU + residual + GroupNorm)
        → (B, 16, 30, 3584)
ST-Conv block 2 (TGLU → ChebConv K=3 → TGLU + residual + GroupNorm)
        → (B, 16, 30, 3584)
1x1 head conv
        → (B, 1, 30, 3584)
Center crop on time
Output : (B, 30, 512)
```

The Chebyshev convolution is implemented directly on a dense `(30, 30)`
rescaled normalised Laplacian buffer so the model traces cleanly to
TorchScript. `torch_geometric` is added as a dependency for future
extensions even though the runtime path does not invoke it.

See `documentation/research_notes.md` for the full design rationale.

## Adjacency

The graph is built once at training time:

1. Channel positions are looked up in MNE's `standard_1005` montage
   (with `T3→T7, T4→T8, T5→P7, T6→P8` aliasing for Niazy's older
   nomenclature).
2. Symmetric k-NN graph with `k=4`.
3. Gaussian edge weights `exp(-d^2/sigma^2)` with `sigma` set to the
   median k-NN distance.
4. Self-loops with weight 1.0.
5. Symmetric normalised Laplacian, then rescaled to `L̃ = L − I`.

## Training

Build the Niazy proof-fit dataset first:

```bash
uv run python examples/build_niazy_proof_fit_context_dataset.py \
  --artifact-bundle output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz \
  --target-epoch-samples 512 \
  --context-epochs 7 \
  --output-dir output/niazy_proof_fit_context_512
```

Then train and export:

```bash
uv run facet-train fit --config src/facet/models/st_gnn/training_niazy_proof_fit.yaml
```

The training pipeline writes:

- `summary.json`
- `loss.png`
- `exports/st_gnn.ts` (TorchScript checkpoint)

## Inference

```python
from facet.models.st_gnn import SpatiotemporalGNNCorrection

context = context | SpatiotemporalGNNCorrection(
    checkpoint_path="training_output/<run>/exports/st_gnn.ts",
    context_epochs=7,
    epoch_samples=512,
)
```

The processor validates that the EEG context contains the 30-channel
set the model was trained on, and resamples native trigger-to-trigger
epochs to the model's 512-sample input.
