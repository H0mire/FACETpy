# Evaluation Run: ST-GNN

## Identity

- model id: `st_gnn`
- run id: `holdout_v1`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `unified_holdout.artifact_corr` | 0.959546 |
| `unified_holdout.artifact_mae` | 2.699292e-04 |
| `unified_holdout.artifact_mse` | 2.821746e-07 |
| `unified_holdout.artifact_snr_db` | 11.004557 |
| `unified_holdout.clean_mae_after` | 2.699292e-04 |
| `unified_holdout.clean_mae_before` | 9.214178e-04 |
| `unified_holdout.clean_mse_after` | 2.821746e-07 |
| `unified_holdout.clean_mse_before` | 3.556096e-06 |
| `unified_holdout.clean_mse_reduction_pct` | 92.065046 |
| `unified_holdout.clean_snr_db_after` | 0.059610 |
| `unified_holdout.clean_snr_db_before` | -10.944946 |
| `unified_holdout.clean_snr_improvement_db` | 11.004556 |
| `unified_holdout.inference_seconds` | 5.347203 |
| `unified_holdout.n_channels` | 30 |
| `unified_holdout.n_examples` | 166 |
| `unified_holdout.residual_error_rms_ratio` | 0.281691 |
| `unified_holdout.rms_baseline_noisy_ratio` | 7.254694 |
| `unified_holdout.rms_recovery_distance` | 0.708038 |
| `unified_holdout.rms_recovery_ratio` | 0.343327 |
| `unified_holdout.samples_per_epoch` | 512 |
| `unified_holdout.sfreq_hz` | 4096.000000 |

## Interpretation

st_gnn on the unified holdout split: SNR improvement = +11.00 dB (before=-10.94 dB → after=+0.06 dB). Artifact correlation with ground truth = +0.9595. Residual RMS ratio (after / before) = 0.282.

## Artifacts

- `holdout_examples`: `/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/model_evaluations/st_gnn/holdout_v1/plots/holdout_examples.png`

## Limitations

- Unified holdout split: window-level seed=42 val_ratio=0.2 → 166 windows (4980 channel-windows).
- Metrics use canonical formulas from examples/evaluate_conv_tasnet.py.
- Target is AAS-corrected 'clean' — fidelity to AAS, not absolute ground truth.

## Configuration

```json
{
  "model_id": "st_gnn",
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/spatiotemporalgnnniazyprooffit_20260510_211512/exports/st_gnn.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "device": "cpu",
  "holdout_split_hash": "sha256:ddaa64a504e062fd",
  "holdout_seed": 42,
  "holdout_val_ratio": 0.2,
  "holdout_n_windows": 166,
  "family": "Graph (GNN)",
  "notes": "(1,7,30,512) in \u2192 (1,30,512) artifact out. Channel order is load-bearing."
}
```

## Raw Metrics

```json
{
  "unified_holdout": {
    "n_examples": 166,
    "n_channels": 30,
    "samples_per_epoch": 512,
    "sfreq_hz": 4096.0,
    "clean_mse_before": 3.556095862222719e-06,
    "clean_mse_after": 2.821745681558241e-07,
    "clean_mae_before": 0.0009214178426191211,
    "clean_mae_after": 0.0002699291508179158,
    "clean_snr_db_before": -10.9449462890625,
    "clean_snr_db_after": 0.0596097894012928,
    "artifact_mse": 2.821745681558241e-07,
    "artifact_mae": 0.0002699291508179158,
    "artifact_corr": 0.959546257527632,
    "artifact_snr_db": 11.004556655883789,
    "residual_error_rms_ratio": 0.2816905038575326,
    "rms_recovery_ratio": 0.3433266878128052,
    "rms_recovery_distance": 0.7080376148223877,
    "rms_baseline_noisy_ratio": 7.25469446182251,
    "clean_mse_reduction_pct": 92.06504607613553,
    "clean_snr_improvement_db": 11.004556078463793,
    "inference_seconds": 5.347203,
    "device": "cpu"
  }
}
```
