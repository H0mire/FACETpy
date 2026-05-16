# Evaluation Run: Cascaded DAE

## Identity

- model id: `cascaded_dae`
- run id: `holdout_v1`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `unified_holdout.artifact_corr` | 0.992279 |
| `unified_holdout.artifact_mae` | 1.728505e-04 |
| `unified_holdout.artifact_mse` | 5.559644e-08 |
| `unified_holdout.artifact_snr_db` | 18.059265 |
| `unified_holdout.clean_mae_after` | 1.728505e-04 |
| `unified_holdout.clean_mae_before` | 9.214178e-04 |
| `unified_holdout.clean_mse_after` | 5.559644e-08 |
| `unified_holdout.clean_mse_before` | 3.556096e-06 |
| `unified_holdout.clean_mse_reduction_pct` | 98.436588 |
| `unified_holdout.clean_snr_db_after` | 7.114318 |
| `unified_holdout.clean_snr_db_before` | -10.944946 |
| `unified_holdout.clean_snr_improvement_db` | 18.059264 |
| `unified_holdout.inference_seconds` | 0.490758 |
| `unified_holdout.n_channels` | 30 |
| `unified_holdout.n_examples` | 166 |
| `unified_holdout.residual_error_rms_ratio` | 0.125036 |
| `unified_holdout.rms_baseline_noisy_ratio` | 7.254694 |
| `unified_holdout.rms_recovery_distance` | 0.610342 |
| `unified_holdout.rms_recovery_ratio` | 1.532418 |
| `unified_holdout.samples_per_epoch` | 512 |
| `unified_holdout.sfreq_hz` | 4096.000000 |

## Interpretation

cascaded_dae on the unified holdout split: SNR improvement = +18.06 dB (before=-10.94 dB → after=+7.11 dB). Artifact correlation with ground truth = +0.9923. Residual RMS ratio (after / before) = 0.125.

## Artifacts

- `holdout_examples`: `/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/model_evaluations/cascaded_dae/holdout_v1/plots/holdout_examples.png`

## Limitations

- Unified holdout split: window-level seed=42 val_ratio=0.2 → 166 windows (4980 channel-windows).
- Metrics use canonical formulas from examples/evaluate_conv_tasnet.py.
- Target is AAS-corrected 'clean' — fidelity to AAS, not absolute ground truth.

## Configuration

```json
{
  "model_id": "cascaded_dae",
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/cascadeddenoisingautoencoderniazyprooffit_20260516_152448/exports/cascaded_dae.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "device": "cpu",
  "holdout_split_hash": "sha256:ddaa64a504e062fd",
  "holdout_seed": 42,
  "holdout_val_ratio": 0.2,
  "holdout_n_windows": 166,
  "family": "Autoencoder (cascaded MLP)",
  "notes": "(1,1,512) in \u2192 (1,1,512) artifact out. Per-segment demean. Glob picks newest matching training_output run."
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
    "clean_mse_after": 5.559644478125847e-08,
    "clean_mae_before": 0.0009214178426191211,
    "clean_mae_after": 0.00017285050125792623,
    "clean_snr_db_before": -10.9449462890625,
    "clean_snr_db_after": 7.114317893981934,
    "artifact_mse": 5.559644478125847e-08,
    "artifact_mae": 0.00017285050125792623,
    "artifact_corr": 0.992279256822137,
    "artifact_snr_db": 18.05926513671875,
    "residual_error_rms_ratio": 0.1250364924255337,
    "rms_recovery_ratio": 1.5324183702468872,
    "rms_recovery_distance": 0.6103419661521912,
    "rms_baseline_noisy_ratio": 7.25469446182251,
    "clean_mse_reduction_pct": 98.43658756863466,
    "clean_snr_improvement_db": 18.059264183044434,
    "inference_seconds": 0.490758,
    "device": "cpu"
  }
}
```
