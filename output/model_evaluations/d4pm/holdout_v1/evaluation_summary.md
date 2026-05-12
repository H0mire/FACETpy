# Evaluation Run: D4PM

## Identity

- model id: `d4pm`
- run id: `holdout_v1`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `unified_holdout.artifact_corr` | 0.926502 |
| `unified_holdout.artifact_mae` | 7.630310e-04 |
| `unified_holdout.artifact_mse` | 1.175873e-06 |
| `unified_holdout.artifact_snr_db` | 4.806132 |
| `unified_holdout.clean_mae_after` | 7.630310e-04 |
| `unified_holdout.clean_mae_before` | 9.214178e-04 |
| `unified_holdout.clean_mse_after` | 1.175873e-06 |
| `unified_holdout.clean_mse_before` | 3.556096e-06 |
| `unified_holdout.clean_mse_reduction_pct` | 66.933610 |
| `unified_holdout.clean_snr_db_after` | -6.138813 |
| `unified_holdout.clean_snr_db_before` | -10.944946 |
| `unified_holdout.clean_snr_improvement_db` | 4.806133 |
| `unified_holdout.inference_seconds` | 1218.604637 |
| `unified_holdout.n_channels` | 30 |
| `unified_holdout.n_examples` | 166 |
| `unified_holdout.residual_error_rms_ratio` | 0.575034 |
| `unified_holdout.samples_per_epoch` | 512 |
| `unified_holdout.sfreq_hz` | 4096.000000 |

## Interpretation

d4pm on the unified holdout split: SNR improvement = +4.81 dB (before=-10.94 dB → after=-6.14 dB). Artifact correlation with ground truth = +0.9265. Residual RMS ratio (after / before) = 0.575.

## Artifacts

- `holdout_examples`: `/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/model_evaluations/d4pm/holdout_v1/plots/holdout_examples.png`

## Limitations

- Unified holdout split: window-level seed=42 val_ratio=0.2 → 166 windows (4980 channel-windows).
- Metrics use canonical formulas from examples/evaluate_conv_tasnet.py.
- Target is AAS-corrected 'clean' — fidelity to AAS, not absolute ground truth.

## Configuration

```json
{
  "model_id": "d4pm",
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/d4pmartifactdiffusionniazyprooffit_20260510_201242/exports/d4pm.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "device": "cpu",
  "holdout_split_hash": "sha256:ddaa64a504e062fd",
  "holdout_seed": 42,
  "holdout_val_ratio": 0.2,
  "holdout_n_windows": 166,
  "family": "Diffusion",
  "notes": "DDPM reverse loop wrapped in TS. Slow on CPU \u2014 recommend --device cuda."
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
    "clean_mse_after": 1.1758725122490432e-06,
    "clean_mae_before": 0.0009214178426191211,
    "clean_mae_after": 0.000763030955567956,
    "clean_snr_db_before": -10.9449462890625,
    "clean_snr_db_after": -6.138813495635986,
    "artifact_mse": 1.1758725122490432e-06,
    "artifact_mae": 0.000763030955567956,
    "artifact_corr": 0.9265018792108938,
    "artifact_snr_db": 4.806131839752197,
    "residual_error_rms_ratio": 0.5750338286096541,
    "clean_mse_reduction_pct": 66.93361040289653,
    "clean_snr_improvement_db": 4.806132793426514,
    "inference_seconds": 1218.604637,
    "device": "cpu"
  }
}
```
