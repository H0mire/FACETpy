# Evaluation Run: IC-U-Net

## Identity

- model id: `ic_unet`
- run id: `holdout_v1`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `unified_holdout.artifact_corr` | 0.961254 |
| `unified_holdout.artifact_mae` | 2.322200e-04 |
| `unified_holdout.artifact_mse` | 2.754390e-07 |
| `unified_holdout.artifact_snr_db` | 11.109480 |
| `unified_holdout.clean_mae_after` | 2.322200e-04 |
| `unified_holdout.clean_mae_before` | 9.214178e-04 |
| `unified_holdout.clean_mse_after` | 2.754390e-07 |
| `unified_holdout.clean_mse_before` | 3.556096e-06 |
| `unified_holdout.clean_mse_reduction_pct` | 92.254455 |
| `unified_holdout.clean_snr_db_after` | 0.164534 |
| `unified_holdout.clean_snr_db_before` | -10.944946 |
| `unified_holdout.clean_snr_improvement_db` | 11.109480 |
| `unified_holdout.inference_seconds` | 1.691554 |
| `unified_holdout.n_channels` | 30 |
| `unified_holdout.n_examples` | 166 |
| `unified_holdout.residual_error_rms_ratio` | 0.278308 |
| `unified_holdout.samples_per_epoch` | 512 |
| `unified_holdout.sfreq_hz` | 4096.000000 |

## Interpretation

ic_unet on the unified holdout split: SNR improvement = +11.11 dB (before=-10.94 dB → after=+0.16 dB). Artifact correlation with ground truth = +0.9613. Residual RMS ratio (after / before) = 0.278.

## Artifacts

- `holdout_examples`: `/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/model_evaluations/ic_unet/holdout_v1/plots/holdout_examples.png`

## Limitations

- Unified holdout split: window-level seed=42 val_ratio=0.2 → 166 windows (4980 channel-windows).
- Metrics use canonical formulas from examples/evaluate_conv_tasnet.py.
- Target is AAS-corrected 'clean' — fidelity to AAS, not absolute ground truth.

## Configuration

```json
{
  "model_id": "ic_unet",
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/icunetniazyprooffit_20260510_223556/exports/ic_unet.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "device": "cpu",
  "holdout_split_hash": "sha256:ddaa64a504e062fd",
  "holdout_seed": 42,
  "holdout_val_ratio": 0.2,
  "holdout_n_windows": 166,
  "family": "Discriminative + ICA",
  "notes": "(1,30,7*512) in \u2192 (1,30,7*512) artifact out, center epoch sliced."
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
    "clean_mse_after": 2.7543902092475037e-07,
    "clean_mae_before": 0.0009214178426191211,
    "clean_mae_after": 0.00023222003073897213,
    "clean_snr_db_before": -10.9449462890625,
    "clean_snr_db_after": 0.16453388333320618,
    "artifact_mse": 2.7543902092475037e-07,
    "artifact_mae": 0.00023222003073897213,
    "artifact_corr": 0.9612539275588182,
    "artifact_snr_db": 11.109479904174805,
    "residual_error_rms_ratio": 0.27830819849681765,
    "clean_mse_reduction_pct": 92.2544545592596,
    "clean_snr_improvement_db": 11.109480172395706,
    "inference_seconds": 1.691554,
    "device": "cpu"
  }
}
```
