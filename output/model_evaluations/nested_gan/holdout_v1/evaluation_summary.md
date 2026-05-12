# Evaluation Run: Nested-GAN

## Identity

- model id: `nested_gan`
- run id: `holdout_v1`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `unified_holdout.artifact_corr` | 0.974561 |
| `unified_holdout.artifact_mae` | 3.376183e-04 |
| `unified_holdout.artifact_mse` | 2.397806e-07 |
| `unified_holdout.artifact_snr_db` | 11.711596 |
| `unified_holdout.clean_mae_after` | 3.376183e-04 |
| `unified_holdout.clean_mae_before` | 9.214178e-04 |
| `unified_holdout.clean_mse_after` | 2.397806e-07 |
| `unified_holdout.clean_mse_before` | 3.556096e-06 |
| `unified_holdout.clean_mse_reduction_pct` | 93.257196 |
| `unified_holdout.clean_snr_db_after` | 0.766649 |
| `unified_holdout.clean_snr_db_before` | -10.944946 |
| `unified_holdout.clean_snr_improvement_db` | 11.711595 |
| `unified_holdout.inference_seconds` | 128.175710 |
| `unified_holdout.n_channels` | 30 |
| `unified_holdout.n_examples` | 166 |
| `unified_holdout.residual_error_rms_ratio` | 0.259669 |
| `unified_holdout.samples_per_epoch` | 512 |
| `unified_holdout.sfreq_hz` | 4096.000000 |

## Interpretation

nested_gan on the unified holdout split: SNR improvement = +11.71 dB (before=-10.94 dB → after=+0.77 dB). Artifact correlation with ground truth = +0.9746. Residual RMS ratio (after / before) = 0.260.

## Artifacts

- `holdout_examples`: `/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/model_evaluations/nested_gan/holdout_v1/plots/holdout_examples.png`

## Limitations

- Unified holdout split: window-level seed=42 val_ratio=0.2 → 166 windows (4980 channel-windows).
- Metrics use canonical formulas from examples/evaluate_conv_tasnet.py.
- Target is AAS-corrected 'clean' — fidelity to AAS, not absolute ground truth.

## Configuration

```json
{
  "model_id": "nested_gan",
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/nestedganniazyprooffit_20260510_222546/exports/nested_gan.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "device": "cpu",
  "holdout_split_hash": "sha256:ddaa64a504e062fd",
  "holdout_seed": 42,
  "holdout_val_ratio": 0.2,
  "holdout_n_windows": 166,
  "family": "GAN (TF+Time)",
  "notes": "(1,7,1,512) in \u2192 (1,1,512) artifact out. Per-epoch demean."
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
    "clean_mse_after": 2.397805758391769e-07,
    "clean_mae_before": 0.0009214178426191211,
    "clean_mae_after": 0.00033761828672140837,
    "clean_snr_db_before": -10.9449462890625,
    "clean_snr_db_after": 0.7666485905647278,
    "artifact_mse": 2.397805758391769e-07,
    "artifact_mae": 0.00033761828672140837,
    "artifact_corr": 0.9745608048549789,
    "artifact_snr_db": 11.71159553527832,
    "residual_error_rms_ratio": 0.2596690806044757,
    "clean_mse_reduction_pct": 93.25719594945613,
    "clean_snr_improvement_db": 11.711594879627228,
    "inference_seconds": 128.17571,
    "device": "cpu"
  }
}
```
