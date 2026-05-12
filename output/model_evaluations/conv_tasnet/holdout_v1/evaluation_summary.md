# Evaluation Run: Conv-TasNet

## Identity

- model id: `conv_tasnet`
- run id: `holdout_v1`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `unified_holdout.artifact_corr` | 0.997335 |
| `unified_holdout.artifact_mae` | 7.061777e-05 |
| `unified_holdout.artifact_mse` | 1.892979e-08 |
| `unified_holdout.artifact_snr_db` | 22.738277 |
| `unified_holdout.clean_mae_after` | 7.061777e-05 |
| `unified_holdout.clean_mae_before` | 9.214178e-04 |
| `unified_holdout.clean_mse_after` | 1.892979e-08 |
| `unified_holdout.clean_mse_before` | 3.556096e-06 |
| `unified_holdout.clean_mse_reduction_pct` | 99.467681 |
| `unified_holdout.clean_snr_db_after` | 11.793331 |
| `unified_holdout.clean_snr_db_before` | -10.944946 |
| `unified_holdout.clean_snr_improvement_db` | 22.738277 |
| `unified_holdout.inference_seconds` | 675.617146 |
| `unified_holdout.n_channels` | 30 |
| `unified_holdout.n_examples` | 166 |
| `unified_holdout.residual_error_rms_ratio` | 0.072960 |
| `unified_holdout.samples_per_epoch` | 512 |
| `unified_holdout.sfreq_hz` | 4096.000000 |

## Interpretation

conv_tasnet on the unified holdout split: SNR improvement = +22.74 dB (before=-10.94 dB → after=+11.79 dB). Artifact correlation with ground truth = +0.9973. Residual RMS ratio (after / before) = 0.073.

## Artifacts

- `holdout_examples`: `/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/model_evaluations/conv_tasnet/holdout_v1/plots/holdout_examples.png`

## Limitations

- Unified holdout split: window-level seed=42 val_ratio=0.2 → 166 windows (4980 channel-windows).
- Metrics use canonical formulas from examples/evaluate_conv_tasnet.py.
- Target is AAS-corrected 'clean' — fidelity to AAS, not absolute ground truth.

## Configuration

```json
{
  "model_id": "conv_tasnet",
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/convtasnetniazyprooffit_20260510_202818/exports/conv_tasnet.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "device": "cpu",
  "holdout_split_hash": "sha256:ddaa64a504e062fd",
  "holdout_seed": 42,
  "holdout_val_ratio": 0.2,
  "holdout_n_windows": 166,
  "family": "Audio (TCN)",
  "notes": "(1,1,512) in \u2192 (1,2,512) sources out. source[1] = artifact."
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
    "clean_mse_after": 1.8929787515276075e-08,
    "clean_mae_before": 0.0009214178426191211,
    "clean_mae_after": 7.061777432681993e-05,
    "clean_snr_db_before": -10.9449462890625,
    "clean_snr_db_after": 11.793331146240234,
    "artifact_mse": 1.8929787515276075e-08,
    "artifact_mae": 7.061777432681993e-05,
    "artifact_corr": 0.9973348568204654,
    "artifact_snr_db": 22.738277435302734,
    "residual_error_rms_ratio": 0.07296021925890112,
    "clean_mse_reduction_pct": 99.46768061805162,
    "clean_snr_improvement_db": 22.738277435302734,
    "inference_seconds": 675.617146,
    "device": "cpu"
  }
}
```
