# Evaluation Run: Denoise-Mamba

## Identity

- model id: `denoise_mamba`
- run id: `holdout_v1`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `unified_holdout.artifact_corr` | 0.961371 |
| `unified_holdout.artifact_mae` | 2.297265e-04 |
| `unified_holdout.artifact_mse` | 2.695126e-07 |
| `unified_holdout.artifact_snr_db` | 11.203945 |
| `unified_holdout.clean_mae_after` | 2.297265e-04 |
| `unified_holdout.clean_mae_before` | 9.214178e-04 |
| `unified_holdout.clean_mse_after` | 2.695126e-07 |
| `unified_holdout.clean_mse_before` | 3.556096e-06 |
| `unified_holdout.clean_mse_reduction_pct` | 92.421111 |
| `unified_holdout.clean_snr_db_after` | 0.258998 |
| `unified_holdout.clean_snr_db_before` | -10.944946 |
| `unified_holdout.clean_snr_improvement_db` | 11.203945 |
| `unified_holdout.inference_seconds` | 83.326378 |
| `unified_holdout.n_channels` | 30 |
| `unified_holdout.n_examples` | 166 |
| `unified_holdout.residual_error_rms_ratio` | 0.275298 |
| `unified_holdout.samples_per_epoch` | 512 |
| `unified_holdout.sfreq_hz` | 4096.000000 |

## Interpretation

denoise_mamba on the unified holdout split: SNR improvement = +11.20 dB (before=-10.94 dB → after=+0.26 dB). Artifact correlation with ground truth = +0.9614. Residual RMS ratio (after / before) = 0.275.

## Artifacts

- `holdout_examples`: `/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/model_evaluations/denoise_mamba/holdout_v1/plots/holdout_examples.png`

## Limitations

- Unified holdout split: window-level seed=42 val_ratio=0.2 → 166 windows (4980 channel-windows).
- Metrics use canonical formulas from examples/evaluate_conv_tasnet.py.
- Target is AAS-corrected 'clean' — fidelity to AAS, not absolute ground truth.

## Configuration

```json
{
  "model_id": "denoise_mamba",
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/denoisemambaniazyprooffit_20260510_193847/exports/denoise_mamba.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "device": "cpu",
  "holdout_split_hash": "sha256:ddaa64a504e062fd",
  "holdout_seed": 42,
  "holdout_val_ratio": 0.2,
  "holdout_n_windows": 166,
  "family": "SSM",
  "notes": "(1,1,512) in \u2192 (1,1,512) artifact out. Per-segment demean."
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
    "clean_mse_after": 2.6951255449603195e-07,
    "clean_mae_before": 0.0009214178426191211,
    "clean_mae_after": 0.00022972654551267624,
    "clean_snr_db_before": -10.9449462890625,
    "clean_snr_db_after": 0.2589983642101288,
    "artifact_mse": 2.6951255449603195e-07,
    "artifact_mae": 0.00022972654551267624,
    "artifact_corr": 0.9613709163272423,
    "artifact_snr_db": 11.20394515991211,
    "residual_error_rms_ratio": 0.27529780875061355,
    "clean_mse_reduction_pct": 92.4211111022307,
    "clean_snr_improvement_db": 11.203944653272629,
    "inference_seconds": 83.326378,
    "device": "cpu"
  }
}
```
