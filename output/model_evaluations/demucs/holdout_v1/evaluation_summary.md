# Evaluation Run: Demucs

## Identity

- model id: `demucs`
- run id: `holdout_v1`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `unified_holdout.artifact_corr` | 0.999636 |
| `unified_holdout.artifact_mae` | 2.616063e-05 |
| `unified_holdout.artifact_mse` | 2.638945e-09 |
| `unified_holdout.artifact_snr_db` | 31.295431 |
| `unified_holdout.clean_mae_after` | 2.616063e-05 |
| `unified_holdout.clean_mae_before` | 9.214178e-04 |
| `unified_holdout.clean_mse_after` | 2.638945e-09 |
| `unified_holdout.clean_mse_before` | 3.556096e-06 |
| `unified_holdout.clean_mse_reduction_pct` | 99.925791 |
| `unified_holdout.clean_snr_db_after` | 20.350485 |
| `unified_holdout.clean_snr_db_before` | -10.944946 |
| `unified_holdout.clean_snr_improvement_db` | 31.295431 |
| `unified_holdout.inference_seconds` | 110.896163 |
| `unified_holdout.n_channels` | 30 |
| `unified_holdout.n_examples` | 166 |
| `unified_holdout.residual_error_rms_ratio` | 0.027241 |
| `unified_holdout.samples_per_epoch` | 512 |
| `unified_holdout.sfreq_hz` | 4096.000000 |

## Interpretation

demucs on the unified holdout split: SNR improvement = +31.30 dB (before=-10.94 dB → after=+20.35 dB). Artifact correlation with ground truth = +0.9996. Residual RMS ratio (after / before) = 0.027.

## Artifacts

- `holdout_examples`: `/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/model_evaluations/demucs/holdout_v1/plots/holdout_examples.png`

## Limitations

- Unified holdout split: window-level seed=42 val_ratio=0.2 → 166 windows (4980 channel-windows).
- Metrics use canonical formulas from examples/evaluate_conv_tasnet.py.
- Target is AAS-corrected 'clean' — fidelity to AAS, not absolute ground truth.

## Configuration

```json
{
  "model_id": "demucs",
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/demucsniazyprooffit_20260510_224653/exports/demucs.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "device": "cpu",
  "holdout_split_hash": "sha256:ddaa64a504e062fd",
  "holdout_seed": 42,
  "holdout_val_ratio": 0.2,
  "holdout_n_windows": 166,
  "family": "Audio (U-Net+LSTM)",
  "notes": "(1,1,7*512=3584) in \u2192 (1,1,3584) artifact out. Center [3*512:4*512] sliced. Single-mean demean across all 7 epochs."
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
    "clean_mse_after": 2.638944618382766e-09,
    "clean_mae_before": 0.0009214178426191211,
    "clean_mae_after": 2.6160631023230962e-05,
    "clean_snr_db_before": -10.9449462890625,
    "clean_snr_db_after": 20.35048484802246,
    "artifact_mse": 2.638944618382766e-09,
    "artifact_mae": 2.6160631023230962e-05,
    "artifact_corr": 0.9996360414850605,
    "artifact_snr_db": 31.29543113708496,
    "residual_error_rms_ratio": 0.0272413352211958,
    "clean_mse_reduction_pct": 99.92579095950656,
    "clean_snr_improvement_db": 31.29543113708496,
    "inference_seconds": 110.896163,
    "device": "cpu"
  }
}
```
