# Evaluation Run: DPAE

## Identity

- model id: `dpae`
- run id: `holdout_v1`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `unified_holdout.artifact_corr` | 0.909176 |
| `unified_holdout.artifact_mae` | 6.776283e-04 |
| `unified_holdout.artifact_mse` | 6.644845e-07 |
| `unified_holdout.artifact_snr_db` | 7.284886 |
| `unified_holdout.clean_mae_after` | 6.776283e-04 |
| `unified_holdout.clean_mae_before` | 9.214178e-04 |
| `unified_holdout.clean_mse_after` | 6.644845e-07 |
| `unified_holdout.clean_mse_before` | 3.556096e-06 |
| `unified_holdout.clean_mse_reduction_pct` | 81.314213 |
| `unified_holdout.clean_snr_db_after` | -3.660060 |
| `unified_holdout.clean_snr_db_before` | -10.944946 |
| `unified_holdout.clean_snr_improvement_db` | 7.284886 |
| `unified_holdout.inference_seconds` | 22.806151 |
| `unified_holdout.n_channels` | 30 |
| `unified_holdout.n_examples` | 166 |
| `unified_holdout.residual_error_rms_ratio` | 0.432271 |
| `unified_holdout.samples_per_epoch` | 512 |
| `unified_holdout.sfreq_hz` | 4096.000000 |

## Interpretation

dpae on the unified holdout split: SNR improvement = +7.28 dB (before=-10.94 dB → after=-3.66 dB). Artifact correlation with ground truth = +0.9092. Residual RMS ratio (after / before) = 0.432.

## Artifacts

- `holdout_examples`: `/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/model_evaluations/dpae/holdout_v1/plots/holdout_examples.png`

## Limitations

- Unified holdout split: window-level seed=42 val_ratio=0.2 → 166 windows (4980 channel-windows).
- Metrics use canonical formulas from examples/evaluate_conv_tasnet.py.
- Target is AAS-corrected 'clean' — fidelity to AAS, not absolute ground truth.

## Configuration

```json
{
  "model_id": "dpae",
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/dualpathwayautoencoderniazyprooffit_20260510_192929/exports/dpae.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "device": "cpu",
  "holdout_split_hash": "sha256:ddaa64a504e062fd",
  "holdout_seed": 42,
  "holdout_val_ratio": 0.2,
  "holdout_n_windows": 166,
  "family": "Discriminative",
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
    "clean_mse_after": 6.644845029768476e-07,
    "clean_mae_before": 0.0009214178426191211,
    "clean_mae_after": 0.0006776283262297511,
    "clean_snr_db_before": -10.9449462890625,
    "clean_snr_db_after": -3.660060167312622,
    "artifact_mse": 6.644845029768476e-07,
    "artifact_mae": 0.0006776283262297511,
    "artifact_corr": 0.9091759590173892,
    "artifact_snr_db": 7.284886360168457,
    "residual_error_rms_ratio": 0.43227058294724463,
    "clean_mse_reduction_pct": 81.31421286934844,
    "clean_snr_improvement_db": 7.284886121749878,
    "inference_seconds": 22.806151,
    "device": "cpu"
  }
}
```
