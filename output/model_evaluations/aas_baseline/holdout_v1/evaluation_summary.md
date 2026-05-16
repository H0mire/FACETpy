# Evaluation Run: AAS (Production)

## Identity

- model id: `aas_baseline`
- run id: `holdout_v1`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `unified_holdout.artifact_corr` | 0.031263 |
| `unified_holdout.artifact_mae` | 9.037281e-04 |
| `unified_holdout.artifact_mse` | 3.552622e-06 |
| `unified_holdout.artifact_snr_db` | 0.004245 |
| `unified_holdout.clean_mae_after` | 9.037281e-04 |
| `unified_holdout.clean_mae_before` | 9.214178e-04 |
| `unified_holdout.clean_mse_after` | 3.552622e-06 |
| `unified_holdout.clean_mse_before` | 3.556096e-06 |
| `unified_holdout.clean_mse_reduction_pct` | 0.097686 |
| `unified_holdout.clean_snr_db_after` | -10.940701 |
| `unified_holdout.clean_snr_db_before` | -10.944946 |
| `unified_holdout.clean_snr_improvement_db` | 0.004246 |
| `unified_holdout.inference_seconds` | 0.153144 |
| `unified_holdout.n_channels` | 30 |
| `unified_holdout.n_examples` | 166 |
| `unified_holdout.residual_error_rms_ratio` | 0.999511 |
| `unified_holdout.rms_baseline_noisy_ratio` | 7.254694 |
| `unified_holdout.rms_recovery_distance` | 6.248655 |
| `unified_holdout.rms_recovery_ratio` | 7.248656 |
| `unified_holdout.samples_per_epoch` | 512 |
| `unified_holdout.sfreq_hz` | 4096.000000 |

## Interpretation

aas_baseline on unified holdout: SNR Δ = +0.00 dB, art corr +0.0313, inference 0.153s for 4980 channel-windows.

## Artifacts

- No external artifacts recorded.

## Limitations

- Baseline reference — no trained model parameters.
- Compared against AAS-derived clean_center target.

## Configuration

```json
{
  "model_id": "aas_baseline",
  "family": "Baseline (AAS)",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "holdout_n_windows": 166,
  "note": "AAS baseline benchmark \u2014 not a learned model"
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
    "clean_mse_after": 3.5526220472092973e-06,
    "clean_mae_before": 0.0009214178426191211,
    "clean_mae_after": 0.0009037281270138919,
    "clean_snr_db_before": -10.9449462890625,
    "clean_snr_db_after": -10.94070053100586,
    "artifact_mse": 3.5526220472092973e-06,
    "artifact_mae": 0.0009037281270138919,
    "artifact_corr": 0.031262693375537935,
    "artifact_snr_db": 0.004244776908308268,
    "residual_error_rms_ratio": 0.9995114382648786,
    "rms_recovery_ratio": 7.248655796051025,
    "rms_recovery_distance": 6.248654842376709,
    "rms_baseline_noisy_ratio": 7.25469446182251,
    "clean_mse_reduction_pct": 0.09768620273528095,
    "clean_snr_improvement_db": 0.004245758056640625,
    "inference_seconds": 0.15314383299846668
  }
}
```
