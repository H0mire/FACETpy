# Evaluation Run: AAS (Simple 6-neighbor mean)

## Identity

- model id: `aas_naive_6nn`
- run id: `holdout_v1`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `unified_holdout.artifact_corr` | 0.937542 |
| `unified_holdout.artifact_mae` | 1.811914e-04 |
| `unified_holdout.artifact_mse` | 4.318700e-07 |
| `unified_holdout.artifact_snr_db` | 9.156204 |
| `unified_holdout.clean_mae_after` | 1.811914e-04 |
| `unified_holdout.clean_mae_before` | 9.214178e-04 |
| `unified_holdout.clean_mse_after` | 4.318700e-07 |
| `unified_holdout.clean_mse_before` | 3.556096e-06 |
| `unified_holdout.clean_mse_reduction_pct` | 87.855502 |
| `unified_holdout.clean_snr_db_after` | -1.788742 |
| `unified_holdout.clean_snr_db_before` | -10.944946 |
| `unified_holdout.clean_snr_improvement_db` | 9.156205 |
| `unified_holdout.inference_seconds` | 0.010043 |
| `unified_holdout.n_channels` | 30 |
| `unified_holdout.n_examples` | 166 |
| `unified_holdout.residual_error_rms_ratio` | 0.348490 |
| `unified_holdout.rms_baseline_noisy_ratio` | 7.254694 |
| `unified_holdout.rms_recovery_distance` | 1.081170 |
| `unified_holdout.rms_recovery_ratio` | 1.855476 |
| `unified_holdout.samples_per_epoch` | 512 |
| `unified_holdout.sfreq_hz` | 4096.000000 |

## Interpretation

aas_naive_6nn on unified holdout: SNR Δ = +9.16 dB, art corr +0.9375, inference 0.010s for 4980 channel-windows.

## Artifacts

- No external artifacts recorded.

## Limitations

- Baseline reference — no trained model parameters.
- Compared against AAS-derived clean_center target.

## Configuration

```json
{
  "model_id": "aas_naive_6nn",
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
    "clean_mse_after": 4.3186997800148674e-07,
    "clean_mae_before": 0.0009214178426191211,
    "clean_mae_after": 0.00018119136802852154,
    "clean_snr_db_before": -10.9449462890625,
    "clean_snr_db_after": -1.7887415885925293,
    "artifact_mse": 4.3186997800148674e-07,
    "artifact_mae": 0.00018119136802852154,
    "artifact_corr": 0.9375421822491363,
    "artifact_snr_db": 9.156204223632812,
    "residual_error_rms_ratio": 0.3484895608358443,
    "rms_recovery_ratio": 1.8554763793945312,
    "rms_recovery_distance": 1.081169843673706,
    "rms_baseline_noisy_ratio": 7.25469446182251,
    "clean_mse_reduction_pct": 87.85550236175165,
    "clean_snr_improvement_db": 9.15620470046997,
    "inference_seconds": 0.010043166999821551
  }
}
```
