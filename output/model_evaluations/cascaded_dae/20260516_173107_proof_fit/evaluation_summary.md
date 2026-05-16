# Evaluation Run: Cascaded DAE

## Identity

- model id: `cascaded_dae`
- run id: `20260516_173107_proof_fit`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `synthetic_niazy_proof_fit_val_split.artifact_corr` | 0.991735 |
| `synthetic_niazy_proof_fit_val_split.artifact_mae` | 1.746195e-04 |
| `synthetic_niazy_proof_fit_val_split.artifact_mse` | 5.758098e-08 |
| `synthetic_niazy_proof_fit_val_split.artifact_snr_db` | 17.786373 |
| `synthetic_niazy_proof_fit_val_split.clean_mae_after` | 1.746195e-04 |
| `synthetic_niazy_proof_fit_val_split.clean_mae_before` | 9.077009e-04 |
| `synthetic_niazy_proof_fit_val_split.clean_mse_after` | 5.758098e-08 |
| `synthetic_niazy_proof_fit_val_split.clean_mse_before` | 3.458729e-06 |
| `synthetic_niazy_proof_fit_val_split.clean_mse_reduction_pct` | 98.335198 |
| `synthetic_niazy_proof_fit_val_split.clean_snr_db_after` | 6.177318 |
| `synthetic_niazy_proof_fit_val_split.clean_snr_db_before` | -11.609056 |
| `synthetic_niazy_proof_fit_val_split.clean_snr_improvement_db` | 17.786375 |
| `synthetic_niazy_proof_fit_val_split.inference_seconds` | 0.788747 |
| `synthetic_niazy_proof_fit_val_split.input_mean_removed` | True |
| `synthetic_niazy_proof_fit_val_split.n_examples` | 4998 |
| `synthetic_niazy_proof_fit_val_split.predicted_clean_corr` | 0.884053 |
| `synthetic_niazy_proof_fit_val_split.predicted_clean_mse` | 5.758098e-08 |
| `synthetic_niazy_proof_fit_val_split.predicted_source_sum_mse` | 7.395822e-23 |
| `synthetic_niazy_proof_fit_val_split.prediction_mean_removed` | True |
| `synthetic_niazy_proof_fit_val_split.residual_error_rms_ratio` | 0.129027 |
| `synthetic_niazy_proof_fit_val_split.samples_per_epoch` | 512 |

## Interpretation

cascaded_dae on Niazy proof-fit val split (4998 pairs): clean SNR Δ = +17.79 dB (before -11.61 → after +6.18), artifact correlation = +0.9917, residual RMS ratio = 0.129.

## Artifacts

- `val_examples`: `output/model_evaluations/cascaded_dae/20260516_173107_proof_fit/plots/cascaded_dae_val_examples.png`

## Limitations

- Per-channel-pair val split (seed=42, val_ratio=0.2). Identical to the split used by evaluate_conv_tasnet.py / evaluate_demucs.py — directly comparable.
- Targets are AAS-corrected 'clean' and AAS-estimated 'artifact' (see dataset_metadata.json) — fidelity to AAS, not absolute ground truth.

## Configuration

```json
{
  "model_id": "cascaded_dae",
  "model_name": "Cascaded DAE",
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/cascadeddenoisingautoencoderniazyprooffit_20260516_152448/exports/cascaded_dae.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "device": "cpu",
  "batch_size": 256,
  "val_ratio": 0.2,
  "seed": 42,
  "n_pairs": 4998,
  "n_channels": 30,
  "n_samples": 512
}
```

## Raw Metrics

```json
{
  "synthetic_niazy_proof_fit_val_split": {
    "n_examples": 4998,
    "samples_per_epoch": 512,
    "input_mean_removed": true,
    "prediction_mean_removed": true,
    "clean_mse_before": 3.4587285426823655e-06,
    "clean_mse_after": 5.758097998409539e-08,
    "clean_mae_before": 0.0009077009162865579,
    "clean_mae_after": 0.00017461949028074741,
    "clean_snr_db_before": -11.60905647277832,
    "clean_snr_db_after": 6.177318096160889,
    "artifact_mse": 5.758098353680907e-08,
    "artifact_mae": 0.00017461949028074741,
    "artifact_corr": 0.991734618778353,
    "artifact_snr_db": 17.786373138427734,
    "predicted_clean_mse": 5.758097998409539e-08,
    "predicted_clean_corr": 0.8840530919085338,
    "predicted_source_sum_mse": 7.395821528670484e-23,
    "residual_error_rms_ratio": 0.1290272021135673,
    "clean_mse_reduction_pct": 98.33519805692993,
    "clean_snr_improvement_db": 17.78637456893921,
    "inference_seconds": 0.788747,
    "device": "cpu"
  }
}
```
