# Evaluation Run: Cascaded Context DAE

## Identity

- model id: `cascaded_context_dae`
- run id: `20260516_173107_proof_fit`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `synthetic_niazy_proof_fit_val_split.artifact_corr` | 0.993487 |
| `synthetic_niazy_proof_fit_val_split.artifact_mae` | 1.614923e-04 |
| `synthetic_niazy_proof_fit_val_split.artifact_mse` | 4.520977e-08 |
| `synthetic_niazy_proof_fit_val_split.artifact_snr_db` | 18.836842 |
| `synthetic_niazy_proof_fit_val_split.clean_mae_after` | 1.614923e-04 |
| `synthetic_niazy_proof_fit_val_split.clean_mae_before` | 9.077009e-04 |
| `synthetic_niazy_proof_fit_val_split.clean_mse_after` | 4.520977e-08 |
| `synthetic_niazy_proof_fit_val_split.clean_mse_before` | 3.458729e-06 |
| `synthetic_niazy_proof_fit_val_split.clean_mse_reduction_pct` | 98.692879 |
| `synthetic_niazy_proof_fit_val_split.clean_snr_db_after` | 7.227786 |
| `synthetic_niazy_proof_fit_val_split.clean_snr_db_before` | -11.609056 |
| `synthetic_niazy_proof_fit_val_split.clean_snr_improvement_db` | 18.836842 |
| `synthetic_niazy_proof_fit_val_split.inference_seconds` | 0.548405 |
| `synthetic_niazy_proof_fit_val_split.input_mean_removed` | True |
| `synthetic_niazy_proof_fit_val_split.n_examples` | 4998 |
| `synthetic_niazy_proof_fit_val_split.predicted_clean_corr` | 0.911242 |
| `synthetic_niazy_proof_fit_val_split.predicted_clean_mse` | 4.520977e-08 |
| `synthetic_niazy_proof_fit_val_split.predicted_source_sum_mse` | 7.435591e-23 |
| `synthetic_niazy_proof_fit_val_split.prediction_mean_removed` | True |
| `synthetic_niazy_proof_fit_val_split.residual_error_rms_ratio` | 0.114329 |
| `synthetic_niazy_proof_fit_val_split.samples_per_epoch` | 512 |

## Interpretation

cascaded_context_dae on Niazy proof-fit val split (4998 pairs): clean SNR Δ = +18.84 dB (before -11.61 → after +7.23), artifact correlation = +0.9935, residual RMS ratio = 0.114.

## Artifacts

- `val_examples`: `output/model_evaluations/cascaded_context_dae/20260516_173107_proof_fit/plots/cascaded_context_dae_val_examples.png`

## Limitations

- Per-channel-pair val split (seed=42, val_ratio=0.2). Identical to the split used by evaluate_conv_tasnet.py / evaluate_demucs.py — directly comparable.
- Targets are AAS-corrected 'clean' and AAS-estimated 'artifact' (see dataset_metadata.json) — fidelity to AAS, not absolute ground truth.

## Configuration

```json
{
  "model_id": "cascaded_context_dae",
  "model_name": "Cascaded Context DAE",
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/cascadedcontextdenoisingautoencoderniazyprooffit_20260516_152603/exports/cascaded_context_dae.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "device": "cpu",
  "batch_size": 128,
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
    "clean_mse_after": 4.520977014976779e-08,
    "clean_mae_before": 0.0009077009162865579,
    "clean_mae_after": 0.0001614922657608986,
    "clean_snr_db_before": -11.60905647277832,
    "clean_snr_db_after": 7.227785587310791,
    "artifact_mse": 4.520977014976779e-08,
    "artifact_mae": 0.0001614922657608986,
    "artifact_corr": 0.9934873809929842,
    "artifact_snr_db": 18.836841583251953,
    "predicted_clean_mse": 4.520977014976779e-08,
    "predicted_clean_corr": 0.9112421993776192,
    "predicted_source_sum_mse": 7.435590846801934e-23,
    "residual_error_rms_ratio": 0.11432939520381069,
    "clean_mse_reduction_pct": 98.69287891224022,
    "clean_snr_improvement_db": 18.83684206008911,
    "inference_seconds": 0.548405,
    "device": "cpu"
  }
}
```
