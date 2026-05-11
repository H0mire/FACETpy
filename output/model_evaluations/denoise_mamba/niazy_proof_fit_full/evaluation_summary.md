# Evaluation Run: DenoiseMamba

## Identity

- model id: `denoise_mamba`
- run id: `niazy_proof_fit_full`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `niazy_proof_fit.artifact_corr` | 0.966387 |
| `niazy_proof_fit.artifact_mae` | 2.114116e-04 |
| `niazy_proof_fit.artifact_mse` | 2.339324e-07 |
| `niazy_proof_fit.artifact_snr_db` | 11.798204 |
| `niazy_proof_fit.clean_mae_after` | 2.114116e-04 |
| `niazy_proof_fit.clean_mae_before` | 9.189954e-04 |
| `niazy_proof_fit.clean_mse_after` | 2.339324e-07 |
| `niazy_proof_fit.clean_mse_before` | 3.539260e-06 |
| `niazy_proof_fit.clean_mse_reduction_pct` | 93.390359 |
| `niazy_proof_fit.clean_snr_db_after` | 0.205093 |
| `niazy_proof_fit.clean_snr_db_before` | -11.593112 |
| `niazy_proof_fit.clean_snr_improvement_db` | 11.798204 |
| `niazy_proof_fit.inference_seconds` | 328.246855 |
| `niazy_proof_fit.n_examples` | 24990 |
| `niazy_proof_fit.n_samples` | 512 |
| `niazy_proof_fit.predicted_artifact_center_abs_mean` | 9.765678e-04 |
| `niazy_proof_fit.predicted_artifact_edge_abs_mean` | 5.873708e-04 |
| `niazy_proof_fit.predicted_artifact_edge_to_center_abs_ratio` | 0.601464 |
| `niazy_proof_fit.residual_error_rms_ratio` | 0.257092 |
| `niazy_proof_fit.sfreq_hz` | 4096.000000 |

## Interpretation

DenoiseMamba evaluated on the Niazy proof-fit context dataset (24990 channel-wise examples of 512 samples). Clean MSE drops by 93.39% and clean-signal SNR improves by 11.80 dB. The clean reference is the AAS-corrected Niazy surrogate, so this run measures proof-fit of artifact morphology, not generalisation to an independent recording.

## Artifacts

- `niazy_proof_fit_examples`: `/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/worktrees/model-denoise_mamba/output/model_evaluations/denoise_mamba/niazy_proof_fit_full/plots/niazy_proof_fit_examples.png`

## Limitations

- Clean ground truth is the AAS-corrected Niazy surrogate, not an independent clean source.
- Single-recording proof-fit; no cross-subject generalisation tested.
- Selective-scan is a pure-PyTorch sequential loop; CPU inference is slower than a CUDA mamba-ssm kernel would be.

## Configuration

```json
{
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/worktrees/model-denoise_mamba/training_output/denoisemambaniazyprooffit_20260510_193847/exports/denoise_mamba_cpu.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "device": "cpu",
  "batch_size": 128,
  "demean_input": true,
  "remove_prediction_mean": true,
  "max_examples": null
}
```

## Raw Metrics

```json
{
  "niazy_proof_fit": {
    "n_examples": 24990,
    "n_samples": 512,
    "clean_mse_before": 3.539259926067049e-06,
    "clean_mse_after": 2.3393236360941561e-07,
    "clean_mae_before": 0.000918995427675437,
    "clean_mae_after": 0.00021141155341577905,
    "clean_snr_db_before": -11.593111589814162,
    "clean_snr_db_after": 0.20509257100682537,
    "artifact_mse": 2.3393236368045203e-07,
    "artifact_mae": 0.0002114115534539547,
    "artifact_corr": 0.9663873310188844,
    "artifact_snr_db": 11.798204159245234,
    "residual_error_rms_ratio": 0.2570922148504147,
    "predicted_artifact_edge_abs_mean": 0.0005873707715499315,
    "predicted_artifact_center_abs_mean": 0.0009765677856848664,
    "predicted_artifact_edge_to_center_abs_ratio": 0.6014644146161434,
    "clean_mse_reduction_pct": 93.39035932663556,
    "clean_snr_improvement_db": 11.798204160820987,
    "sfreq_hz": 4096.0,
    "inference_seconds": 328.24685525894165,
    "device": "cpu"
  }
}
```
