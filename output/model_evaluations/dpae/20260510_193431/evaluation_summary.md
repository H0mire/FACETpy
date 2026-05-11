# Evaluation Run: DPAE

## Identity

- model id: `dpae`
- run id: `20260510_193431`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `niazy_proof_fit.artifact_corr` | 0.913238 |
| `niazy_proof_fit.artifact_mae` | 6.671008e-04 |
| `niazy_proof_fit.artifact_mse` | 6.329423e-07 |
| `niazy_proof_fit.artifact_snr_db` | 7.475478 |
| `niazy_proof_fit.clean_mae_after` | 6.671008e-04 |
| `niazy_proof_fit.clean_mae_before` | 9.189954e-04 |
| `niazy_proof_fit.clean_mse_after` | 6.329423e-07 |
| `niazy_proof_fit.clean_mse_before` | 3.539260e-06 |
| `niazy_proof_fit.clean_mse_reduction_pct` | 82.116536 |
| `niazy_proof_fit.clean_snr_db_after` | -4.117634 |
| `niazy_proof_fit.clean_snr_db_before` | -11.593112 |
| `niazy_proof_fit.clean_snr_improvement_db` | 7.475478 |
| `niazy_proof_fit.inference_seconds` | 100.391249 |
| `niazy_proof_fit.n_examples` | 24990 |
| `niazy_proof_fit.n_samples` | 512 |
| `niazy_proof_fit.predicted_artifact_center_abs_mean` | 0.001342 |
| `niazy_proof_fit.predicted_artifact_edge_abs_mean` | 0.001001 |
| `niazy_proof_fit.predicted_artifact_edge_to_center_abs_ratio` | 0.745667 |
| `niazy_proof_fit.residual_error_rms_ratio` | 0.422888 |
| `niazy_proof_fit.sfreq_hz` | 4096.000000 |

## Interpretation

DPAE evaluated on the Niazy proof-fit context dataset (24990 channel-wise examples of 512 samples). Clean MSE drops by 82.12% and clean-signal SNR improves by 7.48 dB. The clean reference is the AAS-corrected Niazy surrogate, so this run measures proof-fit of artifact morphology, not generalisation to an independent recording.

## Artifacts

- `niazy_proof_fit_examples`: `/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/model_evaluations/dpae/20260510_193431/plots/niazy_proof_fit_examples.png`

## Limitations

- Clean ground truth is the AAS-corrected Niazy surrogate, not an independent clean source.
- Single-recording proof-fit; no cross-subject generalisation tested.

## Configuration

```json
{
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/dualpathwayautoencoderniazyprooffit_20260510_192929/exports/dpae.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "device": "cpu",
  "batch_size": 256,
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
    "clean_mse_after": 6.329422834387846e-07,
    "clean_mae_before": 0.000918995427675437,
    "clean_mae_after": 0.0006671008315511795,
    "clean_snr_db_before": -11.593111589814162,
    "clean_snr_db_after": -4.117633731846274,
    "artifact_mse": 6.329422834163196e-07,
    "artifact_mae": 0.0006671008315538655,
    "artifact_corr": 0.9132380912255171,
    "artifact_snr_db": 7.475477857865061,
    "residual_error_rms_ratio": 0.4228884518592859,
    "predicted_artifact_edge_abs_mean": 0.0010006424967346197,
    "predicted_artifact_center_abs_mean": 0.0013419433335763583,
    "predicted_artifact_edge_to_center_abs_ratio": 0.7456667295092471,
    "clean_mse_reduction_pct": 82.11653575435098,
    "clean_snr_improvement_db": 7.475477857967888,
    "sfreq_hz": 4096.0,
    "inference_seconds": 100.39124894142151,
    "device": "cpu"
  }
}
```
