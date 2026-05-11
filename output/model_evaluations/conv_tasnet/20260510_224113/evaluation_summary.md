# Evaluation Run: Conv-TasNet

## Identity

- model id: `conv_tasnet`
- run id: `20260510_224113`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `baseline_reference.cascaded_context_dae__synthetic_spike_20260502_115926.artifact_corr` | 0.731800 |
| `baseline_reference.cascaded_context_dae__synthetic_spike_20260502_115926.clean_snr_improvement_db` | 3.158500 |
| `baseline_reference.cascaded_context_dae__synthetic_spike_20260502_115926.residual_error_rms_ratio` | 0.695100 |
| `baseline_reference.cascaded_dae__synthetic_spike_20260502_115914.artifact_corr` | 0.086200 |
| `baseline_reference.cascaded_dae__synthetic_spike_20260502_115914.clean_snr_improvement_db` | -0.046300 |
| `baseline_reference.cascaded_dae__synthetic_spike_20260502_115914.residual_error_rms_ratio` | 1.005300 |
| `dataset.n_channels_per_example` | 30 |
| `dataset.n_pairs_evaluated` | 4998 |
| `dataset.samples_per_epoch` | 512 |
| `dataset.sfreq_hz` | 4096.000000 |
| `synthetic_niazy_proof_fit_val_split.artifact_corr` | 0.996864 |
| `synthetic_niazy_proof_fit_val_split.artifact_mae` | 7.355550e-05 |
| `synthetic_niazy_proof_fit_val_split.artifact_mse` | 2.166425e-08 |
| `synthetic_niazy_proof_fit_val_split.artifact_snr_db` | 22.031729 |
| `synthetic_niazy_proof_fit_val_split.clean_mae_after` | 7.355550e-05 |
| `synthetic_niazy_proof_fit_val_split.clean_mae_before` | 9.077009e-04 |
| `synthetic_niazy_proof_fit_val_split.clean_mse_after` | 2.166425e-08 |
| `synthetic_niazy_proof_fit_val_split.clean_mse_before` | 3.458729e-06 |
| `synthetic_niazy_proof_fit_val_split.clean_mse_reduction_pct` | 99.373635 |
| `synthetic_niazy_proof_fit_val_split.clean_snr_db_after` | 10.422672 |
| `synthetic_niazy_proof_fit_val_split.clean_snr_db_before` | -11.609056 |
| `synthetic_niazy_proof_fit_val_split.clean_snr_improvement_db` | 22.031729 |
| `synthetic_niazy_proof_fit_val_split.input_mean_removed` | True |
| `synthetic_niazy_proof_fit_val_split.n_examples` | 4998 |
| `synthetic_niazy_proof_fit_val_split.predicted_clean_corr` | 0.955374 |
| `synthetic_niazy_proof_fit_val_split.predicted_clean_mse` | 2.083746e-08 |
| `synthetic_niazy_proof_fit_val_split.predicted_source_sum_mse` | 5.519127e-09 |
| `synthetic_niazy_proof_fit_val_split.prediction_mean_removed` | True |
| `synthetic_niazy_proof_fit_val_split.residual_error_rms_ratio` | 0.079143 |
| `synthetic_niazy_proof_fit_val_split.samples_per_epoch` | 512 |

## Interpretation

Conv-TasNet was trained from scratch on the Niazy proof-fit context dataset (Niazy/AAS 2x direct artifact library, 512-sample center epochs). Metrics reported here are on the held-out 20 % validation split using the same RNG seed as training, so each test pair is one (example, channel) mixture with known clean and artifact targets. Supervised clean-MSE reduction and artifact correlation are the primary quality signals; predicted_source_sum_mse is reported as a consistency check (Conv-TasNet does not enforce clean + artifact = noisy during training). Comparison to cascaded_dae and cascaded_context_dae is recorded for orientation only — those runs were evaluated on a different synthetic-spike dataset, so the absolute numbers are not directly comparable.

## Artifacts

- `examples_plot`: `output/model_evaluations/conv_tasnet/20260510_224113/plots/conv_tasnet_examples.png`
- `summary_plot`: `output/model_evaluations/conv_tasnet/20260510_224113/plots/conv_tasnet_metric_summary.png`

## Limitations

- Validation is on the same Niazy/AAS-derived artifact library used for training; it does not test cross-subject or cross-scanner generalisation.
- Real Niazy EDF trigger-locked metrics are not included in this run; they require the EDF input and a longer pipeline. Add later when a comparable run is available across all baseline models.
- Source consistency (clean + artifact = noisy) is reported but not enforced during training. The DeepLearningCorrection pipeline only consumes the artifact source.

## Configuration

```json
{
  "model_id": "conv_tasnet",
  "model_name": "Conv-TasNet",
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/convtasnetniazyprooffit_20260510_202818/exports/conv_tasnet.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "device": "cpu",
  "batch_size": 256,
  "val_ratio": 0.2,
  "seed": 42,
  "keep_input_mean": false,
  "keep_prediction_mean": false
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
    "clean_mse_after": 2.1664249061359442e-08,
    "clean_mae_before": 0.0009077009162865579,
    "clean_mae_after": 7.35555004212074e-05,
    "clean_snr_db_before": -11.60905647277832,
    "clean_snr_db_after": 10.422672271728516,
    "artifact_mse": 2.1664249061359442e-08,
    "artifact_mae": 7.35555004212074e-05,
    "artifact_corr": 0.9968637383731933,
    "artifact_snr_db": 22.031728744506836,
    "predicted_clean_mse": 2.083745975767215e-08,
    "predicted_clean_corr": 0.9553737684031675,
    "predicted_source_sum_mse": 5.519127377340283e-09,
    "residual_error_rms_ratio": 0.07914319705224314,
    "clean_mse_reduction_pct": 99.37363546187531,
    "clean_snr_improvement_db": 22.031728744506836
  },
  "dataset": {
    "n_pairs_evaluated": 4998,
    "n_channels_per_example": 30,
    "samples_per_epoch": 512,
    "sfreq_hz": 4096.0
  },
  "baseline_reference": {
    "cascaded_dae__synthetic_spike_20260502_115914": {
      "clean_snr_improvement_db": -0.0463,
      "artifact_corr": 0.0862,
      "residual_error_rms_ratio": 1.0053
    },
    "cascaded_context_dae__synthetic_spike_20260502_115926": {
      "clean_snr_improvement_db": 3.1585,
      "artifact_corr": 0.7318,
      "residual_error_rms_ratio": 0.6951
    }
  }
}
```
