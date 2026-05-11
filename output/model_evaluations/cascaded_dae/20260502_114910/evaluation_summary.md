# Evaluation Run: Cascaded DAE

## Identity

- model id: `cascaded_dae`
- run id: `20260502_114910`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `niazy.corrected_center_epochs` | 833 |
| `niazy.epoch_length_max` | 303 |
| `niazy.epoch_length_median` | 292.000000 |
| `niazy.epoch_length_min` | 292 |
| `niazy.n_channels` | 31 |
| `niazy.n_samples` | 333824 |
| `niazy.n_triggers` | 840 |
| `niazy.predicted_artifact_rms` | 9.454017e-04 |
| `niazy.sfreq_hz` | 2048.000000 |
| `niazy.template_peak_to_peak_median_after` | 0.015149 |
| `niazy.template_peak_to_peak_median_before` | 0.015243 |
| `niazy.template_peak_to_peak_reduction_pct` | 0.617196 |
| `niazy.template_rms_after` | 0.008334 |
| `niazy.template_rms_before` | 0.008348 |
| `niazy.template_rms_reduction_pct` | 0.162515 |
| `niazy.trigger_locked_rms_after` | 0.008491 |
| `niazy.trigger_locked_rms_before` | 0.008442 |
| `niazy.trigger_locked_rms_reduction_pct` | -0.586210 |
| `synthetic.artifact_corr` | 0.033572 |
| `synthetic.artifact_mae` | 0.001396 |
| `synthetic.artifact_mse` | 4.237575e-06 |
| `synthetic.artifact_snr_db` | -1.450771 |
| `synthetic.clean_mae_after` | 0.001396 |
| `synthetic.clean_mae_before` | 7.775867e-04 |
| `synthetic.clean_mse_after` | 4.237575e-06 |
| `synthetic.clean_mse_before` | 3.034173e-06 |
| `synthetic.clean_mse_reduction_pct` | -39.661615 |
| `synthetic.clean_snr_db_after` | -41.844196 |
| `synthetic.clean_snr_db_before` | -40.393425 |
| `synthetic.clean_snr_improvement_db` | -1.450771 |
| `synthetic.input_mean_removed` | True |
| `synthetic.n_examples` | 1860 |
| `synthetic.predicted_artifact_center_abs_mean_uv` | 908.278442 |
| `synthetic.predicted_artifact_edge_abs_mean_uv` | 852.513245 |
| `synthetic.predicted_artifact_edge_to_center_abs_ratio` | 0.938603 |
| `synthetic.prediction_mean_removed` | True |
| `synthetic.residual_error_rms_ratio` | 1.181785 |
| `synthetic.sfreq_hz` | 4096.000000 |

## Interpretation

Synthetic metrics are supervised against clean/artifact targets. Niazy metrics are unsupervised trigger-locked proxies because no true clean EEG is available.

## Artifacts

- `legacy_metrics`: `output/model_evaluations/cascaded_dae/20260502_114910/context_artifact_model_metrics.json`
- `legacy_report`: `output/model_evaluations/cascaded_dae/20260502_114910/context_artifact_model_evaluation_report.md`
- `niazy_cleaning_examples`: `output/model_evaluations/cascaded_dae/20260502_114910/niazy_cleaning_examples.png`
- `niazy_trigger_locked_templates`: `output/model_evaluations/cascaded_dae/20260502_114910/niazy_trigger_locked_templates.png`
- `synthetic_cleaning_examples`: `output/model_evaluations/cascaded_dae/20260502_114910/synthetic_cleaning_examples.png`
- `synthetic_metric_summary`: `output/model_evaluations/cascaded_dae/20260502_114910/synthetic_metric_summary.png`

## Limitations

- Synthetic metrics use generated clean/artifact targets and may not represent real scanner distributions.
- Niazy metrics are trigger-locked proxy metrics without clean EEG ground truth.

## Configuration

```json
{
  "checkpoint": "training_output/cascadeddenoisingautoencoder_20260502_114910/exports/cascaded_dae.ts",
  "synthetic_dataset": "output/synthetic_spike_artifact_context_512/synthetic_spike_artifact_context_dataset.npz",
  "niazy_input": "examples/datasets/NiazyFMRI.edf",
  "context_epochs": 7,
  "epoch_samples": 512,
  "device": "cpu",
  "input_mean_removed": true,
  "prediction_mean_removed": true,
  "trigger_regex": "\\b1\\b"
}
```

## Raw Metrics

```json
{
  "synthetic": {
    "n_examples": 1860,
    "sfreq_hz": 4096.0,
    "clean_mse_before": 3.034173005289631e-06,
    "clean_mse_after": 4.237575012666639e-06,
    "clean_mae_before": 0.0007775867125019431,
    "clean_mae_after": 0.001395822619087994,
    "clean_snr_db_before": -40.39342498779297,
    "clean_snr_db_after": -41.84419631958008,
    "artifact_mse": 4.237575012666639e-06,
    "artifact_mae": 0.0013958225026726723,
    "artifact_corr": 0.03357186922719476,
    "artifact_snr_db": -1.450770616531372,
    "residual_error_rms_ratio": 1.1817852149965962,
    "input_mean_removed": true,
    "prediction_mean_removed": true,
    "predicted_artifact_edge_abs_mean_uv": 852.5132446289062,
    "predicted_artifact_center_abs_mean_uv": 908.2784423828125,
    "predicted_artifact_edge_to_center_abs_ratio": 0.938603401184082,
    "clean_mse_reduction_pct": -39.661614722662186,
    "clean_snr_improvement_db": -1.4507713317871094
  },
  "niazy": {
    "n_channels": 31,
    "n_samples": 333824,
    "sfreq_hz": 2048.0,
    "n_triggers": 840,
    "corrected_center_epochs": 833,
    "epoch_length_min": 292,
    "epoch_length_median": 292.0,
    "epoch_length_max": 303,
    "trigger_locked_rms_before": 0.008441969752311707,
    "trigger_locked_rms_after": 0.00849145743995905,
    "template_rms_before": 0.008347880095243454,
    "template_rms_after": 0.00833431351929903,
    "template_peak_to_peak_median_before": 0.01524333842098713,
    "template_peak_to_peak_median_after": 0.015149257145822048,
    "predicted_artifact_rms": 0.0009454017272219062,
    "trigger_locked_rms_reduction_pct": -0.5862101985593116,
    "template_rms_reduction_pct": 0.1625152229025595,
    "template_peak_to_peak_reduction_pct": 0.6171960010777466
  }
}
```
