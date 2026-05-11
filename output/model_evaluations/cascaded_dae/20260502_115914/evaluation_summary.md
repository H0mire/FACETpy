# Evaluation Run: Cascaded DAE

## Identity

- model id: `cascaded_dae`
- run id: `20260502_115914`
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
| `niazy.predicted_artifact_rms` | 4.620621e-04 |
| `niazy.sfreq_hz` | 2048.000000 |
| `niazy.template_peak_to_peak_median_after` | 0.015248 |
| `niazy.template_peak_to_peak_median_before` | 0.015243 |
| `niazy.template_peak_to_peak_reduction_pct` | -0.027530 |
| `niazy.template_rms_after` | 0.008335 |
| `niazy.template_rms_before` | 0.008348 |
| `niazy.template_rms_reduction_pct` | 0.158566 |
| `niazy.trigger_locked_rms_after` | 0.008443 |
| `niazy.trigger_locked_rms_before` | 0.008442 |
| `niazy.trigger_locked_rms_reduction_pct` | -0.015875 |
| `synthetic.artifact_corr` | 0.086248 |
| `synthetic.artifact_mae` | 9.221299e-04 |
| `synthetic.artifact_mse` | 3.066671e-06 |
| `synthetic.artifact_snr_db` | -0.046269 |
| `synthetic.clean_mae_after` | 9.221299e-04 |
| `synthetic.clean_mae_before` | 7.775867e-04 |
| `synthetic.clean_mse_after` | 3.066671e-06 |
| `synthetic.clean_mse_before` | 3.034173e-06 |
| `synthetic.clean_mse_reduction_pct` | -1.071068 |
| `synthetic.clean_snr_db_after` | -40.439690 |
| `synthetic.clean_snr_db_before` | -40.393425 |
| `synthetic.clean_snr_improvement_db` | -0.046265 |
| `synthetic.input_mean_removed` | True |
| `synthetic.n_examples` | 1860 |
| `synthetic.predicted_artifact_center_abs_mean_uv` | 300.055817 |
| `synthetic.predicted_artifact_edge_abs_mean_uv` | 278.738678 |
| `synthetic.predicted_artifact_edge_to_center_abs_ratio` | 0.928956 |
| `synthetic.prediction_mean_removed` | True |
| `synthetic.residual_error_rms_ratio` | 1.005341 |
| `synthetic.sfreq_hz` | 4096.000000 |

## Interpretation

Synthetic metrics are supervised against clean/artifact targets. Niazy metrics are unsupervised trigger-locked proxies because no true clean EEG is available.

## Artifacts

- `legacy_metrics`: `output/model_evaluations/cascaded_dae/20260502_115914/context_artifact_model_metrics.json`
- `legacy_report`: `output/model_evaluations/cascaded_dae/20260502_115914/context_artifact_model_evaluation_report.md`
- `niazy_cleaning_examples`: `output/model_evaluations/cascaded_dae/20260502_115914/niazy_cleaning_examples.png`
- `niazy_trigger_locked_templates`: `output/model_evaluations/cascaded_dae/20260502_115914/niazy_trigger_locked_templates.png`
- `synthetic_cleaning_examples`: `output/model_evaluations/cascaded_dae/20260502_115914/synthetic_cleaning_examples.png`
- `synthetic_metric_summary`: `output/model_evaluations/cascaded_dae/20260502_115914/synthetic_metric_summary.png`

## Limitations

- Synthetic metrics use generated clean/artifact targets and may not represent real scanner distributions.
- Niazy metrics are trigger-locked proxy metrics without clean EEG ground truth.

## Configuration

```json
{
  "checkpoint": "training_output/cascadeddenoisingautoencoder_20260502_115914/exports/cascaded_dae.ts",
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
    "clean_mse_after": 3.0666710699733812e-06,
    "clean_mae_before": 0.0007775867125019431,
    "clean_mae_after": 0.0009221298969350755,
    "clean_snr_db_before": -40.39342498779297,
    "clean_snr_db_after": -40.43968963623047,
    "artifact_mse": 3.0666710699733812e-06,
    "artifact_mae": 0.0009221298969350755,
    "artifact_corr": 0.08624756369937937,
    "artifact_snr_db": -0.0462685152888298,
    "residual_error_rms_ratio": 1.005341080775384,
    "input_mean_removed": true,
    "prediction_mean_removed": true,
    "predicted_artifact_edge_abs_mean_uv": 278.7386779785156,
    "predicted_artifact_center_abs_mean_uv": 300.0558166503906,
    "predicted_artifact_edge_to_center_abs_ratio": 0.9289560914039612,
    "clean_mse_reduction_pct": -1.071068282101395,
    "clean_snr_improvement_db": -0.0462646484375
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
    "trigger_locked_rms_after": 0.008443309925496578,
    "template_rms_before": 0.008347880095243454,
    "template_rms_after": 0.008334643207490444,
    "template_peak_to_peak_median_before": 0.01524333842098713,
    "template_peak_to_peak_median_after": 0.015247534960508347,
    "predicted_artifact_rms": 0.00046206210390664637,
    "trigger_locked_rms_reduction_pct": -0.015875124220920966,
    "template_rms_reduction_pct": 0.15856585866095285,
    "template_peak_to_peak_reduction_pct": -0.027530317869484122
  }
}
```
