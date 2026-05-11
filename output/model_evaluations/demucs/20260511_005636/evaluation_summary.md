# Evaluation Run: Demucs

## Identity

- model id: `demucs`
- run id: `20260511_005636`
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
| `baseline_reference.conv_tasnet__niazy_proof_fit_20260510_224113.artifact_corr` | 0.997000 |
| `baseline_reference.conv_tasnet__niazy_proof_fit_20260510_224113.clean_snr_improvement_db` | 22.030000 |
| `baseline_reference.conv_tasnet__niazy_proof_fit_20260510_224113.residual_error_rms_ratio` | 0.079000 |
| `dataset.context_epochs` | 7 |
| `dataset.n_channels_per_example` | 30 |
| `dataset.n_pairs_evaluated` | 4998 |
| `dataset.samples_per_epoch` | 512 |
| `dataset.sfreq_hz` | 4096.000000 |
| `synthetic_niazy_proof_fit_val_split.artifact_corr` | 0.999633 |
| `synthetic_niazy_proof_fit_val_split.artifact_mae` | 2.615030e-05 |
| `synthetic_niazy_proof_fit_val_split.artifact_mse` | 2.576238e-09 |
| `synthetic_niazy_proof_fit_val_split.artifact_snr_db` | 31.279306 |
| `synthetic_niazy_proof_fit_val_split.clean_mae_after` | 2.615030e-05 |
| `synthetic_niazy_proof_fit_val_split.clean_mae_before` | 9.077009e-04 |
| `synthetic_niazy_proof_fit_val_split.clean_mse_after` | 2.576238e-09 |
| `synthetic_niazy_proof_fit_val_split.clean_mse_before` | 3.458729e-06 |
| `synthetic_niazy_proof_fit_val_split.clean_mse_reduction_pct` | 99.925515 |
| `synthetic_niazy_proof_fit_val_split.clean_snr_db_after` | 19.670250 |
| `synthetic_niazy_proof_fit_val_split.clean_snr_db_before` | -11.609056 |
| `synthetic_niazy_proof_fit_val_split.clean_snr_improvement_db` | 31.279306 |
| `synthetic_niazy_proof_fit_val_split.input_mean_removed` | True |
| `synthetic_niazy_proof_fit_val_split.n_examples` | 4998 |
| `synthetic_niazy_proof_fit_val_split.prediction_mean_removed` | True |
| `synthetic_niazy_proof_fit_val_split.residual_error_rms_ratio` | 0.027292 |
| `synthetic_niazy_proof_fit_val_split.samples_per_epoch` | 512 |

## Interpretation

Demucs was trained from scratch on the Niazy proof-fit context dataset (Niazy/AAS 2x direct artifact library, 7-epoch context of 512-sample epochs). Inference consumes a single 3584-sample mixture per channel and returns a 3584-sample artifact prediction; the metrics here use the center 512-sample slice of that prediction, matching how DemucsCorrection subtracts the artifact at inference. The validation split is the same 20% held-out subset used during training (same RNG seed). Comparison to cascaded_dae, cascaded_context_dae and conv_tasnet is recorded for orientation only — the cascaded models were evaluated on a synthetic-spike dataset, while conv_tasnet was evaluated on the same Niazy proof-fit set used here, so the Demucs vs Conv-TasNet comparison is the most direct.

## Artifacts

- `examples_plot`: `output/model_evaluations/demucs/20260511_005636/plots/demucs_examples.png`
- `summary_plot`: `output/model_evaluations/demucs/20260511_005636/plots/demucs_metric_summary.png`

## Limitations

- Validation is in-distribution: the val split shares the artifact library used for training; no cross-subject or cross-scanner generalisation tested.
- Real Niazy EDF trigger-locked metrics are not run here; they require the EDF input and a longer pipeline. Add later in a unified cross-model evaluator.
- Only the center epoch of the 7-epoch prediction is scored against ground truth, consistent with how the pipeline adapter applies the correction.

## Configuration

```json
{
  "model_id": "demucs",
  "model_name": "Demucs",
  "checkpoint": "training_output/demucsniazyprooffit_20260510_224653/exports/demucs_cpu.ts",
  "dataset": "output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "device": "cpu",
  "batch_size": 64,
  "val_ratio": 0.2,
  "seed": 42,
  "context_epochs": 7,
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
    "clean_mse_after": 2.5762376676397025e-09,
    "clean_mae_before": 0.0009077009162865579,
    "clean_mae_after": 2.615029734442942e-05,
    "clean_snr_db_before": -11.60905647277832,
    "clean_snr_db_after": 19.670249938964844,
    "artifact_mse": 2.5762376676397025e-09,
    "artifact_mae": 2.615029734442942e-05,
    "artifact_corr": 0.9996334725048432,
    "artifact_snr_db": 31.279306411743164,
    "residual_error_rms_ratio": 0.027291959489125252,
    "clean_mse_reduction_pct": 99.92551489265932,
    "clean_snr_improvement_db": 31.279306411743164
  },
  "dataset": {
    "n_pairs_evaluated": 4998,
    "n_channels_per_example": 30,
    "samples_per_epoch": 512,
    "context_epochs": 7,
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
    },
    "conv_tasnet__niazy_proof_fit_20260510_224113": {
      "clean_snr_improvement_db": 22.03,
      "artifact_corr": 0.997,
      "residual_error_rms_ratio": 0.079
    }
  }
}
```
