# Evaluation Run: SepFormer

## Identity

- model id: `sepformer`
- run id: `20260511_012740`
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
| `baseline_reference.demucs__niazy_proof_fit_20260511_005636.artifact_corr` | 0.999600 |
| `baseline_reference.demucs__niazy_proof_fit_20260511_005636.clean_snr_improvement_db` | 31.279300 |
| `baseline_reference.demucs__niazy_proof_fit_20260511_005636.residual_error_rms_ratio` | 0.027300 |
| `dataset.context_epochs` | 7 |
| `dataset.n_channels_per_example` | 30 |
| `dataset.n_pairs_evaluated` | 4998 |
| `dataset.samples_per_epoch` | 512 |
| `dataset.sfreq_hz` | 4096.000000 |
| `synthetic_niazy_proof_fit_val_split.artifact_corr` | 0.993805 |
| `synthetic_niazy_proof_fit_val_split.artifact_mae` | 9.946461e-05 |
| `synthetic_niazy_proof_fit_val_split.artifact_mse` | 4.303860e-08 |
| `synthetic_niazy_proof_fit_val_split.artifact_snr_db` | 19.050570 |
| `synthetic_niazy_proof_fit_val_split.clean_mae_after` | 9.946462e-05 |
| `synthetic_niazy_proof_fit_val_split.clean_mae_before` | 9.080743e-04 |
| `synthetic_niazy_proof_fit_val_split.clean_mse_after` | 4.303860e-08 |
| `synthetic_niazy_proof_fit_val_split.clean_mse_before` | 3.458718e-06 |
| `synthetic_niazy_proof_fit_val_split.clean_mse_reduction_pct` | 98.755649 |
| `synthetic_niazy_proof_fit_val_split.clean_snr_db_after` | 7.438765 |
| `synthetic_niazy_proof_fit_val_split.clean_snr_db_before` | -11.611805 |
| `synthetic_niazy_proof_fit_val_split.clean_snr_improvement_db` | 19.050570 |
| `synthetic_niazy_proof_fit_val_split.input_mean_removed` | True |
| `synthetic_niazy_proof_fit_val_split.n_examples` | 4998 |
| `synthetic_niazy_proof_fit_val_split.prediction_mean_removed` | True |
| `synthetic_niazy_proof_fit_val_split.residual_error_rms_ratio` | 0.111551 |
| `synthetic_niazy_proof_fit_val_split.samples_per_epoch` | 512 |

## Interpretation

SepFormer was trained channel-wise on the Niazy proof-fit context dataset. The reported metrics are computed on the deterministic val-split slice that facet-train held out at training time (val_ratio=0.2, seed=42), so they answer the proof-of-fit question 'can SepFormer learn the AAS-estimated artifact morphology' rather than a generalization claim.

## Artifacts

- `examples_plot`: `output/model_evaluations/sepformer/20260511_012740/plots/sepformer_examples.png`
- `summary_plot`: `output/model_evaluations/sepformer/20260511_012740/plots/sepformer_metric_summary.png`

## Limitations

- Clean target is an AAS surrogate of the same Niazy recording used to train the model.
- Not a generalization benchmark — no held-out subject or independent recording is used.
- Baseline reference metrics are copied from prior evaluation runs and not re-computed on this exact dataset for cascaded_dae / cascaded_context_dae.

## Configuration

```json
{
  "model_id": "sepformer",
  "model_name": "SepFormer",
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/sepformerniazyprooffit_20260510_230104/exports/sepformer.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
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
    "clean_mse_before": 3.4587178561196197e-06,
    "clean_mse_after": 4.303860023924244e-08,
    "clean_mae_before": 0.0009080742602236569,
    "clean_mae_after": 9.946461796062067e-05,
    "clean_snr_db_before": -11.611804962158203,
    "clean_snr_db_after": 7.438765048980713,
    "artifact_mse": 4.303860023924244e-08,
    "artifact_mae": 9.946461068466306e-05,
    "artifact_corr": 0.993805069447956,
    "artifact_snr_db": 19.050569534301758,
    "residual_error_rms_ratio": 0.11155050023971469,
    "clean_mse_reduction_pct": 98.7556487106026,
    "clean_snr_improvement_db": 19.050570011138916
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
    },
    "demucs__niazy_proof_fit_20260511_005636": {
      "clean_snr_improvement_db": 31.2793,
      "artifact_corr": 0.9996,
      "residual_error_rms_ratio": 0.0273
    }
  }
}
```
