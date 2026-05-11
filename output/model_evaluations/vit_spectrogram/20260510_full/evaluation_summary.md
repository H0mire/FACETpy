# Evaluation Run: ViT Spectrogram Inpainter

## Identity

- model id: `vit_spectrogram`
- run id: `20260510_full`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `synthetic.artifact_corr` | 0.966014 |
| `synthetic.artifact_mae` | 2.140137e-04 |
| `synthetic.artifact_mse` | 2.450824e-07 |
| `synthetic.artifact_snr_db` | 11.596004 |
| `synthetic.clean_mae_after` | 2.140137e-04 |
| `synthetic.clean_mae_before` | 9.189959e-04 |
| `synthetic.clean_mse_after` | 2.450824e-07 |
| `synthetic.clean_mse_before` | 3.539260e-06 |
| `synthetic.clean_mse_reduction_pct` | 93.075322 |
| `synthetic.clean_snr_db_after` | 0.002875 |
| `synthetic.clean_snr_db_before` | -11.593130 |
| `synthetic.clean_snr_improvement_db` | 11.596005 |
| `synthetic.input_mean_removed` | True |
| `synthetic.n_channels` | 30 |
| `synthetic.n_examples` | 833 |
| `synthetic.residual_error_rms_ratio` | 0.263148 |
| `synthetic.sfreq_hz` | 4096.000000 |

## Interpretation

Niazy proof-fit metrics use the AAS-corrected EEG as a surrogate clean target. Clean-signal improvements are bounded by the AAS surrogate's residual; metrics should be compared head-to-head with cascaded_context_dae on the same dataset.

## Artifacts

- `metrics_json`: `/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/model_evaluations/vit_spectrogram/20260510_full/vit_spectrogram_metrics.json`
- `synthetic_cleaning_examples`: `/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/model_evaluations/vit_spectrogram/20260510_full/synthetic_cleaning_examples.png`
- `synthetic_metric_summary`: `/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/model_evaluations/vit_spectrogram/20260510_full/synthetic_metric_summary.png`

## Limitations

- Proof-fit only: training and inference draw from the same Niazy recording.
- Magnitude-only spectrogram reconstruction; phase is preserved from the noisy input and therefore retains some artifact-locked phase structure (see research_notes.md).

## Configuration

```json
{
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/vitspectrograminpainterniazyprooffit_20260510_211842/exports/vit_spectrogram_cpu.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "device": "cpu",
  "batch_size": 64,
  "input_mean_removed": true
}
```

## Raw Metrics

```json
{
  "synthetic": {
    "n_examples": 833,
    "n_channels": 30,
    "sfreq_hz": 4096.0,
    "clean_mse_before": 3.5392604331718758e-06,
    "clean_mse_after": 2.450823899380339e-07,
    "clean_mae_before": 0.0009189959382638335,
    "clean_mae_after": 0.00021401367848739028,
    "clean_snr_db_before": -11.593130111694336,
    "clean_snr_db_after": 0.0028749792836606503,
    "artifact_mse": 2.450823899380339e-07,
    "artifact_mae": 0.00021401367848739028,
    "artifact_corr": 0.9660141305573977,
    "artifact_snr_db": 11.596003532409668,
    "residual_error_rms_ratio": 0.2631478293833044,
    "input_mean_removed": true,
    "clean_mse_reduction_pct": 93.07532196158871,
    "clean_snr_improvement_db": 11.596005090977997
  }
}
```
