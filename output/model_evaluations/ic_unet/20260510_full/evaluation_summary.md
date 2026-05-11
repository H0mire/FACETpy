# Evaluation Run: IC-U-Net

## Identity

- model id: `ic_unet`
- run id: `20260510_full`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `input_mean_removed` | True |
| `n_examples` | 833 |
| `prediction_mean_removed` | True |
| `real_proxy.predicted_artifact_rms_uv` | 1880.316180 |
| `real_proxy.trigger_locked_rms_after_uv` | 57.278296 |
| `real_proxy.trigger_locked_rms_before_uv` | 1910.125255 |
| `real_proxy.trigger_locked_rms_reduction_pct` | 97.001333 |
| `sfreq_hz` | 4096.000000 |
| `synthetic.artifact_corr` | 0.966703 |
| `synthetic.artifact_mae` | 2.109103e-04 |
| `synthetic.artifact_mse` | 2.355785e-07 |
| `synthetic.artifact_snr_db` | 11.767769 |
| `synthetic.clean_mae_after` | 2.109103e-04 |
| `synthetic.clean_mae_before` | 9.189959e-04 |
| `synthetic.clean_mse_after` | 2.355785e-07 |
| `synthetic.clean_mse_before` | 3.539260e-06 |
| `synthetic.clean_mse_reduction_pct` | 93.343850 |
| `synthetic.clean_snr_db_after` | 0.174640 |
| `synthetic.clean_snr_db_before` | -11.593130 |
| `synthetic.clean_snr_improvement_db` | 11.767770 |
| `synthetic.residual_error_rms_ratio` | 0.257995 |

## Interpretation

IC-U-Net trained on the Niazy proof-fit context dataset. The clean target is the AAS-corrected Niazy signal; the artifact target is the AAS-estimated artifact from the same recording. SNR-after-vs-before and the artifact correlation indicate how closely the U-Net captures the AAS estimate.

## Artifacts

- `synthetic_examples`: `plots/synthetic_cleaning_examples.png`

## Limitations

- The proof-fit dataset uses the same Niazy recording for training and evaluation; numbers reported here characterise fit quality, not generalisation.
- The clean target is itself an AAS estimate, so absolute clean-SNR values inherit AAS bias.

## Configuration

```json
{
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/icunetniazyprooffit_20260510_223556/exports/ic_unet.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "batch_size": 32,
  "device": "cpu",
  "input_mean_removed": true,
  "prediction_mean_removed": true
}
```

## Raw Metrics

```json
{
  "n_examples": 833,
  "sfreq_hz": 4096.0,
  "input_mean_removed": true,
  "prediction_mean_removed": true,
  "synthetic": {
    "clean_mse_before": 3.5392604331718758e-06,
    "clean_mse_after": 2.3557849715416523e-07,
    "clean_mae_before": 0.0009189959382638335,
    "clean_mae_after": 0.00021091032249387354,
    "clean_snr_db_before": -11.593130111694336,
    "clean_snr_db_after": 0.17463961243629456,
    "artifact_mse": 2.3557849715416523e-07,
    "artifact_mae": 0.00021091032249387354,
    "artifact_corr": 0.9667025249266686,
    "artifact_snr_db": 11.767768859863281,
    "residual_error_rms_ratio": 0.2579951630480285,
    "clean_snr_improvement_db": 11.76776972413063,
    "clean_mse_reduction_pct": 93.34384960919535
  },
  "real_proxy": {
    "trigger_locked_rms_before_uv": 1910.1252546533942,
    "trigger_locked_rms_after_uv": 57.278295571450144,
    "predicted_artifact_rms_uv": 1880.316180177033,
    "trigger_locked_rms_reduction_pct": 97.00133300515711
  }
}
```
