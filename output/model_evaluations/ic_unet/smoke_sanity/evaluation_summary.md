# Evaluation Run: IC-U-Net

## Identity

- model id: `ic_unet`
- run id: `smoke_sanity`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `input_mean_removed` | True |
| `n_examples` | 833 |
| `prediction_mean_removed` | True |
| `real_proxy.predicted_artifact_rms_uv` | 1904.442208 |
| `real_proxy.trigger_locked_rms_after_uv` | 30.401607 |
| `real_proxy.trigger_locked_rms_before_uv` | 1910.125255 |
| `real_proxy.trigger_locked_rms_reduction_pct` | 98.408397 |
| `sfreq_hz` | 4096.000000 |
| `synthetic.artifact_corr` | 0.966022 |
| `synthetic.artifact_mae` | 2.167889e-04 |
| `synthetic.artifact_mse` | 2.440125e-07 |
| `synthetic.artifact_snr_db` | 11.615005 |
| `synthetic.clean_mae_after` | 2.167889e-04 |
| `synthetic.clean_mae_before` | 9.189959e-04 |
| `synthetic.clean_mse_after` | 2.440125e-07 |
| `synthetic.clean_mse_before` | 3.539260e-06 |
| `synthetic.clean_mse_reduction_pct` | 93.105552 |
| `synthetic.clean_snr_db_after` | 0.021875 |
| `synthetic.clean_snr_db_before` | -11.593130 |
| `synthetic.clean_snr_improvement_db` | 11.615006 |
| `synthetic.residual_error_rms_ratio` | 0.262573 |

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
  "checkpoint": "/tmp/ic_unet_smoke.ts",
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
    "clean_mse_after": 2.4401248310823576e-07,
    "clean_mae_before": 0.0009189959382638335,
    "clean_mae_after": 0.00021678890334442258,
    "clean_snr_db_before": -11.593130111694336,
    "clean_snr_db_after": 0.02187540754675865,
    "artifact_mse": 2.4401248310823576e-07,
    "artifact_mae": 0.00021678890334442258,
    "artifact_corr": 0.9660219640109989,
    "artifact_snr_db": 11.615004539489746,
    "residual_error_rms_ratio": 0.2625728047409382,
    "clean_snr_improvement_db": 11.615005519241095,
    "clean_mse_reduction_pct": 93.10555163386066
  },
  "real_proxy": {
    "trigger_locked_rms_before_uv": 1910.1252546533942,
    "trigger_locked_rms_after_uv": 30.40160663658753,
    "predicted_artifact_rms_uv": 1904.4422078877687,
    "trigger_locked_rms_reduction_pct": 98.40839722094015
  }
}
```
