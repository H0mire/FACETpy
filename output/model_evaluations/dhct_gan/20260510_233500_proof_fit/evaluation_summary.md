# Evaluation Run: DHCT-GAN

## Identity

- model id: `dhct_gan`
- run id: `20260510_233500_proof_fit`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `synthetic.artifact_corr` | 0.157722 |
| `synthetic.artifact_mae` | 0.001270 |
| `synthetic.artifact_mse` | 1.827220e-05 |
| `synthetic.artifact_snr_db` | -7.128783 |
| `synthetic.clean_mae_after` | 0.001270 |
| `synthetic.clean_mae_before` | 9.189959e-04 |
| `synthetic.clean_mse_after` | 1.827220e-05 |
| `synthetic.clean_mse_before` | 3.539260e-06 |
| `synthetic.clean_mse_reduction_pct` | -416.271690 |
| `synthetic.clean_snr_db_after` | -18.721912 |
| `synthetic.clean_snr_db_before` | -11.593130 |
| `synthetic.clean_snr_improvement_db` | -7.128782 |
| `synthetic.n_windows` | 24990 |
| `synthetic.residual_error_rms_ratio` | 2.272161 |
| `synthetic.sfreq_hz` | 4096.000000 |

## Interpretation

DHCT-GAN supervised proof-fit metrics on the Niazy bundle. Clean SNR improvement and artifact correlation should both move in the right direction relative to the no-correction baseline.

## Artifacts

- `supervised_examples`: `plots/supervised_examples.png`

## Limitations

- Targets are AAS-derived. The clean reference is the AAS-corrected Niazy surrogate, so generalization claims beyond the AAS estimate are not warranted.
- Per-window demeaning is applied before inference; absolute baseline drift is not corrected.
- Real-data trigger-locked proxies and the FACET framework metric battery are not yet included.

## Configuration

```json
{
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/dhctganniazyprooffit_20260510_213159/exports/dhct_gan.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "device": "cpu",
  "batch_size": 64,
  "demean_input": true,
  "remove_prediction_mean": true
}
```

## Raw Metrics

```json
{
  "synthetic": {
    "n_windows": 24990,
    "sfreq_hz": 4096.0,
    "clean_mse_before": 3.5392604331718758e-06,
    "clean_mse_after": 1.8272199667990208e-05,
    "clean_mae_before": 0.0009189959382638335,
    "clean_mae_after": 0.0012697484344244003,
    "clean_snr_db_before": -11.593130111694336,
    "clean_snr_db_after": -18.721912384033203,
    "clean_snr_improvement_db": -7.128782272338867,
    "clean_mse_reduction_pct": -416.27169045637754,
    "artifact_mse": 1.8272199667990208e-05,
    "artifact_mae": 0.0012697484344244003,
    "artifact_corr": 0.15772187648404576,
    "artifact_snr_db": -7.128782749176025,
    "residual_error_rms_ratio": 2.2721612516991616
  }
}
```
