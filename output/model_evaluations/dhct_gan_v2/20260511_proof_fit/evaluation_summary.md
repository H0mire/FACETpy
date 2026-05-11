# Evaluation Run: DHCT-GAN v2

## Identity

- model id: `dhct_gan_v2`
- run id: `20260511_proof_fit`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `synthetic.artifact_corr` | 0.567279 |
| `synthetic.artifact_mae` | 8.879856e-04 |
| `synthetic.artifact_mse` | 2.400815e-06 |
| `synthetic.artifact_snr_db` | 1.685538 |
| `synthetic.clean_mae_after` | 8.879856e-04 |
| `synthetic.clean_mae_before` | 9.189959e-04 |
| `synthetic.clean_mse_after` | 2.400815e-06 |
| `synthetic.clean_mse_before` | 3.539260e-06 |
| `synthetic.clean_mse_reduction_pct` | 32.166189 |
| `synthetic.clean_snr_db_after` | -9.907591 |
| `synthetic.clean_snr_db_before` | -11.593130 |
| `synthetic.clean_snr_improvement_db` | 1.685539 |
| `synthetic.context_epochs` | 7 |
| `synthetic.n_windows` | 24990 |
| `synthetic.residual_error_rms_ratio` | 0.823613 |
| `synthetic.sfreq_hz` | 4096.000000 |

## Interpretation

DHCT-GAN v2 supervised proof-fit metrics on the Niazy bundle, with 7-epoch context input. Clean SNR improvement and artifact correlation should both move in the right direction relative to the no-correction baseline and relative to v1's single-epoch predictions.

## Artifacts

- `supervised_examples`: `plots/supervised_examples.png`

## Limitations

- Targets are AAS-derived. The clean reference is the AAS-corrected Niazy surrogate, so generalization claims beyond the AAS estimate are not warranted.
- Per-window demeaning is applied before inference; absolute baseline drift is not corrected.
- Real-data trigger-locked proxies and the FACET framework metric battery are not yet included.

## Configuration

```json
{
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/dhctganv2niazyprooffit_20260510_220534/exports/dhct_gan_v2.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "device": "cpu",
  "batch_size": 64,
  "context_epochs": 7,
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
    "context_epochs": 7,
    "clean_mse_before": 3.5392604331718758e-06,
    "clean_mse_after": 2.4008152195165167e-06,
    "clean_mae_before": 0.0009189959382638335,
    "clean_mae_after": 0.000887985632289201,
    "clean_snr_db_before": -11.593130111694336,
    "clean_snr_db_after": -9.907590866088867,
    "clean_snr_improvement_db": 1.6855392456054688,
    "clean_mse_reduction_pct": 32.16618938197476,
    "artifact_mse": 2.4008152195165167e-06,
    "artifact_mae": 0.000887985632289201,
    "artifact_corr": 0.5672793573218547,
    "artifact_snr_db": 1.6855376958847046,
    "residual_error_rms_ratio": 0.8236128230526188
  }
}
```
