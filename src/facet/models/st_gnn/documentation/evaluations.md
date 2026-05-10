# ST-GNN Evaluations

Standardised evaluation runs for `st_gnn` should be stored under:

```text
output/model_evaluations/st_gnn/<run_id>/
```

Each run must include:

- `evaluation_manifest.json`
- `metrics.json`
- `evaluation_summary.md`

The first evaluation should compare:

- supervised reconstruction error on the Niazy proof-fit dataset
- artifact correlation against the AAS-derived target
- `cascaded_context_dae` baseline on the same dataset
- `cascaded_dae` baseline on the same dataset
- spatial-consistency plots (per-electrode RMSE topomap) to highlight
  any improvements expected from the graph constraint

Runs are produced by following the standard
`facet.evaluation.ModelEvaluationWriter` flow. Do not invent a custom
metrics format.

## Runs

### `niazy_proof_fit_20260510_211512`

- Training run: `training_output/spatiotemporalgnnniazyprooffit_20260510_211512/`
- Checkpoint: `exports/st_gnn.ts`
- Validation split: 166 examples (val_ratio=0.2, seed=42)
- Best training val MSE: `2.70e-07`

Headline evaluation metrics on the held-out validation split:

| Metric                              | Value     |
| ----------------------------------- | --------- |
| `clean_reconstruction_mse_before`   | 3.56e-06  |
| `clean_reconstruction_mse_after`    | 2.82e-07  |
| `clean_snr_db_before`               | -10.94 dB |
| `clean_snr_db_after`                |  +0.06 dB |
| `artifact_prediction_mse`           | 2.82e-07  |
| `artifact_prediction_correlation`   | 0.960     |
| `residual_rms_ratio`                | 0.277     |
| `rms_noisy`                         | 1.91e-03  |
| `rms_corrected`                     | 1.27e-04  |

The SNR uplift is ~11 dB and the predicted artifact correlates 0.96
with the AAS-derived target. The residual_rms_ratio of 0.28 indicates
that ~72% of the noisy waveform RMS is removed; the remainder is the
underlying clean signal plus a small artifact residual.
