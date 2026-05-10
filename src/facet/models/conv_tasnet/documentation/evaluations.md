# Conv-TasNet Evaluations

Standardised evaluation runs for `conv_tasnet` are stored under:

```text
output/model_evaluations/conv_tasnet/<run_id>/
```

Each run must contain:

- `evaluation_manifest.json`
- `metrics.json`
- `evaluation_summary.md`
- `plots/`

The first evaluation should compare:

- supervised synthetic correction on the 512-sample Niazy proof-fit
  context dataset
- Niazy real-data trigger-locked proxy metrics
- visual plots showing noisy mixture, predicted artifact, predicted
  clean, and reference clean for at least one representative epoch
  per channel
- runtime and VRAM use against `cascaded_context_dae` and
  `cascaded_dae` baselines

## Run `20260510_224113` — Niazy proof-fit val split (supervised)

- Run directory: `output/model_evaluations/conv_tasnet/20260510_224113/`
- Checkpoint: `training_output/convtasnetniazyprooffit_20260510_202818/exports/conv_tasnet.ts`
- Training run: 13 epochs, early-stopped, best epoch 12 (val MSE 1.98e-8).
- Validation split: 4998 (channel × example) pairs from
  `output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz`,
  same 20 % seed-42 split that training held out.

Key metrics:

| Metric | Value |
| --- | ---: |
| `clean_snr_db_before` | -11.61 dB |
| `clean_snr_db_after` | 10.42 dB |
| `clean_snr_improvement_db` | **+22.03 dB** |
| `clean_mse_reduction_pct` | **99.37 %** |
| `artifact_corr` | **0.997** |
| `artifact_snr_db` | 22.03 dB |
| `residual_error_rms_ratio` | 0.079 |
| `predicted_source_sum_mse` | 5.52e-09 |

Interpretation: Conv-TasNet trained from scratch produces a highly
correlated artifact estimate on the held-out split, and the predicted
clean and artifact sources sum back to the mixture with very low error
even though source-additivity is not enforced during training. These
numbers are dataset-favourable: they measure in-distribution quality on
the same Niazy/AAS-derived artifact library used for training. Real
EDF trigger-locked evaluation and cross-subject generalisation are
listed as next steps.

Comparison context (orientation only, baselines were evaluated on the
synthetic-spike dataset, not the Niazy proof-fit set, so not directly
comparable):

| Model | Dataset | clean_snr_improvement_db | artifact_corr | residual_rms_ratio |
| --- | --- | ---: | ---: | ---: |
| cascaded_dae 20260502_115914 | synthetic_spike | -0.05 | 0.09 | 1.005 |
| cascaded_context_dae 20260502_115926 | synthetic_spike | +3.16 | 0.73 | 0.695 |
| conv_tasnet 20260510_224113 | niazy_proof_fit | **+22.03** | **0.997** | **0.079** |

When a run completes, update this file with:

- the run id and run directory
- a one-paragraph interpretation
- the metric table emitted by `ModelEvaluationWriter`
