# IC-U-Net Evaluations

This file links specific evaluation runs to their generated artifacts.
Generated outputs live in `output/model_evaluations/ic_unet/<run_id>/` and
follow `src/facet/models/evaluation_standard.md`.

## Runs

### `20260510_full` — Niazy proof-fit, 16 epochs

- **Training run**: `training_output/icunetniazyprooffit_20260510_223556/`
  (16 epochs, early-stopped, best_metric ≈ 1.37 × 10⁻⁶, 17.6 s wall-clock on
  RTX 5090).
- **Checkpoint**:
  `training_output/icunetniazyprooffit_20260510_223556/exports/ic_unet.ts`
- **Dataset**:
  `output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz`
  (833 examples, 7 × 30 × 512, 4096 Hz).
- **Evaluation manifest**:
  `output/model_evaluations/ic_unet/20260510_full/evaluation_manifest.json`
- **Metrics**:
  `output/model_evaluations/ic_unet/20260510_full/metrics.json`

#### Headline numbers (Niazy proof-fit, AAS-corrected target)

| Metric | IC-U-Net | DPAE | cascaded_context_dae |
|---|---:|---:|---:|
| `clean_snr_db_before` | −11.59 | −11.59 | −11.59 |
| `clean_snr_db_after` | **+0.17** | −4.12 | +0.00 |
| `clean_snr_improvement_db` | **+11.77** | +7.48 | +11.60 |
| `clean_mse_reduction_pct` | **93.3 %** | 82.1 % | 93.1 % |
| `artifact_corr` | **0.967** | 0.913 | 0.966 |
| `artifact_mse` | 2.36 × 10⁻⁷ | 6.33 × 10⁻⁷ | 2.45 × 10⁻⁷ |
| `residual_error_rms_ratio` | **0.258** | 0.423 | 0.263 |
| `trigger_locked_rms_reduction_pct` | 97.0 % | – | – |

Comparison numbers for DPAE come from `output/model_evaluations/dpae/20260510_193431/metrics.json`;
for cascaded_context_dae from `output/model_evaluations/cascaded_context_dae/20260510_vit_compare/metrics.json`.

## Comparison baselines

- `cascaded_context_dae` — closest peer; same 7-epoch context but channel-wise.
- `cascaded_dae` — single-channel windowed DAE baseline.
- `dpae` (+7.48 dB) — the discriminative-CNN comparator from
  `docs/research/dl_eeg_gradient_artifacts.pdf` Section 3.2.
