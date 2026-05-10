# DHCT-GAN Evaluations

This file is the human-readable index of evaluation runs for the
`dhct_gan` model. Each run's machine-readable artifacts live under
`output/model_evaluations/dhct_gan/<run_id>/`.

## Runs

### `20260510_233500_proof_fit` — first full-training evaluation

- Source training run: `training_output/dhctganniazyprooffit_20260510_213159/`
  - 16 epochs, early-stopped on patience=15.
  - Best epoch: **1** (val_loss 0.200).
  - Loss climbed monotonically from epoch 2 onwards (val_loss reached 0.67 at
    epoch 16). The adversarial term destabilized training.
- Checkpoint: `training_output/dhctganniazyprooffit_20260510_213159/exports/dhct_gan.ts`
- Dataset: `output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz` (24990 windows × 512 samples).
- Manifest: `output/model_evaluations/dhct_gan/20260510_233500_proof_fit/evaluation_manifest.json`

Metric table (supervised group):

| Metric | DHCT-GAN | cascaded_dae | cascaded_context_dae | vit_spectrogram |
|---|---:|---:|---:|---:|
| `clean_snr_improvement_db` | **−7.13** | −0.05 | +3.16 | +11.60 |
| `artifact_corr` | **0.158** | 0.086 | 0.732 | 0.966 |
| `clean_mse_reduction_pct` | **−416.27** | −1.07 | +51.68 | +93.08 |
| `residual_error_rms_ratio` | **2.27** | ~1.0 | ~0.7 | ~0.26 |

DHCT-GAN is the worst-performing model so far on the proof-fit benchmark. It
makes the signal noisier than no correction at all.

## Metric groups reported

Supervised synthetic-style metrics on the Niazy proof-fit dataset
(paired noisy / clean / artifact). Real-data trigger-locked proxies and
the FACET framework metric battery are not yet computed for this model.

## Comparison anchors

- `cascaded_dae` (no context) — also single-epoch.
- `cascaded_context_dae` (7-epoch context).
- `vit_spectrogram` — current best on this benchmark.
