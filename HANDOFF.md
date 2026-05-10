# DHCT-GAN Hand-off

## Status

Training and evaluation are complete, but **metrics are poor**: the model
makes the signal noisier than no correction at all (`clean_snr_improvement_db
= −7.13`). Adversarial training destabilized the generator after epoch 1.

The model is **salvageable**, not fundamentally unsuitable. The dual-branch
hybrid CNN-Transformer architecture is sound — the failure is in the
training recipe (adversarial weight, single-epoch input contract). See
hypotheses + next experiments below.

## Branch and worktree

- Branch: `feature/model-dhct_gan`
- Worktree: `worktrees/model-dhct_gan` (off `feature/add-deeplearning`)
- Latest commits:
  - `7e81eee` feat(dhct_gan): add supervised evaluation script
  - `3bab219` fix(dhct_gan): make generator traceable for torchscript export
  - `0adb076` feat(models): add dhct_gan model package and tests

## GPU fleet runs

### Smoke run (good)

- Job id: `6fd3be33cd56` — `dhct_gan_niazy_smoke`
- Worker: `gpu2`
- Status: `finished`, exit code `0`
- Training output:
  `training_output/dhctganniazysmoke_20260510_212829/`
- Final val_loss after 1 epoch (subset of 1024 windows): **0.084**
- Export: `training_output/dhctganniazysmoke_20260510_212829/exports/dhct_gan.ts`

### Full run (red flag)

- Job id: `b4c4d0e87ead` — `dhct_gan_niazy_full`
- Worker: `gpu2`
- Status: `finished`, exit code `0`
- Training output:
  `training_output/dhctganniazyprooffit_20260510_213159/`
- Configured `max_epochs=80`, ran **16 epochs** before early-stopping on
  `patience=15` of no improvement.
- Best epoch: **1**, best val_loss: **0.200**.
- val_loss climbed monotonically after epoch 1 (epoch 16: 0.67).
- Export: `training_output/dhctganniazyprooffit_20260510_213159/exports/dhct_gan.ts`
- Total wall-clock: 192 s on RTX 5090.

## Evaluation

- Manifest: `output/model_evaluations/dhct_gan/20260510_233500_proof_fit/evaluation_manifest.json`
- Metrics: `output/model_evaluations/dhct_gan/20260510_233500_proof_fit/metrics.json`
- Summary: `output/model_evaluations/dhct_gan/20260510_233500_proof_fit/evaluation_summary.md`
- Plot: `output/model_evaluations/dhct_gan/20260510_233500_proof_fit/plots/supervised_examples.png`

### Supervised proof-fit metrics (24990 windows)

| Metric | Value |
|---|---:|
| `clean_snr_db_before` | −11.59 |
| `clean_snr_db_after` | **−18.72** |
| `clean_snr_improvement_db` | **−7.13** |
| `clean_mse_before` | 3.54e-06 |
| `clean_mse_after` | 1.83e-05 |
| `clean_mse_reduction_pct` | **−416.27** |
| `artifact_corr` | **0.158** |
| `residual_error_rms_ratio` | 2.27 |

### Comparison vs current baselines

| Model | `clean_snr_improvement_db` | `artifact_corr` | `clean_mse_reduction_pct` |
|---|---:|---:|---:|
| **dhct_gan (this run)** | **−7.13** | **0.158** | **−416.27** |
| `cascaded_dae` (single-epoch DAE) | −0.05 | 0.086 | −1.07 |
| `cascaded_context_dae` (7-epoch DAE) | +3.16 | 0.732 | +51.68 |
| `vit_spectrogram` | +11.60 | 0.966 | +93.08 |

Single-epoch input is clearly the weak class on this dataset
(`cascaded_dae` is also negative). DHCT-GAN under-performs even
`cascaded_dae` because the adversarial term drags it further into a
degenerate mode.

## Why the metrics are poor — ranked hypotheses

1. **Adversarial loss dominates and destabilizes the generator.**
   With `beta_adv = 0.1` plus the discriminator's private `lr = 1e-4` and
   `betas = (0.9, 0.999)`, the disc converges to a confident classifier in
   the first ~hundred steps, after which its gradient pushes the generator
   toward perceptually "noisier" outputs that fool the disc rather than
   recover the artifact morphology. The val_loss climb starting at epoch
   2 is the classic GAN-mode-collapse signature.
2. **Single-epoch input is not enough context for gradient artifacts.**
   `cascaded_dae` (the only other single-epoch model) is also negative on
   this benchmark, while every model with multi-epoch context is positive.
   Gradient artifacts are strongly periodic across TR boundaries; a model
   that sees only the center epoch cannot exploit that periodicity. The
   architectural innovation of DHCT-GAN (dual decoder + gating) does not
   compensate for the missing temporal context.
3. **Consistency loss redundant and possibly conflicting.**
   `L1(noisy − artifact_pred, clean_target)` and
   `L1(artifact_pred, artifact_target)` are mathematically equivalent only
   when both are demeaned the same way. The dataset wrapper demeans
   `noisy` and `clean` using the **noisy** mean but demeans `artifact`
   using its **own** mean, so the two L1 terms pull the prediction
   toward two slightly different offsets. This adds gradient noise.

## Suggested next experiments (in priority order)

1. **Disable adversarial training first** — set `beta_adv = 0.0` in
   `training_niazy_proof_fit.yaml`. If the generator on its own reaches
   `cascaded_dae`-level metrics or better, the architecture is at least
   competitive in the single-epoch class. Then re-introduce the adversarial
   term with `beta_adv ≤ 0.01` and warm it up after a few reconstruction-
   only epochs.
2. **Switch to a multi-epoch input** — same generator backbone, but feed
   `(7, 1, 512)` like `cascaded_context_dae` does. This is the single
   biggest leverage point on this dataset. The encoder stem just needs
   to accept `n_context_epochs` channels; the rest of the model can stay
   the same. Compare apples-to-apples against `cascaded_context_dae`
   (single best autoencoder so far).
3. **Drop the consistency loss** — train only with
   `L1(artifact_pred, artifact_target)`. Re-add only after the
   recon-only run converges, and only after fixing the
   noisy-vs-artifact mean mismatch in the dataset wrapper.
4. **Lower the generator learning rate** to `1e-4` (matching the disc).
   The current `1e-3` is aggressive for a transformer-equipped
   generator on a smallish dataset.
5. **Increase model capacity** — `base_channels = 32` and
   `depth = 5` (with `epoch_samples = 1024`, matching the published
   model). Pair this with the multi-epoch input change for a stronger
   baseline.

## Salvage assessment

The model is **salvageable**, not fundamentally unsuitable. The dual-branch
hybrid CNN-Transformer generator is structurally sound (the smoke run on
a smaller subset hit val_loss 0.084 in 1 epoch). The training recipe is
the actual problem. The single biggest blocker is probably the
single-epoch input, not the adversarial loss. The orchestrator should
not respawn another GAN agent without first deciding between (a)
keeping single-epoch but disabling adversarial training, or
(b) reworking the input to multi-epoch context. Both can be tried in
parallel.

## Completion checklist

- [x] `README.md`
- [x] `documentation/model_card.md`
- [x] `documentation/research_notes.md`
- [x] `documentation/evaluations.md`
- [x] `processor.py` (Adapter + Correction subclass, registered)
- [x] `training.py` (build_model, build_loss, build_dataset)
- [x] `training_niazy_proof_fit.yaml`
- [x] `training_niazy_proof_fit_smoke.yaml`
- [x] `evaluate.py`
- [x] `tests/models/dhct_gan/` (9 tests, all green locally)
- [x] Smoke run + fetch verified
- [x] Full run + fetch verified
- [x] Evaluation written via `ModelEvaluationWriter`
- [x] HANDOFF.md (this file)
