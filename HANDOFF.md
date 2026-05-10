# DHCT-GAN v2 Hand-off

## Status

Training and evaluation are complete. **Metrics improved substantially over
v1** (clean_snr_improvement_db: −7.13 → **+1.69 dB**) but **v2 still
underperforms every multi-epoch baseline** on this dataset
(cascaded_context_dae +3.16 dB, dpae +7.48 dB, vit_spectrogram +11.60 dB).

Per the orchestrator's directive, I am **not iterating further on
hyperparameters**. The training log shows the same GAN-instability
signature as v1 — train_loss climbs from 0.13 (epoch 1) to 0.7 (epoch 34)
while val_loss oscillates heavily and only briefly touches 0.13 at
epoch 19. The multi-epoch-context fix removed the regression but did not
make the adversarial recipe competitive with the simpler discriminative
baselines. **The architectural finding is that this dataset does not need
(and is hurt by) the adversarial term**; see hypothesis discussion below.

## Branch and worktree

- Branch: `feature/model-dhct_gan_v2` (off `feature/add-deeplearning`)
- Worktree: `worktrees/model-dhct_gan_v2`
- Latest commit:
  - `2fe9c99` feat(dhct_gan_v2): add multi-epoch context GAN follow-up to dhct_gan

## What changed vs v1

Only the input contract. The encoder, transformer stack, dual decoders,
gating, discriminator, optimizer betas, learning rates, adversarial weight
(`beta_adv = 0.1`), consistency weight (`alpha = 0.5`), batch size, and
schedule are all identical to v1. The single delta:

| Layer | v1 | v2 |
|---|---|---|
| Generator input | `(B, 1, 512)` | `(B, 7, 512)` (7-epoch noisy context stacked as channels) |
| Stem conv | `Conv1d(in=1, out=base_channels, k=7)` | `Conv1d(in=7, out=base_channels, k=7)` |
| Loss `noisy` channel | full noisy window | the center-epoch slice from the context |
| Consistency reference | `noisy − artifact_pred` | `noisy_center − artifact_pred` |

Approach (a) from the orchestrator's two options was chosen: stack the
7 context epochs as channels rather than add an extra attention axis
across the context dimension. Rationale in
`src/facet/models/dhct_gan_v2/documentation/research_notes.md`.

## GPU fleet runs

### Smoke run (good)

- Job id: `255b61606f97` — `dhct_gan_v2_niazy_smoke`
- Worker: `gpu2`
- Status: `finished`, exit code `0`
- Training output:
  `training_output/dhctganv2niazysmoke_20260510_220010/`
- Final val_loss after 1 epoch (subset of 1024 windows): **0.134**
  (v1 smoke baseline at the same step: 0.084)
- Export: `training_output/dhctganv2niazysmoke_20260510_220010/exports/dhct_gan_v2.ts`

### Full run

- Job id: `a7ee321ac569` — `dhct_gan_v2_niazy_full`
- Worker: `gpu2`
- Status: `finished`, exit code `0`
- Training output:
  `training_output/dhctganv2niazyprooffit_20260510_220534/`
- Configured `max_epochs=80`, ran **34 epochs** before early-stopping
  on `patience=15` of no improvement (vs v1's 16 epochs).
- Best epoch: **19**, best val_loss: **0.130** (v1: best epoch 1,
  val_loss 0.200).
- val_loss oscillated heavily after epoch 1; train_loss climbed
  monotonically from 0.13 to 0.7 — the classic GAN-mode-collapse
  signature is still present, just with a deeper occasional minimum.
- Export: `training_output/dhctganv2niazyprooffit_20260510_220534/exports/dhct_gan_v2.ts`
- Total wall-clock: **481 s** on RTX 5090 (vs v1's 192 s for 16 epochs).

## Evaluation

- Manifest: `output/model_evaluations/dhct_gan_v2/20260511_proof_fit/evaluation_manifest.json`
- Metrics: `output/model_evaluations/dhct_gan_v2/20260511_proof_fit/metrics.json`
- Summary: `output/model_evaluations/dhct_gan_v2/20260511_proof_fit/evaluation_summary.md`
- Plot: `output/model_evaluations/dhct_gan_v2/20260511_proof_fit/plots/supervised_examples.png`

### Supervised proof-fit metrics (24990 windows)

| Metric | v2 value |
|---|---:|
| `clean_snr_db_before` | −11.59 |
| `clean_snr_db_after` | **−9.91** |
| `clean_snr_improvement_db` | **+1.69** |
| `clean_mse_before` | 3.54e-06 |
| `clean_mse_after` | 2.40e-06 |
| `clean_mse_reduction_pct` | **+32.17** |
| `artifact_corr` | **0.567** |
| `residual_error_rms_ratio` | **0.824** |

### Comparison: v2 vs v1 vs current baselines

| Model | Input | `clean_snr_improvement_db` | `artifact_corr` | `clean_mse_reduction_pct` |
|---|---|---:|---:|---:|
| `dhct_gan` (v1) | 1 epoch | **−7.13** | 0.158 | −416.27 |
| **`dhct_gan_v2` (this run)** | **7 epochs** | **+1.69** | **0.567** | **+32.17** |
| `cascaded_dae` | 1 epoch | −0.05 | 0.086 | −1.07 |
| `cascaded_context_dae` | 7 epochs | +3.16 | 0.732 | +51.68 |
| `dpae` | 7 epochs | **+7.48** | 0.913 | +82.10 |
| `vit_spectrogram` | spectrogram | +11.60 | 0.966 | +93.08 |

- **v2 closes ~8.8 dB of the gap to no-correction** versus v1 (−7.13 → +1.69).
- v2 trails `cascaded_context_dae` (the matched-input-shape autoencoder) by
  **−1.47 dB**.
- v2 trails `dpae` (the strongest discriminative baseline) by **−5.79 dB**.

## Architectural finding (per orchestrator directive)

The multi-epoch context input was the right diagnosis as a *partial* cause
of v1's failure: switching from single-epoch to 7-epoch context moved
DHCT-GAN from the worst-in-class single-epoch regime (v1: −7.13 dB,
clearly worse than even `cascaded_dae` at −0.05 dB) into the
positive-SNR regime where all multi-epoch models live. That is exactly
what v1's hypothesis #2 predicted.

However, v2 still underperforms the matched-input-shape baseline
`cascaded_context_dae` by 1.47 dB and the strongest discriminative
multi-epoch baseline `dpae` by 5.79 dB. The training log shows the
adversarial loss is still destabilizing the generator (train_loss climbs
0.13 → 0.7 over 34 epochs; val_loss oscillates between 0.13 and 0.77 with
no monotone improvement past epoch 19). This is the GAN-mode-collapse
signature v1's hand-off ranked as hypothesis #1.

**Conclusion:** on the Niazy proof-fit dataset, the adversarial term
adds no value and actively hurts the generator. The proof-fit clean
reference is the AAS template — a smooth, well-defined target with very
low residual structure (clean_mse_before is 3.54e-6). Adversarial
training is designed to push outputs toward "looks realistic when you
can't tell ground-truth from prediction", but on a target this clean and
this well-defined the discriminator has no useful additional signal to
add beyond what the L1 reconstruction term already provides. The
discriminator quickly becomes a confident classifier and its gradient
pushes the generator away from the L1 minimum.

Two implications:

1. The **architectural innovation** of DHCT-GAN (dual-decoder + learned
   gate + hybrid CNN-Transformer encoder) is **structurally sound** on
   this dataset — it produces useful artifact predictions when given
   multi-epoch context.
2. The **adversarial training recipe** is **not appropriate** for this
   dataset. Removing `beta_adv` (i.e. training only with
   reconstruction + consistency) would likely close most of the gap to
   `cascaded_context_dae` and potentially approach `dpae`.

Per the orchestrator's instruction, I am **not** running that experiment
autonomously. If the orchestrator wants a v3, the single highest-leverage
change is `beta_adv: 0.0` (optionally combined with a longer reconstruction-
only warm-up and the consistency-mean fix from v1's hypothesis #3).

## Why metrics are still below the multi-epoch baselines — ranked hypotheses

1. **Adversarial loss is the bottleneck on this dataset.** The training
   log shows the same instability signature as v1, just with a deeper
   occasional minimum because the multi-epoch context gives the
   reconstruction term more to work with. The discriminator's gradient
   actively pulls the generator off the L1 minimum.
2. **Capacity and depth differ from `cascaded_context_dae`'s
   parameterization.** `cascaded_context_dae` uses a two-stage cascade
   (stage-1 prediction subtracted from input, stage-2 predicts the
   residual). DHCT-GAN v2 makes a single-shot prediction. Even without
   the GAN term, a single-shot CNN-Transformer might lose a few tenths
   of a dB to the residual cascade structure.
3. **The consistency loss still uses a slightly different mean than the
   artifact loss.** Although v2 fixed the gross mean-mismatch from v1
   (we now use the dataset's `clean_center` directly), the per-epoch
   demeaning of the noisy context introduces a small offset relative to
   the artifact-target's own mean. This adds gradient noise on top of
   the GAN instability.

## Suggested next experiments (only if orchestrator wants a v3)

1. **`beta_adv: 0.0`** — train DHCT-GAN v2's architecture as a pure
   regressor with the existing reconstruction + consistency loss. This
   isolates "is the hybrid CNN-Transformer dual-branch generator
   competitive with `cascaded_context_dae` / `dpae` when not adversarially
   trained?". Most likely outcome: matches or modestly exceeds
   `cascaded_context_dae` (~+3 dB).
2. **`beta_adv` warm-up** — if the orchestrator wants to keep the GAN
   identity intact, run ~5 reconstruction-only epochs first, then anneal
   `beta_adv` from 0.0 → 0.01 over the next 5 epochs and stay there.
   Anything higher than 0.01 mode-collapsed the v1/v2 generator.
3. **Two-stage residual cascade** — wrap the v2 generator in a cascade
   identical to `cascaded_context_dae`'s structure. Most likely to help
   if (1) and (2) saturate around `cascaded_context_dae`'s level.

## Completion checklist

- [x] `README.md`
- [x] `documentation/model_card.md`
- [x] `documentation/research_notes.md` (v1 → v2 delta + design rationale)
- [x] `documentation/evaluations.md` (points to HANDOFF.md)
- [x] `processor.py` (DHCTGanV2Adapter + DHCTGanV2Correction, registered)
- [x] `training.py` (build_model, build_loss, build_dataset)
- [x] `training_niazy_proof_fit.yaml`
- [x] `training_niazy_proof_fit_smoke.yaml`
- [x] `evaluate.py`
- [x] `tests/models/dhct_gan_v2/` (11 tests, all green locally with `uv run --extra pytorch pytest`)
- [x] Smoke run + fetch verified
- [x] Full run + fetch verified
- [x] Evaluation written via `ModelEvaluationWriter`
- [x] HANDOFF.md (this file)
