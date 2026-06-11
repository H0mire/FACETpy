# DHCT-GAN v2 — Paper-Accurate Edition

A more faithful reimplementation of the DHCT-GAN generator/discriminator/loss
for the FACETpy EEG-fMRI gradient-artifact pipeline.

**Source paper:** Cai, Y., Meng, Z. & Huang, J. *DHCT-GAN: Improving EEG Signal
Quality with a Dual-Branch Hybrid CNN-Transformer Network.* MDPI **Sensors**
25(1), 231 (2025). https://doi.org/10.3390/s25010231

This package is a **new, additive** edition. It does **not** modify the original
`src/facet/models/dhct_gan_v2/`. Both editions can be imported simultaneously:
the processor here registers under the globally-unique name
`dhct_gan_v2_paper_accurate_correction` (the original keeps
`dhct_gan_v2_correction`).

## What the paper actually specifies

DHCT-GAN is a conditional GAN that removes physiological artifacts (EMG/EOG/ECG/
mixed) from 2 s @ 512 Hz single-channel EEG. The generator is a U-shaped hybrid
CNN-Transformer with two parallel branches (a CleanEEG branch `Y1` and a
NoiseEEG branch `Y2`); two tanh gating networks fuse them as
`Y_pre = mask1 ⊙ Y1 + mask2 ⊙ (X_raw − Y2)` (Eq. 4-5). Three discriminators
(D1 clean / D2 noise / D3 fused) drive a per-branch loss
`Loss_i = L_mse + λ1·L_feat + λ2·L_adv` with **LSGAN** least-squares
adversarial/discriminator terms (Eq. 12-13), **MSE** reconstruction (Eq. 10),
and a discriminator **feature-matching** term (Eq. 11). The encoder repeats a
Local-Global Transformer Block (LGTB: LSA over 8 fixed blocks → FF → GSA → FF)
five times per stage.

## What changed vs the original `dhct_gan_v2` (paper-faithful fixes)

| # | Change | Paper reference |
|---|--------|-----------------|
| 1 | **LSGAN** least-squares adversarial + discriminator loss replaces vanilla BCE. Discriminators output raw scores (no sigmoid). | Eq. 12-13 |
| 2 | **Feature-matching loss** `L_feat` added — MSE between an intermediate discriminator activation for the real vs predicted signal. | Eq. 11 |
| 3 | **MSE reconstruction** is now the default (`recon="mse"`); L1 retained as a documented spike-robust alternative (`recon="l1"`). | Eq. 10 |
| 4 | **Three discriminators** (clean / noise / fused), one shared architecture, one shared private Adam (betas 0.9, 0.999, lr 1e-4) stepped inside the loss → preserves the facet-train single-optimizer contract. | Algorithm 1 |
| 5 | **Two independent tanh gating networks** producing `mask1`, `mask2` (not constrained to sum to 1) replace the single complementary sigmoid gate; fusion follows Eq. 4-5. | Eq. 4-5 |
| 6 | **Paper-faithful LGTB**: local self-attention splits the sequence into a configurable number of *equal blocks* (default 8) instead of a sliding window; a feedforward follows *both* the local and the global attention; the LGTB is repeated `lgtb_depth` times per stage. | Sec. 3.2, Fig. 2a |
| 7 | **Paper discriminator structure**: strided conv blocks (k=3, s=2, p=1) + BN + LeakyReLU with the paper's doubling-every-two-layers channel progression (64,64,128,128,…), global pool + FC → scalar (configurable/scaled for CPU). | Fig. 2b |

## Deliberate, documented deviations (kept for EEG-fMRI / CPU)

These are **not** faithfulness bugs — they are intentional adaptations for
single-/few-channel fMRI gradient-artifact removal on a tiny proof-fit dataset,
runnable on CPU. See `documentation/paper_accuracy_review.md` for the full
rationale.

- **7-epoch trigger-aligned context input.** Not in the paper (single-segment),
  but the gradient artifact is strongly TR-periodic, so cross-epoch context is
  essential. The stem mixes the 7 context epochs.
- **Shared encoder + dual decoders** instead of two fully-duplicated branch
  generators — roughly halves parameters/CPU cost. Exposed conceptually via the
  shared-encoder design; full duplication is a GPU-only option.
- **Reduced dims** (512-sample epochs, depth 4, channel widths 16,32,64,128 vs
  the paper's 1024-sample, 5-stage, 64..1024). The proof-fit NPZ uses
  512-sample resampled trigger-to-trigger epochs and is tiny. All dims are
  kwargs, so a faithful full-scale run is possible on GPU.
- **Subtractive output.** `forward` returns the *artifact*
  `noisy_center − fused_clean`, not the clean signal `Y_pre` the paper emits,
  because FACETpy subtracts a predicted gradient artifact (more robust here).
  The artifact is derived from the full fused path, so the gating still
  influences inference.
- **Data domain.** Trained on the Niazy fMRI gradient-artifact proof-fit NPZ
  bundle, not EMG/EOG/ECG at −7..2 dB SNR.
- **Conv-for-FC gating.** The paper's gating networks are FC layers; FACETpy
  works on variable-length resampled epochs, so we use the 1D-conv equivalent
  (kernel 3 + 1×1) with tanh.

## Files

- `training.py` — paper-faithful generator, discriminator, loss + the
  `build_model` / `build_loss` / `build_dataset` facet-train factories.
- `processor.py` — `DHCTGanV2PaperAccurateAdapter` +
  `@register_processor DHCTGanV2PaperAccurateCorrection`
  (name `dhct_gan_v2_paper_accurate_correction`).
- `training_niazy_proof_fit_smoke.yaml` — illustrative tiny CPU config (not
  executed by the test).
- `documentation/paper_accuracy_review.md` — discrepancy table + EEG-fMRI
  applicability assessment.

## Loss / factory contract

`build_loss(name=None, recon="mse", alpha_consistency=0.5, lambda_feat=0.1,
lambda_adv=0.1, disc_channels=16, disc_depth=8, disc_lr=1e-4, …)` returns a
`nn.Module` whose `forward(pred, target)` expects `pred` of shape `(B, 1, T)`
(the artifact = `noisy_center − fused_clean`) and `target` of shape `(B, 3, T)`
packing `[artifact_target, clean_target, noisy_center]`. The three
discriminators step internally against detached generator outputs each
gradient-enabled forward; the returned scalar is the summed generator loss
`Loss1 + Loss2 + Loss3`.
