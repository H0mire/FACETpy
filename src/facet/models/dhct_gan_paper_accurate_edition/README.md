# DHCT-GAN — Paper-Accurate Edition

A strictly more faithful re-implementation of **DHCT-GAN** for use inside FACETpy's
EEG-fMRI gradient-artifact correction pipeline.

> Cai, Y., Meng, Z., Huang, X. *DHCT-GAN: Improving EEG Signal Quality with a
> Dual-Branch Hybrid CNN-Transformer Network.* MDPI **Sensors** 2025, 25(1):231.

This package (`facet.models.dhct_gan_paper_accurate_edition`) is a sibling of the
original `facet.models.dhct_gan`. It does **not** modify the original. It keeps the
same `facet-train` factory contract (`build_model` / `build_loss` /
`build_dataset`) and the same TorchScript ARTIFACT-subtraction inference contract,
but brings the architecture and losses much closer to the paper.

## What changed vs. the original `dhct_gan` (and why)

All changes below are paper-faithful unless marked **[FACETpy deviation]**.

### Losses

| Item | Original | Paper-accurate edition | Paper ref |
| --- | --- | --- | --- |
| Reconstruction | `nn.L1Loss` | `nn.MSELoss` | Eq. 10 |
| Adversarial / discriminator | vanilla BCE-with-logits | **LSGAN** least-squares (Eqs. 12-13): `L_adv = mean((D(G(X))-1)^2)`, `L_D = mean(0.5·D(G(X))^2 + 0.5·(D(Y)-1)^2)` | Eqs. 12-13 |
| Feature / perceptual loss | none | **`L_feat`** = mean MSE between intermediate discriminator conv features of target and prediction (`lambda_feat`, default 1.0) | Eq. 11 |
| Discriminators | 1 PatchGAN on the artifact head | **3 discriminators**: D1 (clean), D2 (artifact/noise), D3 (fused), each with its own private Adam (betas 0.9/0.999) | Eqs. 6-9, 13 |
| Generator loss | single combined objective | **`Loss1 + Loss2 + Loss3`**, each = MSE + `lambda_feat`·feature + `lambda_adv`·LSGAN | Eqs. 6-9 |

`lambda_feat`/`lambda_adv` are not numerically specified in the paper; defaults
(1.0 / 0.1) are documented guesses, keeping adversarial weight small because for
low-channel gradient-artifact regression the reconstruction term dominates.

### Architecture

| Item | Original | Paper-accurate edition | Paper ref |
| --- | --- | --- | --- |
| Gating fusion | single conv gate (sigmoid), `g·clean + (1-g)·(x-artifact)` | **two independent gating heads** with masks `Ymask1`/`Ymask2` (tanh), `Ypre = Ymask1·Y1 + Ymask2·(Xraw-Y2)`, masks not forced to sum to 1 | Eqs. 4-5 |
| LGTB local attention | fixed `window_size` (halved per stage) | **fixed block COUNT** `n_local_blocks` (default 8): split → attend per block → concat | Sec. 2 (Local SA) |
| LGTB inner depth | 1 | configurable `n_lgtb` (default 2; paper draws the residual stack x5) | Sec. 2 |
| CNN vs transformer path | sequential | **parallel CNN path** alongside the transformer path, fused (Conv+BN) | "CNN-LGTB" block |
| Preprocessing AvgPool | none | optional `stem_pool` AvgPool after the stem | Sec. 2 (Preprocessing) |

### Documented FACETpy-appropriate deviations

These keep the model sensible for ~25k single-channel **512-sample** Niazy
windows and narrow-band fMRI gradient artifacts, and are intentionally **not**
copied from the paper:

- **[FACETpy deviation] Encoder depth/width.** Default 4 stages, widths 16→128
  (paper: 5 stages, 64→1024 on 1024-sample inputs). 512-sample windows cannot
  survive 5× downsampling, and 64–1024 widths are oversized for narrow-band
  gradient artifacts. `depth`/`base_channels` are kwargs so a 1024-sample dataset
  can opt back into paper scale.
- **[FACETpy deviation] Conv-based gating heads** (not the paper's FC) — keeps the
  generator length-agnostic for `torch.jit.trace` export.
- **[FACETpy deviation] LayerNorm in the LGTB** where the paper labels BN — more
  stable for variable-length 1-D sequences and standard in transformer blocks.
- **[FACETpy deviation] Exported head = artifact (Y2).** The paper deploys the
  fused clean `Ypre` (output_type CLEAN). FACETpy subtracts a predicted artifact
  (output_type ARTIFACT), which is more robust for high-amplitude periodic
  gradient artifacts. `forward()` returns only the artifact head; the loss reads
  Y1/Y2/Ypre/masks via the private `_compute_outputs`.
- **[FACETpy deviation] Input length 512** (Niazy NPZ window), not the paper's
  1024 (2 s @ 512 Hz). The model is fully convolutional and length-agnostic;
  `epoch_samples` stays configurable.
- **[FACETpy deviation] Artifact domain.** The paper mixes EMG/EOG/ECG at SNR
  −7…+2 dB; FACETpy uses the Niazy AAS-derived fMRI-gradient NPZ. Dataset reader
  is unchanged from the original.
- **[FACETpy deviation] Generator optimizer betas.** The paper uses generator Adam
  betas 0.5/0.9. The standard `facet-train` wrapper builds a single AdamW over
  generator params with fixed betas 0.9/0.999, so 0.5/0.9 cannot be set without
  bypassing the CLI. The **discriminator** betas (0.9/0.999) match the paper and
  are applied by the private optimizers inside the loss module.

## How it stays compatible with `facet-train`

- `build_model` returns only the **generator** `nn.Module`, so the wrapper's single
  optimizer over `model.parameters()` trains the generator.
- The **three discriminators and their private Adam optimizers live inside the loss
  module** (`DHCTGanLossPA`). On each `loss_fn(pred, target)` call, when grad is
  enabled, one LSGAN discriminator step per discriminator runs on *detached*
  predictions before the generator loss is computed. In `eval`/`no_grad` the
  discriminator optimizers are not stepped. This preserves the single-CLI-optimizer
  contract while giving alternating GAN updates.
- The loss receives `pred = generator(x)` (the artifact head Y2). It reconstructs
  the clean/fused estimate analytically as `noisy - artifact` and supervises that
  with MSE + feature + adversarial terms without changing the wrapper's signature.
- **Limitation (be honest about it):** because the wrapper only passes the artifact
  head into the loss, the generator's `clean_decoder`, `clean_head`, `gate1`, and
  `gate2` receive **no gradient** — they are architecture-only and inert at
  inference. The reconstructed `clean_pred` and `fused_pred` are identical
  (`noisy - artifact`), so D1/D3 are redundant duplicates and `Loss1`/`Loss3` are
  the same term. This matches the original `dhct_gan`'s single-optimizer limitation
  (its gate is likewise dead code), so it is **not a regression**; the
  gradient-carrying gains (MSE, LSGAN, feature-matching, extra discriminator
  capacity, parallel CNN path, fixed-block attention, configurable LGTB depth) are
  the genuine improvements. See `documentation/paper_accuracy_review.md` for the
  full analysis and how a custom wrapper could fully realize Eqs. 4-9.

## Files

- `training.py` — generator, feature discriminator, 3-discriminator LSGAN loss,
  dataset, and the `build_model`/`build_loss`/`build_dataset` factories.
- `processor.py` — `DHCTGanPaperAccurateAdapter` + `@register_processor`
  `DHCTGanPaperAccurateCorrection` (unique name `dhct_gan_paper_accurate_correction`).
- `training_niazy_proof_fit_smoke.yaml` — illustrative CPU smoke config (not run
  by the pytest smoke test).
- `documentation/paper_accuracy_review.md` — full discrepancy table + EEG-fMRI
  applicability assessment.
