# DHCT-GAN v2 Model Card

## Summary

`dhct_gan_v2` is a per-channel multi-epoch-context dual-branch hybrid
CNN-Transformer generative adversarial denoiser adapted from
[DHCT-GAN (Cai et al., 2025)](https://www.mdpi.com/1424-8220/25/1/231) for
fMRI gradient artifact removal on the FACETpy Niazy proof-fit dataset.

It is a focused follow-up on `dhct_gan` (v1). The v1 model produced
`clean_snr_improvement_db = -7.13` (worse than no correction). The v1
hand-off ranked three hypotheses; v2 addresses only the second one —
single-epoch input is not enough context for periodic fMRI gradient
artifacts — by feeding the generator the full 7-epoch noisy context tensor
that is already exposed by the dataset bundle.

## Intended role

This model is the multi-epoch-context GAN-family entry in FACETpy. It
complements the multi-epoch auto-encoder baselines (`cascaded_context_dae`,
`dpae`) by adding adversarial supervision on the artifact-head output.

## Input and output

- Input: `(batch, context_epochs, samples)` per-channel context windows;
  `context_epochs = 7` and `samples = 512` by default. The center epoch is
  at index `context_epochs // 2`.
- Output (TorchScript): `(batch, 1, samples)` predicted artifact for the
  center epoch.
- Correction: predicted artifact is subtracted from the noisy center epoch
  via `DeepLearningCorrection`.

## Compatibility

- Channel-count agnostic — training and inference both operate per channel.
- Requires trigger metadata at inference to delimit trigger-to-trigger
  epochs. The processor only corrects trigger epochs that have at least
  `context_epochs // 2` neighbors on each side; the leading and trailing
  epochs are left untouched.
- Native artifact length may differ from the model-domain length; the
  processor resamples each epoch before prediction and resamples the
  predicted artifact back before subtraction.
- Checkpoint is coupled to the model-domain epoch length (default 512) and
  the number of context epochs (default 7).

## Training reference

- Dataset: Niazy proof-fit, 833 examples × 30 channels = 24990 windows.
- Generator parameters: ~2.7 M (stem input channels increased from 1 to 7
  vs v1; negligible parameter delta).
- Discriminator parameters: ~0.3 M (private to the loss module).
- Loss: `L1(artifact) + 0.5 * L1(noisy_center - artifact, clean) + 0.1 * BCE_adv`.
- Generator optimizer: AdamW, lr `1e-3`, weight decay `1e-4`, grad clip 1.0.
- Discriminator optimizer: Adam, lr `1e-4`, betas `(0.9, 0.999)`.
- Batch size: 64 (full) / 32 (smoke).
- Max epochs: 80 (full) / 1 (smoke).

## Evaluation

Follow `src/facet/models/evaluation_standard.md`. Outputs land in
`output/model_evaluations/dhct_gan_v2/<run_id>/`.

## Adaptations from the published model

- Encoder depth reduced from 5 to 4 stages to fit 512-sample windows.
- Channel widths reduced from 64–1024 to 16–128 to fit RTX 5090 VRAM and
  the much smaller proof-fit dataset.
- Three sibling discriminators (clean / noise / fused) collapsed to a
  single PatchGAN discriminator on the artifact head because the
  `facet-train` CLI uses a single optimizer over generator parameters.
- Adversarial weight `beta = 0.1` (softer than typical pix2pix) to maintain
  reconstruction primacy on EEG.
- **New for v2:** input channels expanded from 1 to `context_epochs` so the
  generator can exploit cross-epoch TR periodicity. All other architectural
  parameters are unchanged from v1 to keep the v1↔v2 comparison
  interpretable.

See `documentation/research_notes.md` for rationale.
