# DHCT-GAN Model Card

## Summary

`dhct_gan` is a per-channel dual-branch hybrid CNN-Transformer generative
adversarial denoiser adapted from
[DHCT-GAN (Cai et al., 2025)](https://www.mdpi.com/1424-8220/25/1/231) for
fMRI gradient artifact removal on the FACETpy Niazy proof-fit dataset.

## Intended role

This model is the first GAN-family baseline in FACETpy. It complements the
auto-encoder baselines (`cascaded_dae`, `cascaded_context_dae`) by adding a
discriminative signal that pushes the generator toward sharper, more
realistic artifact morphology than pure MSE reconstruction.

## Input and output

- Input: `(batch, 1, samples)` per-channel noisy time-domain windows.
- Default samples: 512 (Niazy proof-fit dataset).
- Output (TorchScript): `(batch, 1, samples)` predicted artifact.
- Correction: predicted artifact is subtracted via `DeepLearningCorrection`.

## Compatibility

- Channel-count agnostic — training and inference both operate per channel.
- Requires trigger metadata at inference to delimit trigger-to-trigger epochs.
- Native artifact length may differ from the model-domain length; the
  processor resamples each epoch before prediction and resamples the
  predicted artifact back before subtraction.
- Checkpoint is coupled to the model-domain epoch length (default 512).

## Training reference

- Dataset: Niazy proof-fit, 833 examples × 30 channels = 24990 windows.
- Generator parameters: ~2.7 M.
- Discriminator parameters: ~0.3 M (private to the loss module).
- Loss: `L1(artifact) + 0.5 * L1(noisy - artifact, clean) + 0.1 * BCE_adv`.
- Generator optimizer: AdamW, lr `1e-3`, weight decay `1e-4`, grad clip 1.0.
- Discriminator optimizer: Adam, lr `1e-4`, betas `(0.9, 0.999)`.
- Batch size: 64 (full) / 32 (smoke).
- Max epochs: 80 (full) / 1 (smoke).

## Evaluation

Follow `src/facet/models/evaluation_standard.md`. Outputs land in
`output/model_evaluations/dhct_gan/<run_id>/`.

## Adaptations from the published model

- Encoder depth reduced from 5 to 4 stages to fit 512-sample windows.
- Channel widths reduced from 64-1024 to 16-128 to fit RTX 5090 VRAM and the
  much smaller proof-fit dataset.
- Three sibling discriminators (clean / noise / fused) collapsed to a single
  PatchGAN discriminator on the artifact head because the `facet-train` CLI
  uses a single optimizer over generator parameters.
- Adversarial weight `beta = 0.1` (softer than typical pix2pix) to maintain
  reconstruction primacy on EEG.

See `documentation/research_notes.md` for rationale.
