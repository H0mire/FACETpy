# Nested-GAN Model Card

## Summary

Nested-GAN is a generator-only realisation of the two-stage time-frequency /
time-domain GAN described in *End-to-End EEG Artifact Removal Method via
Nested Generative Adversarial Network* (PMID 41183389). The published
paper uses two adversarial discriminator banks; the FACETpy
implementation trains the generator with a multi-resolution STFT loss
that captures the same spectral fidelity without an alternating GAN
training loop. The justification is recorded in
[`research_notes.md`](research_notes.md).

## Intended Role

A generative-family baseline alongside `dhct_gan` (and `dhct_gan_v2`
when available). Tests the hypothesis that **nested time-frequency
decomposition outperforms single-domain GAN refinement** when removing
the fMRI gradient artifact from EEG.

## Input and Output

- Input shape: `(batch, 7, 1, 512)` — seven trigger-defined context
  epochs per channel, 512 samples per epoch.
- Output shape: `(batch, 1, 512)` — predicted center-epoch artifact.
- Output type: `artifact`. The pipeline subtracts the prediction.

## Architecture

1. **Inner spectrogram Restormer**
   - Center epoch → STFT (`n_fft=64`, `hop=16`, Hann window, `center=True`).
   - 2-channel real/imag input → Conv2d projection to feature space.
   - N Restormer blocks: each block is `MDTA` (multi-DConv head
     transposed attention from Zamir 2022) plus `GDFN` (gated DConv
     feed-forward), with channel-wise layer norm and residual
     connections.
   - Conv2d projection back to 2 channels (real/imag artifact spec).
   - iSTFT → time-domain inner artifact estimate, length 512.

2. **Outer time-domain refiner**
   - Build a 7-channel × 512-sample tensor from the noisy context.
   - Replace the center-channel slot with `context_center − inner_output`
     so the outer branch refines the inner branch's residual.
   - 1D U-Net (3 encoder levels, base width 32, kernel 5, GELU,
     average-pooling downsample, transposed-conv upsample, skip
     connections). Final 1×1 conv produces a single-channel residual
     artifact correction.

3. **Final prediction**
   - `inner_artifact + outer_residual` → `(batch, 1, 512)`.

## Loss

`L1(prediction, target) + 0.5 · MultiResolutionSTFTLoss(prediction, target)`
where the multi-resolution loss is the mean L1 error on
`log(|STFT|)` at four window sizes `{32, 64, 128, 256}` with
hop = window / 4.

## Training Defaults

Full run (`training_niazy_proof_fit.yaml`):

- Optimizer: AdamW, lr=5e-4, weight_decay=1e-4
- grad_clip_norm: 1.0
- batch_size: 64
- max_epochs: 80
- val_ratio: 0.2
- early_stopping patience: 15

Smoke run (`training_niazy_proof_fit_smoke.yaml`):

- max_epochs: 1, batch_size 32, smaller model (`inner_channels=32`,
  `inner_blocks=2`, `outer_base_channels=16`).

## Compatibility Notes

- Channel-wise inference — checkpoint is independent of channel count.
- Triggers required at inference.
- Native artifact epochs may have different lengths; the processor
  resamples each trigger-to-trigger epoch to the model length
  (`epoch_samples`) before prediction and resamples the predicted
  center artifact back to the native epoch length before subtraction.
- Edge epochs (the first three and last three epochs) are not
  corrected because the 7-epoch context cannot be assembled.

## Evaluation

Use the standard layout described in
`src/facet/models/evaluation_standard.md`. See `evaluations.md`.
