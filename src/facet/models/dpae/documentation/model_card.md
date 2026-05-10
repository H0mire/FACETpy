# DPAE Model Card

## Summary

`dpae` is a 1D-CNN dual-pathway denoising autoencoder that predicts the
gradient-artifact waveform from a single noisy EEG epoch. It instantiates the
architecture proposed by Xiong et al. 2023 (Frontiers in Neuroscience,
"A general dual-pathway network for EEG denoising"). It is the discriminative
reference baseline against the more experimental sequence-model, source-
separation, and diffusion models in the FACETpy thesis project.

## Intended Role

Discriminative control baseline for the Niazy proof-fit dataset. Cleanly
trainable and well-understood; a regression failure here suggests the dataset
or training stack is broken, not the architecture.

## Input And Output

- Input: `(batch, 1, samples)` — one noisy single-channel epoch.
- Default samples: `512` (matches the dataset builder's resampled epoch length
  and the paper's 2-second segment at 256 Hz).
- Output: `(batch, 1, samples)` — predicted artifact waveform.
- Correction: `DeepLearningCorrection` subtracts the prediction from the noisy
  signal, yielding the corrected EEG.

## Compatibility Notes

- Compatible with different EEG channel counts because inference is channel-wise.
- Requires trigger metadata at inference because native epochs are resampled
  to `epoch_samples` per trigger interval.
- Native artifact lengths may vary; the processor resamples each native epoch
  to the model-domain length and resamples the prediction back.

## Architectural Recipe

- Local pathway: 5 conv layers, kernel size 3, dilation rates `(1, 2, 4, 8, 1)`,
  channel widths `(F, F, 2F, 2F, latent_filters)`, two MaxPool(2) at the end.
- Global pathway: 3 conv layers, kernel sizes `(15, 11, 7)`, channel widths
  `(F, 2F, latent_filters)`, two MaxPool(2) interleaved.
- Both pathways downsample by a factor of 4 so their bottleneck shapes match.
- Fusion: concat over channel axis -> BatchNorm1d -> 1x1 Conv -> SeLU.
- Decoder: `ConvTranspose1d(stride=2) x 2 -> Conv1d(k=3) -> Conv1d(k=1)` to one
  channel. Output length equals input length.
- Residual: `output = decoder(...) + residual_scale * input` with
  `residual_scale` initialised to 0 (pure decoder output at start).

Default `base_filters=32`, `latent_filters=128` -> roughly 2 M parameters,
matching the paper's reported 1D-CNN parameter budget.

## Loss

Mean squared error on the artifact waveform (`torch.nn.MSELoss`).

## Training Reference

```bash
uv run facet-train fit --config src/facet/models/dpae/training_niazy_proof_fit.yaml
```

- Optimiser: AdamW (`facet-train` default)
- Learning rate: 1e-3
- Weight decay: 1e-4
- Batch size: 128
- Max epochs: 80, with early stopping (patience 15) on validation loss
- Gradient clipping at L2 norm 1.0

## Evaluation Notes

Use the standard evaluation structure described in
`src/facet/models/evaluation_standard.md` and emit run files via
`facet.evaluation.ModelEvaluationWriter`. Compare against
`cascaded_dae` (single-segment baseline) and `cascaded_context_dae`
(7-epoch context baseline).
