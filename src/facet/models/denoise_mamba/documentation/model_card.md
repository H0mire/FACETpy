# DenoiseMamba Model Card

## Summary

`denoise_mamba` is a single-channel sequence-to-sequence gradient artifact
denoiser that stacks ConvSSD blocks: a local depthwise 1D convolution
combined with a Mamba-1 style selective state space (SSD) layer.
The selective scan is implemented in pure PyTorch.

## Intended Role

DenoiseMamba is the first State-Space-Model entry in the FACETpy thesis
benchmark. It belongs to the "Sequence Modeling (State Space / Mamba)"
family from `docs/research/architecture_catalog.md` and is the
implementation of Section 6.2 of
`docs/research/dl_eeg_gradient_artifacts.pdf`. It exists to test whether
linear-time selective state-space architectures match or exceed
autoencoder-style baselines (`cascaded_dae`, `cascaded_context_dae`) on the
Niazy proof-fit dataset.

## Input And Output

- Input: `(batch, 1, samples)` single-channel float32 EEG.
- Output: `(batch, 1, samples)` predicted gradient artifact.
- Default samples: `512` (matches the Niazy proof-fit context bundle's
  `target-epoch-samples`).
- Correction: predicted artifact is subtracted via `DeepLearningCorrection`.

## Compatibility Notes

- Compatible with arbitrary EEG channel counts because inference is
  channel-wise.
- Trigger metadata is not required at inference time. Chunks are slid over
  the raw signal in `chunk_size_samples` steps.
- The TorchScript checkpoint is coupled to the per-segment sample length
  used during training (default 512 samples).
- The model is implemented in pure PyTorch and does not depend on the
  `mamba-ssm` CUDA kernel. It runs on CPU for tests.

## Hyperparameters

Default training values (full configuration in
`training_niazy_proof_fit.yaml`):

| Hyperparameter | Default |
|---|---|
| `d_model` | 64 |
| `d_state` | 16 |
| `expand` | 2 |
| `d_conv` | 4 |
| `n_blocks` | 4 |
| `dropout` | 0.1 |
| `input_kernel_size` | 7 |
| optimizer | AdamW |
| `learning_rate` | 1e-3 |
| `weight_decay` | 1e-4 |
| `grad_clip_norm` | 1.0 |
| `batch_size` | 64 |
| `max_epochs` | 60 |
| loss | MSE |

## Training Reference

Full run:

```bash
uv run facet-train fit \
  --config src/facet/models/denoise_mamba/training_niazy_proof_fit.yaml
```

Smoke run:

```bash
uv run facet-train fit \
  --config src/facet/models/denoise_mamba/training_niazy_proof_fit_smoke.yaml
```

## Evaluation Notes

Use the standard evaluation structure described in
`src/facet/models/evaluation_standard.md`. Compare metrics against
`cascaded_dae` and `cascaded_context_dae` results from the same Niazy
proof-fit dataset.
