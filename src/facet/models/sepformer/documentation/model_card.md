# SepFormer Model Card

## Summary

`sepformer` is a compact channel-wise dual-path Transformer for fMRI
gradient artifact correction. It predicts the centre-epoch artifact
signal from a seven-epoch trigger-locked context.

The architecture follows Subakan et al. 2021 ("Attention is All You
Need in Speech Separation", arXiv:2010.13154):

1. 1D convolutional encoder maps the raw waveform to a feature
   sequence.
2. Feature sequence is split into chunks of size `K` with hop `K/2`.
3. `N` dual-path Transformer blocks alternate intra-chunk and
   inter-chunk multi-head self-attention.
4. A 1×1 mask network produces a mask which is multiplied with the
   encoded mixture.
5. Overlap-add reassembly + 1D transposed-convolution decode the masked
   features back to the waveform.
6. The centre epoch of the reconstructed waveform is returned as the
   artifact prediction.

## Intended role

First audio-source-separation baseline for FACETpy. Sits in the
"Audio-Inspired Source Separation" family of the architecture catalog
together with Conv-TasNet (TCN) and Demucs (U-Net + LSTM). The
architectural argument for trying SepFormer specifically is that its
dual-path attention mechanism maps cleanly onto the slice-vs-TR
structure of fMRI gradient artifacts:

- intra-chunk attention models the artifact within one slice
  acquisition,
- inter-chunk attention models drift across consecutive TRs.

## Input and output

- Input: `(batch, 7, 1, samples)`.
- Default samples: `512`.
- Output: `(batch, 1, samples)` centre-epoch artifact estimate.
- Correction: predicted artifact is subtracted by
  `DeepLearningCorrection`.

## Compatibility notes

- Compatible with different EEG channel counts because inference is
  channel-wise.
- Requires trigger metadata.
- Native artifact lengths may vary; the processor resamples native
  epochs to the model-domain length and resamples predictions back.
- The checkpoint is coupled to context length and model-domain epoch
  length (defaults `context_epochs=7`, `epoch_samples=512`).

## Default hyperparameters

The compact configuration used on the Niazy proof-fit dataset
(`training_niazy_proof_fit.yaml`):

| Hyperparameter | Value |
|---|---:|
| Encoder filters | 128 |
| Encoder kernel / stride | 16 / 8 |
| Dual-path blocks `N` | 2 |
| Intra layers / block | 4 |
| Inter layers / block | 4 |
| `d_model` | 128 |
| `d_ffn` | 256 |
| Attention heads | 4 |
| Chunk size `K` | 64 feature frames |
| Dropout | 0.1 |
| Mask activation | ReLU |
| Loss | MSE (configurable: `si_snr`, `si_snr_mse`) |
| Optimizer | AdamW (`lr=5e-4`, `wd=1e-4`) |
| Batch size | 64 |
| Max epochs | 50 (early-stop patience 10) |

Approximate parameter count: **≈ 2.3 M**.

See `documentation/research_notes.md` for the parameter-count
derivation and the reduction relative to the original SepFormer (≈ 26 M
params for speech separation).

## Current training reference

```bash
uv run facet-train fit --config src/facet/models/sepformer/training_niazy_proof_fit.yaml
```

## Evaluation notes

Use the standard evaluation structure described in
`src/facet/models/evaluation_standard.md`. Baseline comparisons of
interest: `cascaded_dae`, `cascaded_context_dae`. Other audio-family
models (`conv_tasnet`, `demucs`) when available.
