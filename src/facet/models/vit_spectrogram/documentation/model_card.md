# ViT Spectrogram Inpainter — Model Card

## Summary

`vit_spectrogram` is a Vision-Transformer-based spectrogram inpainter that
removes the fMRI gradient artifact (GA) from EEG by reconstructing the
clean center epoch from a 7-epoch trigger-defined context, operating in the
time-frequency domain. It is the first member of the **Vision-Inspired
(Spectrogram Inpainting)** family described in
`docs/research/dl_eeg_gradient_artifacts.pdf`, Section 7.2.1.

## Intended Role

This model is a baseline for vision-style artifact-removal architectures.
It complements the time-domain `cascaded_dae` and `cascaded_context_dae`
baselines by exploring whether a transformer that reasons about the GA in
the time-frequency plane can reconstruct cleaner EEG using only the
non-GA portions of the surrounding context.

## Input And Output

- Inference input contract: `(batch, 7, 1, samples)`.
- Inference output contract: `(batch, 1, samples)` predicted **clean**
  center epoch (the inpainted reconstruction).
- Current default `samples`: 512 (matches the Niazy proof-fit context
  dataset).
- Correction: the adapter computes `artifact = noisy - predicted_clean`
  per channel and lets `DeepLearningCorrection` subtract it.

## Architecture

| Block | Details |
|---|---|
| STFT front-end | `n_fft = 64`, `hop_length = 16`, `center = True`, Hann window. Per-channel signal of length `7 × samples` → magnitude + phase spectrograms. |
| Patchifier | Crop to `(freq_bins, time_frames) = (32, 224)`. Patches of `(4, 16)` → `(8, 14) = 112` patches per channel. |
| Token mixer | Linear patch embedding to `embed_dim = 192`, factorized 2-D learnable positional embedding, structural mask token covering center-epoch patches with one-patch margin. |
| Encoder | 6 pre-norm transformer blocks. Hand-rolled multi-head self-attention with `n_heads = 6` and `mlp_ratio = 4`. GELU activations, dropout 0.0. |
| Decoder head | Single `Linear(embed_dim → patch_freq × patch_time)`. |
| iSTFT back-end | Predicted log-magnitude → magnitude (via `expm1`) combined with the **input's original phase**, padded to the full STFT shape and inverted with `torch.istft`. The center-epoch slice is returned. |

Default parameter count: **~2.7 M**. The smoke configuration uses a
shallower (`depth=2`, `embed_dim=96`) variant for fast fleet validation.

## Compatibility Notes

- Compatible with different EEG channel counts because inference is
  channel-wise.
- Requires trigger metadata in the `ProcessingContext` at inference time.
- Native artifact-epoch lengths may vary across recordings; the adapter
  resamples each native trigger-to-trigger epoch to the model's
  `epoch_samples` and resamples the prediction back to the native length
  before subtraction.
- The deployed checkpoint is coupled to `context_epochs`, `epoch_samples`,
  and all STFT/patch parameters baked into the model — only one
  configuration of these settings is supported per checkpoint.

## Phase Handling

We reconstruct the time-domain signal from the **predicted clean
magnitude** and the **input's original (noisy) phase**. This matches the
existing `SpectrogramMixin` convention. The report explicitly flags this
choice as lossy at GA-dominated time-frequency bins where the noisy
phase is not equal to the clean phase. A complex/2-channel real+imag
variant is identified in `documentation/research_notes.md` as the most
promising follow-up if proof-fit metrics are weak.

## Current Training Reference

Full Niazy proof-fit run:

```bash
uv run facet-train fit \
  --config src/facet/models/vit_spectrogram/training_niazy_proof_fit.yaml
```

Smoke run (single epoch, capped dataset, smaller model):

```bash
uv run facet-train fit \
  --config src/facet/models/vit_spectrogram/training_niazy_proof_fit_smoke.yaml
```

Both jobs are submitted via `tools/gpu_fleet/fleet.py submit` and run on
the orchestrator-controlled GPU fleet.

## Evaluation Notes

Use the standard evaluation structure described in
`src/facet/models/evaluation_standard.md`. Comparison baselines:

- `cascaded_dae` — single-channel cascaded DAE, no trigger context.
- `cascaded_context_dae` — single-channel cascaded DAE with the same
  7-epoch trigger context. This is the closest baseline and the primary
  comparator for the proof-fit experiment.

Compare on synthetic proof-fit metrics (clean reconstruction error,
artifact prediction error, artifact correlation, residual RMS ratio) and
on visual inspection of the cleaned waveform vs. its AAS-corrected
surrogate.
