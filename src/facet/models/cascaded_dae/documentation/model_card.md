# Cascaded DAE Model Card

## Summary

`cascaded_dae` is a channel-wise cascaded denoising autoencoder that predicts artifact windows from noisy single-channel windows. It uses a residual two-stage cascade: stage 1 predicts an initial artifact estimate, stage 2 receives the stage-1-corrected signal and predicts the remaining residual artifact.

## Intended Role

This model is the non-context autoencoder baseline. It tests whether a simple per-window DAE can predict fMRI artifact structure without using neighboring trigger epochs.

## Input And Output

- Input: `(batch, 1, samples)`.
- Current default samples: `512`.
- Output: `(batch, 1, samples)` artifact estimate.
- Correction: predicted artifact is subtracted by `DeepLearningCorrection`.

## Compatibility Notes

- Compatible with different EEG channel counts because inference is channel-wise.
- Fixed model-domain window length.
- No explicit multi-epoch context.

## Evaluation Notes

Use the standard evaluation structure described in `src/facet/models/evaluation_standard.md`.
