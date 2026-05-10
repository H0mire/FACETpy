# Demo 01 Model Card

## Summary

Demo 01 is the frozen seven-epoch context CNN proof-of-concept for deep-learning artifact correction.

## Intended Role

This model documents the first implementation path and should be treated as a demonstration baseline, not as the preferred architecture for future work.

## Input And Output

- Input: trigger-defined multi-epoch context.
- Default context: seven epochs.
- Output: center-epoch artifact estimate.
- Correction: predicted artifact is subtracted from noisy EEG through `DeepLearningCorrection`.

## Compatibility Notes

- Channel-wise inference keeps channel count flexible.
- Native trigger-delta variability is handled by resampling to the model-domain epoch length and back.
- The checkpoint remains coupled to the configured context length and model-domain epoch length.

## Evaluation Notes

Use the standard evaluation structure described in `src/facet/models/evaluation_standard.md`.
