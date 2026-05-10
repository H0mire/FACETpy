# Cascaded Context DAE Model Card

## Summary

`cascaded_context_dae` is a seven-epoch channel-wise cascaded denoising autoencoder. It predicts the center-epoch artifact from neighboring trigger-defined artifact epochs. It uses a residual two-stage cascade: stage 1 predicts an initial center artifact, stage 2 receives the context with the center epoch corrected by stage 1 and predicts the remaining residual artifact.

## Intended Role

This model is the first context-aware DAE baseline after Demo 01. It keeps the channel-wise compatibility of `cascaded_dae` while adding temporal trigger context.

## Input And Output

- Input: `(batch, 7, 1, samples)`.
- Current default samples: `512`.
- Output: `(batch, 1, samples)` center-epoch artifact estimate.
- Correction: predicted artifact is subtracted by `DeepLearningCorrection`.

## Compatibility Notes

- Compatible with different EEG channel counts because inference is channel-wise.
- Requires trigger metadata.
- Native artifact lengths may vary; the processor resamples native epochs to the model-domain length and resamples predictions back.
- The checkpoint is coupled to context length and model-domain epoch length.

## Current Training Reference

The first 512-sample smoke model was trained with:

```bash
uv run facet-train fit --config src/facet/models/cascaded_context_dae/training.yaml
```

## Evaluation Notes

Use the standard evaluation structure described in `src/facet/models/evaluation_standard.md`.
