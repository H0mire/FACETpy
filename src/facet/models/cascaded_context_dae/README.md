# Cascaded Context DAE

This model is the 7-epoch context variant of `cascaded_dae`.

## Scope

- Input shape: `(batch, 7, 1, samples)`.
- Output shape: `(batch, 1, samples)`.
- Current training default: `samples=512`.
- Predicts the center-epoch artifact signal.
- Uses a residual two-stage cascade: stage 1 estimates the center artifact,
  stage 2 sees the context after center-epoch stage-1 subtraction and predicts
  the residual artifact.
- Applies per channel, so inference remains compatible with different channel counts.
- Requires trigger metadata during inference, because it reconstructs neighboring artifact epochs from the `ProcessingContext`.

## Training

Build the default 7-epoch, 512-sample context dataset first:

```bash
uv run python examples/build_epoch_context_dataset.py \
  --clean-source output/synthetic_spike_source/synthetic_spike_source_raw.fif::spike_onset \
  --artifact-library output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz \
  --artifact-library output/artifact_libraries/large_mff_aas/large_mff_aas_artifact.npz \
  --target-epoch-samples 512 \
  --context-epochs 7 \
  --output-dir output/synthetic_spike_artifact_context_512
```

Then train and export the TorchScript checkpoint:

```bash
uv run facet-train fit --config src/facet/models/cascaded_context_dae/training.yaml
```

The current smoke-trained checkpoint was produced from this configuration with:

- input shape: `(batch, 7, 1, 512)`
- output shape: `(batch, 1, 512)`
- best epoch: 18
- best validation loss: approximately `2e-6`

## Inference

```python
from facet.models.cascaded_context_dae import CascadedContextDenoisingAutoencoderCorrection

context = context | CascadedContextDenoisingAutoencoderCorrection(
    checkpoint_path="training_output/<run>/exports/cascaded_context_dae.ts",
    context_epochs=7,
    epoch_samples=512,
)
```

## Architectural Decision

The model keeps the channel-wise compatibility of `cascaded_dae`, but restores the temporal context used by the earlier 7-epoch experiments. This makes the checkpoint independent of channel count but still coupled to the context length and model input epoch length.

Native artifact epochs may have different lengths at inference time. The processor resamples each trigger-to-trigger epoch to the model length before prediction and resamples the predicted center artifact back to the native epoch length before subtraction.
