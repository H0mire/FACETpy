# Cascaded DAE: Channel-Wise Cascaded Denoising Autoencoder

Cascaded DAE ports the denoising-autoencoder idea from the older `feature/deeplearning`
prototype into the current model-package architecture.

## Scope

- Predicts the artifact signal, not the clean EEG signal.
- FACETpy applies correction by subtracting the predicted artifact via
  `DeepLearningCorrection`.
- Uses a two-stage residual cascade of fully connected denoising autoencoders.
- Stage 1 predicts an initial artifact estimate; stage 2 receives the signal
  after subtracting stage 1 and predicts the remaining residual artifact.
- Trains and infers per channel with input shape `(batch, 1, samples)`.
- Current training default: `samples=512`.
- Supports arbitrary channel counts at inference because the same model is
  applied independently to every selected channel.

## Files

- `training.py`: PyTorch model classes and `facet-train` factories.
- `processor.py`: TorchScript inference adapter and pipeline processor.
- `training.yaml`: default CLI configuration.

## Training

```bash
uv run facet-train fit --config src/facet/models/cascaded_dae/training.yaml
```

The default dataset factory expects the Demo01 synthetic context `.npz` and uses
the center noisy/artifact epoch as per-channel training pairs.

## Inference

```python
from facet.models.cascaded_dae import CascadedDenoisingAutoencoderCorrection

context = context | CascadedDenoisingAutoencoderCorrection(
    checkpoint_path="training_output/<run>/exports/cascaded_dae.ts",
    chunk_size_samples=512,
    chunk_overlap_samples=0,
)
```

## Architectural Decision

The old branch trained a fully connected model on `channels * samples`, which
binds the checkpoint to one channel count and one window size. Cascaded DAE keeps the
window size fixed but removes the channel-count dependency by training on
single-channel windows. This preserves the denoising-autoencoder concept while
matching the current FACETpy requirement that models remain usable across EEG
datasets with different channel montages.
