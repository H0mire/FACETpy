# Demo 01: Seven-Epoch Context CNN

Demo 01 is the first closed-beta proof-of-concept for FACETpy deep-learning artifact correction.

It is intentionally model-specific and frozen. New models should not extend this package directly unless they are explicitly Demo-01 variants.

## Contents

- `processor.py`: Demo-01 TorchScript adapter plus a compatibility processor that subclasses the generic `DeepLearningCorrection`.
- `training.py`: `facet-train` factories for the seven-epoch context CNN.
- `training.yaml`: reference training configuration for the frozen Demo 01 setup.

## Model Assumptions

- Trigger-defined artifact epochs.
- Odd-numbered multi-epoch context.
- Default context length of seven epochs.
- Single-channel inference.
- Fixed model-domain epoch length.
- Variable native trigger deltas handled by resampling.
- Center-epoch artifact prediction.
- Artifact subtraction from noisy EEG.
- TorchScript inference.

These assumptions belong to Demo 01 and should not be treated as global FACETpy requirements.

## Integration Shape

Demo 01 now follows the model-folder guideline:

```text
Demo01EpochContextTorchScriptAdapter
  builds the model-specific trigger/epoch context
  returns DeepLearningPrediction(artifact_data=...)

EpochContextDeepLearningCorrection
  inherits from DeepLearningCorrection
  keeps the old processor name for closed-beta compatibility
  delegates correction application to the generic DL correction path
```

New models should prefer the same pattern: keep model-specific input construction in an adapter and let `DeepLearningCorrection` apply `artifact`, `clean`, or `both` predictions consistently.
