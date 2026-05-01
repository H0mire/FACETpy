# Demo 01: Seven-Epoch Context CNN

Demo 01 is the first closed-beta proof-of-concept for FACETpy deep-learning artifact correction.

It is intentionally model-specific and frozen. New models should not extend this package directly unless they are explicitly Demo-01 variants.

## Contents

- `processor.py`: specialized TorchScript epoch-context correction processor used by Demo 01.
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
