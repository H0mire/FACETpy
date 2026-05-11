# Context Artifact Model Evaluation

## Scope

This report compares the same TorchScript context-artifact model on two inputs:

- **Synthetic training-style dataset:** supervised evaluation with known clean center epoch and artifact target.
- **Niazy EEG-fMRI recording:** unsupervised/proxy evaluation because no true clean EEG reference is available.

Checkpoint: `training_output/cascadedcontextdenoisingautoencoder_20260502_115926/exports/cascaded_context_dae.ts`

## Synthetic Metrics

| Metric | Value |
| --- | ---: |
| examples | 1860 |
| clean MSE before | 3.034173e-06 |
| clean MSE after | 1.466207e-06 |
| clean MSE reduction | 51.68 % |
| clean SNR before | -40.393 dB |
| clean SNR after | -37.235 dB |
| clean SNR improvement | 3.158 dB |
| artifact MAE | 5.216702e-04 |
| artifact correlation | 0.7318 |
| residual RMS ratio | 0.6951 |
| input mean removed | True |
| prediction mean removed | True |
| predicted artifact edge mean abs | 199.45 uV |
| predicted artifact center mean abs | 577.18 uV |
| edge/center abs ratio | 0.35 |

## Niazy Proxy Metrics

| Metric | Value |
| --- | ---: |
| channels | 31 |
| triggers | 840 |
| corrected center epochs | 833 |
| native epoch length min/median/max | 292 / 292.0 / 303 samples |
| trigger-locked RMS before | 8.441970e-03 |
| trigger-locked RMS after | 8.387580e-03 |
| trigger-locked RMS reduction | 0.64 % |
| median-template RMS before | 8.347880e-03 |
| median-template RMS after | 8.291841e-03 |
| median-template RMS reduction | 0.67 % |
| median-template peak-to-peak reduction | 3.91 % |
| predicted artifact RMS | 5.432844e-04 |

## Interpretation

The current checkpoint is a minimal demonstration model, not a validated artifact corrector. In this run it worsens
the supervised synthetic clean reconstruction and does not reduce the Niazy trigger-locked proxy metrics. This points
to insufficient training/model capacity and/or a mismatch between the synthetic target distribution and Niazy inference
distribution.

The model also shows a systematic boundary artifact: the predicted artifact amplitude is much larger at the first and
last samples than in the center of the epoch. Boundary-sensitive metrics and plots should therefore be interpreted
with care until the model is retrained with edge-safe augmentation and padding.

The Niazy metrics must not be interpreted as clean-ground-truth metrics. They only quantify whether trigger-locked
structure becomes smaller after correction.

## Plots

- `synthetic_cleaning_examples.png`
- `synthetic_metric_summary.png`
- `niazy_cleaning_examples.png`
- `niazy_trigger_locked_templates.png`
