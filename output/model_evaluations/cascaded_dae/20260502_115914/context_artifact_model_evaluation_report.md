# Context Artifact Model Evaluation

## Scope

This report compares the same TorchScript context-artifact model on two inputs:

- **Synthetic training-style dataset:** supervised evaluation with known clean center epoch and artifact target.
- **Niazy EEG-fMRI recording:** unsupervised/proxy evaluation because no true clean EEG reference is available.

Checkpoint: `training_output/cascadeddenoisingautoencoder_20260502_115914/exports/cascaded_dae.ts`

## Synthetic Metrics

| Metric | Value |
| --- | ---: |
| examples | 1860 |
| clean MSE before | 3.034173e-06 |
| clean MSE after | 3.066671e-06 |
| clean MSE reduction | -1.07 % |
| clean SNR before | -40.393 dB |
| clean SNR after | -40.440 dB |
| clean SNR improvement | -0.046 dB |
| artifact MAE | 9.221299e-04 |
| artifact correlation | 0.0862 |
| residual RMS ratio | 1.0053 |
| input mean removed | True |
| prediction mean removed | True |
| predicted artifact edge mean abs | 278.74 uV |
| predicted artifact center mean abs | 300.06 uV |
| edge/center abs ratio | 0.93 |

## Niazy Proxy Metrics

| Metric | Value |
| --- | ---: |
| channels | 31 |
| triggers | 840 |
| corrected center epochs | 833 |
| native epoch length min/median/max | 292 / 292.0 / 303 samples |
| trigger-locked RMS before | 8.441970e-03 |
| trigger-locked RMS after | 8.443310e-03 |
| trigger-locked RMS reduction | -0.02 % |
| median-template RMS before | 8.347880e-03 |
| median-template RMS after | 8.334643e-03 |
| median-template RMS reduction | 0.16 % |
| median-template peak-to-peak reduction | -0.03 % |
| predicted artifact RMS | 4.620621e-04 |

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
