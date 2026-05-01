# Demo Model 01: Seven-Epoch Context CNN

## Status

This model is marked as the first demonstration model. It is useful as a proof of concept for the FACETpy deep-learning pipeline, but it should not be optimized further in its current form.

The model demonstrates that a deep-learning correction component can be trained from synthetic EEG/artifact pairs, exported, and executed inside the FACETpy pipeline. Its main value is therefore architectural and methodological rather than final correction quality.

## Motivation

The first deep-learning prototype was designed to test whether fMRI gradient-artifact correction can be formulated as supervised artifact prediction. Instead of predicting a cleaned EEG signal directly, the model predicts the artifact component of the center epoch. The predicted artifact is then subtracted from the noisy EEG signal by the pipeline.

This formulation was chosen because it preserves the classical correction logic used in artifact-removal pipelines:

```text
corrected EEG = noisy EEG - predicted artifact
```

It also makes the model easier to evaluate, because the prediction can be inspected as an estimated artifact waveform.

## Input And Output Design

The model uses a fixed seven-epoch context window.

```text
input  = seven consecutive trigger-aligned epochs
output = artifact estimate for the center epoch
```

In the current implementation, correction is performed per channel. This means that the model sees one EEG channel at a time, but receives neighboring artifact epochs as temporal context. The input tensor has the conceptual shape:

```text
context_epochs x channels x samples = 7 x 1 x 292
```

The center-only prediction avoids forcing the model to make correction decisions at the boundaries of the context window. This is important because edge epochs have less surrounding information and are more vulnerable to boundary artifacts.

## Architecture

The model is a small fully convolutional 1D network:

- model name: `SevenEpochContextArtifactNet`
- input layout: seven epochs concatenated over the channel dimension
- convolution type: `Conv1d`
- hidden channels: 32
- kernel size: 9
- padding: reflection padding
- activation: GELU
- output: one predicted artifact waveform for the center epoch

The architecture is intentionally simple. It was not intended to be the final model class, but a minimal model that can validate the complete training, export, and inference path.

## Training Dataset

The training dataset consists of synthetic clean EEG segments with added fMRI artifact estimates. The clean source includes generated spike-like EEG events, while the artifact component is sampled from extracted artifact libraries.

The current dataset contains:

- 1860 training examples,
- 7 context epochs per example,
- 292 samples per epoch,
- synthetic spike EEG as clean source,
- Niazy AAS-derived gradient-artifact estimates,
- and Large-MFF AAS-derived gradient-artifact estimates.

The artifact sources intentionally differ in sampling frequency, channel count, waveform morphology, and amplitude range. Different scanner systems, acquisition sequences, EEG caps, amplifiers, and preprocessing settings naturally produce different gradient-artifact amplitudes. Therefore, amplitude differences between artifact sources should not automatically be treated as errors. They are a realistic part of the domain variation that a later model must handle.

## Training Configuration

The first demo model was trained with the following approximate setup:

- framework: PyTorch via the FACETpy training adapter,
- training entry point: `facet-train fit`,
- configuration file: `src/facet/models/demo01/training.yaml`,
- optimizer: Adam,
- learning rate: 1e-3,
- weight decay: 1e-4,
- gradient clipping: 1.0,
- batch size: 16,
- epochs: 10,
- loss: L1,
- validation split: 20%,
- export format: TorchScript.

Both input and target are demeaned per epoch in the dataset loader. During inference, the predicted artifact mean can also be removed before subtraction.

## Pipeline Integration

The model is integrated through the context-aware deep-learning correction processor. The relevant processor is:

```text
EpochContextDeepLearningCorrection
```

This processor supports trigger-aligned context construction, center-epoch prediction, artifact subtraction, and memory-conscious per-channel execution. The per-channel execution path is important because EEG-fMRI recordings can contain many channels and long recordings; holding all channels and all context windows in memory at once is not always practical.

The implementation therefore preserves the idea that deep-learning correction should work inside the existing FACETpy execution model rather than requiring a completely separate correction workflow.

## Evaluation Summary

A comparison report was generated against an older cascaded denoising-autoencoder baseline. The older baseline performs better on the Niazy dataset in absolute artifact-suppression metrics, but it is strongly tied to the same Niazy distribution used during training. The current context-CNN model is methodologically cleaner and better integrated into the pipeline, but its absolute correction quality is not yet sufficient to treat it as a final model.

Important observations:

- the current model validates the full training and inference infrastructure,
- the current model supports multi-source artifact training,
- the current model preserves spikes better conceptually because it predicts the artifact rather than directly hallucinating a clean EEG signal,
- the current model still leaves visible residual gradient artifacts in some evaluations,
- the SNR metric can be misleading and should not be used as the primary quality criterion,
- median trigger-locked artifact residuals, RMS ratios, and qualitative epoch plots are more informative for this stage.

## Methodological Interpretation

The first demo model should be interpreted as a feasibility prototype. It answers the engineering question of whether a modern deep-learning correction path can be embedded into FACETpy. It does not yet answer the scientific question of which architecture generalizes best across scanners, acquisition protocols, and EEG montages.

The model also makes clear that random train/validation splits are not enough for this problem. Since fMRI artifacts are scanner- and protocol-dependent, future evaluation should use source-aware validation, for example leave-one-artifact-source-out evaluation.

## Why This Model Is Frozen

This model should not be further optimized because its main design choices were intentionally simple:

- fixed seven-epoch context,
- single-stage artifact prediction,
- small CNN architecture,
- L1-only loss,
- limited number of real artifact sources,
- no explicit scanner/protocol conditioning,
- no residual refinement stage,
- no source-aware validation protocol.

Further improvements should be implemented as a new model generation rather than incremental patches to this prototype. This keeps the thesis narrative clean: Demo Model 01 establishes feasibility, while later models can address specific scientific and architectural weaknesses.

## Recommended Next Model Direction

The next model should keep the useful parts of Demo Model 01:

- artifact prediction instead of direct clean-signal prediction,
- center-region prediction,
- pipeline-compatible inference,
- per-channel execution support,
- multi-source artifact libraries,
- and explicit source metadata.

It should improve the weak points:

- use source-aware train/validation/test splits,
- evaluate leave-one-source-out generalization,
- consider a combined loss such as L1 plus MSE or a peak-weighted artifact loss,
- consider a residual refinement stage,
- increase artifact-source diversity,
- document scanner/protocol metadata where available,
- and compare correction quality using robust trigger-locked artifact metrics.

## Thesis-Ready Summary

The first demonstration model implemented a seven-epoch context convolutional neural network for supervised fMRI gradient-artifact prediction. The model received neighboring trigger-aligned epochs as context and predicted only the artifact waveform of the center epoch, which was subsequently subtracted from the noisy EEG signal. This design preserved the classical artifact-correction formulation while enabling integration into the FACETpy processing pipeline, including per-channel execution for memory-efficient inference. Training used synthetic spike-containing EEG combined with artifact estimates extracted from multiple EEG-fMRI recordings. Since gradient-artifact morphology and amplitude vary naturally across scanner systems, protocols, montages, and preprocessing pipelines, these source differences were treated as relevant domain variability rather than as simple scaling errors. The model demonstrated the feasibility of a pipeline-integrated deep-learning correction workflow, but its correction quality and validation protocol were not sufficient for a final generalizable model. It is therefore treated as a frozen proof-of-concept model and used as the methodological baseline for subsequent model generations.
