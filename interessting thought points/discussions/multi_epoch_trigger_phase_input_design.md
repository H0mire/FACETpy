# Multi-Epoch Input with Trigger-Phase Conditioning

## Objective

The goal of this design is to reduce ambiguity between true neural transients and structured fMRI-related gradient artifacts by giving the model more information than a single isolated time window.

Instead of predicting from one short segment alone, the model receives a temporally extended input that covers multiple consecutive artifact epochs together with explicit information about the relative trigger phase.

This design is intended to improve:

- spike preservation,
- avoidance of spike hallucination,
- robustness to varying spike positions,
- and generalization across different artifact lengths and sequence timings.

## Motivation

A single short window is often insufficient to determine whether a sharp transient belongs to:

- a true epileptiform spike,
- a residual gradient artifact,
- or a random fluctuation.

Gradient artifacts are typically coupled to the MRI sequence and therefore exhibit temporal regularity. In contrast, spikes are not strictly bound to a fixed phase of the MRI cycle and may appear at arbitrary positions relative to the artifact structure.

Therefore, it is reasonable to assume that the model benefits from two additional forms of information:

1. temporal context across multiple consecutive artifact epochs,
2. explicit knowledge of the relative trigger position or phase.

## Core Idea

The model input consists of a sequence of consecutive windows rather than one isolated segment. In addition, each time point or epoch is associated with trigger-phase information.

Conceptually, the model receives:

- a local target epoch,
- one or more neighboring epochs before and after it,
- and a representation of where each sample lies relative to the MRI trigger cycle.

The output is then produced for the center epoch or center region only.

This is analogous to context-aware sequence modeling in which neighboring intervals provide disambiguating information but only the middle segment is supervised.

## Proposed Input Structure

### 1. Temporal Input

A practical design is to use three consecutive epochs:

- previous epoch,
- current epoch,
- next epoch.

These can either be concatenated along the time axis or encoded as separate sequence elements.

Two implementation variants are possible.

#### Variant A: Concatenated Temporal Context

The three epochs are concatenated into one longer input tensor:

- input: `[channels, 3 * epoch_length]`
- output target: center epoch only

This is simple and works well with fully convolutional temporal models.

#### Variant B: Explicit Epoch Sequence

Each epoch is kept as a distinct sequence element:

- input: `[n_epochs, channels, epoch_length]`
- example: `[3, C, T]`
- output target: prediction for the middle epoch

This is better suited for architectures that model relations between epochs explicitly, such as temporal attention or sequence encoders.

## Trigger-Phase Representation

The second key component is the trigger-related conditioning signal.

The model should know where each sample or epoch lies relative to the MRI acquisition cycle. This can be represented in several ways.

### 1. Relative Sample Position to Nearest Trigger

For each sample, compute the signed distance to the nearest trigger or to the start of the current artifact epoch.

This yields a temporal feature such as:

- `delta_to_trigger_samples`
- or normalized `delta_to_trigger_seconds`

### 2. Normalized Phase Within the Artifact Cycle

Each sample receives a normalized phase value in the range `[0, 1]`, where:

- `0` corresponds to the beginning of the artifact cycle,
- `1` corresponds to the end.

This is especially attractive if artifact duration varies between datasets, because the model sees relative rather than absolute timing.

### 3. Sinusoidal Phase Encoding

The trigger phase can also be represented through sinusoidal features:

- `sin(phase)`
- `cos(phase)`

This avoids discontinuities at the phase boundary and is common in periodic signal modeling.

## Recommended Representation

A robust formulation is:

- signal input: noisy EEG epochs,
- auxiliary input: normalized trigger phase per sample,
- prediction target: artifact estimate for the center epoch.

Thus, the model receives both the observed waveform and explicit timing context tied to the MRI sequence.

## Output Strategy

The preferred training target remains artifact prediction:

- input: noisy multi-epoch context + phase features,
- output: artifact estimate for the center epoch,
- reconstruction: `clean_hat = noisy_center - artifact_hat`

This formulation is preferable to direct clean-signal synthesis because it reduces the risk that the model invents plausible-looking EEG activity not supported by the input.

## Why Center Prediction Is Important

If the model receives a larger context window, it should not necessarily be supervised on the full input range. A cleaner strategy is to predict only the center epoch.

This has three advantages:

1. the model can use left and right context without boundary artifacts,
2. the target does not depend on incomplete context at the input borders,
3. the training objective remains aligned with the intended local correction task.

## Expected Advantages

Compared with a single isolated window, the multi-epoch plus trigger-phase design should provide the following benefits.

### Better Distinction Between Artifact and Neural Events

Because gradient artifacts repeat across consecutive epochs, the model can use neighboring structure to identify what is systematic artifact and what is locally unique.

### Reduced Positional Shortcut Learning

Because the model is not restricted to one fixed spike position and instead sees varying context and explicit trigger phase, it is less likely to learn a trivial rule such as “preserve sharp activity at the center of the window.”

### Improved Generalization Across Sequence Timing

If the phase is represented in normalized form, the model can become less dependent on one specific artifact duration in raw sample counts.

### Better Support for Variable-Length Inference

A fully convolutional implementation can process longer chunks during inference while still benefiting from the same context principle used in training.

## Expected Limitations

This design is stronger than a single-window baseline, but it also introduces complexity.

### More Complex Dataset Construction

Training examples must now include consecutive epochs and correctly aligned phase metadata.

### Dependence on Reliable Trigger Information

If trigger extraction is noisy or inconsistent, the phase-conditioning signal may become unreliable.

### Higher Memory and Compute Demand

A multi-epoch input contains more samples than a single-window baseline and therefore increases training cost.

### Boundary Effects in Fixed-Window Models

The first practical seven-epoch experiments showed that boundary behavior must be explicitly monitored. A convolutional model with zero-padding produced large artificial edge amplitudes at the beginning and end of the predicted artifact epoch. This is particularly problematic because the predicted artifact is subtracted from the EEG; a boundary error therefore becomes a correction artifact.

For fixed-window context models, edge-safe design is required:

- avoid zero-padding where possible,
- prefer reflection padding or valid center-region prediction,
- avoid circular augmentation such as `np.roll` for artifact jitter,
- and include edge diagnostics in the evaluation.

This observation strengthens the argument for center-epoch prediction: the model should use context to support the target prediction, but boundary samples should not dominate the learned correction behavior.

## Suggested Data Representation for Implementation

A practical implementation could use the following structure per example:

- `signal_context`: shape `[3, 1, T]` for single-channel training or `[3, C, T]` for multichannel training
- `phase_context`: shape `[3, 1, T]` or `[3, P, T]` where `P` is the number of phase features
- `target_artifact`: shape `[1, T]` or `[C, T]` for the center epoch only

If signal and phase are fused directly, the model input channels could become:

- noisy EEG,
- normalized phase,
- optional `sin(phase)`,
- optional `cos(phase)`

For a single-channel model, this could be represented as:

- input: `[feature_channels, 3 * T]`
- target: `[1, T]`

## Suitable Model Classes

This input design is particularly compatible with:

- fully convolutional 1D models,
- temporal U-Nets,
- temporal convolutional networks,
- lightweight attention over consecutive epochs,
- or hybrid CNN + sequence encoder models.

As a first practical baseline, a fully convolutional artifact-prediction model with center prediction is the most defensible choice.

## Research Implication

The multi-epoch plus trigger-phase approach operationalizes a key domain assumption: fMRI gradient artifacts are temporally structured and sequence-coupled, whereas epileptic spikes are transient and not strictly phase-locked to the MRI trigger cycle.

This makes the design scientifically attractive because it encodes known structure of the artifact generation mechanism rather than expecting the model to infer this structure from isolated short windows alone.

## Thesis-Ready Core Statement

A context-aware model input consisting of multiple consecutive epochs together with explicit trigger-phase information provides a principled way to separate temporally structured gradient artifacts from transient epileptiform activity. By predicting only the center epoch while using neighboring epochs as context, the model can exploit the repetitive nature of MRI-induced artifacts without relying on fixed positional assumptions about spike occurrence. This design is therefore expected to improve spike preservation, reduce hallucination risk, and support more robust generalization across varying artifact durations and acquisition settings.

Additional experimental note: early seven-epoch model tests showed that the success of this design depends on edge-safe implementation details. Reflection padding, non-circular temporal jitter, and local mean normalization were necessary to avoid artificial boundary and offset artifacts.
