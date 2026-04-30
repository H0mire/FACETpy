# Coupling Between Training Context and Inference Context

## Central Question

A key design question for context-aware artifact correction is whether the model must be trained for exactly the same context configuration that will later be used during inference. In particular, the question arises whether the number of consecutive epochs provided to the model can remain flexible at inference time, or whether it must be fixed during training.

This issue is highly relevant for models that use multiple consecutive artifact epochs, neighboring temporal context, or trigger-phase-conditioned input.

## Short Answer

The model must be trained with assumptions that are compatible with the intended inference setup. However, this does not necessarily mean that the exact number of epochs must be fixed forever.

The decisive factor is the architecture.

- If the model expects a fixed input structure, then the number of epochs is fixed during both training and inference.
- If the model is designed to accept variable-length temporal context, then the inference context can in principle be varied later.

Nevertheless, technical feasibility does not automatically imply methodological validity. A model should not be evaluated on context configurations it has never seen during training unless this distribution shift is intentional and explicitly studied.

## Case 1: Fixed-Epoch Input Design

Some architectures encode the context structure explicitly, for example by assuming exactly three consecutive epochs:

- previous epoch,
- current epoch,
- next epoch.

In such a formulation, the model input may be defined as a tensor of fixed shape such as:

- `[3, C, T]`

where `3` denotes the number of epochs, `C` the number of channels, and `T` the epoch length.

In this case, the number of epochs is a hard architectural constraint. A model trained in this manner cannot simply be used with five or seven epochs at inference time without changing the model definition itself.

## Case 2: Variable-Length Context by Architecture

If the architecture is fully convolutional and operates on a continuous temporal input rather than a rigid epoch-indexed tensor, the situation changes.

For example, the model may receive:

- one long temporal context window,
- with prediction restricted to the center region only.

In this case, variable context lengths are technically possible as long as:

- the network contains no layers that require a fixed temporal dimension,
- and the receptive field remains appropriate for the target region.

Thus, a model trained on context-aware temporal chunks may in principle be applied to shorter or longer chunks during inference.

## Important Distinction: Technically Possible vs. Scientifically Sensible

Even if the architecture allows variable input length, changing the context size during inference is not automatically meaningful.

For example:

- training with 3 epochs,
- inference with 7 epochs.

This may work technically, but the model has not necessarily learned how to exploit the larger context in a stable and calibrated way. Depending on the training distribution, it may:

1. benefit from the additional context,
2. ignore the extra information,
3. or behave unpredictably because the temporal structure differs from what it has seen during training.

Therefore, the inference context should not be treated as a free knob unless the model has been prepared for this variability.

## Does It Make Sense to Keep the Number of Epochs Configurable?

Yes, but only under specific conditions.

A configurable inference context can be useful when:

- different datasets provide different amounts of usable neighboring context,
- recordings contain edge segments where fewer neighboring epochs are available,
- computational resources vary,
- or the effect of context size is itself part of the experimental investigation.

However, this flexibility is only meaningful if the model has been trained in a way that makes it robust to such variation.

## Recommended Strategy

The most principled solution is not to decouple training and inference completely, but to design them consistently around a controlled range of context lengths.

### Recommended Principle

The architecture should support variable temporal context, but the model should be trained on a distribution of plausible context sizes rather than on a single rigid length only.

For example, during training the model could receive:

- 3 epochs,
- 5 epochs,
- or 7 epochs,

with the target always restricted to the center epoch or center prediction region.

In this way, the model learns to operate under varying amounts of context and can later be deployed more flexibly.

## Why Center Prediction Helps

A context-aware model should ideally predict only the center region while using the surrounding signal as auxiliary information.

This provides two benefits.

1. It avoids boundary effects from incomplete context.
2. It allows the input context size to change without changing the semantic meaning of the target.

Thus, center prediction is an important prerequisite for making context variation scientifically defensible.

## Implications for the Planned Artifact Correction Model

For the intended EEG-fMRI correction problem, this leads to the following interpretation.

- It is not necessary to commit permanently to one exact number of epochs.
- However, the model must be trained in a way that is compatible with the range of context sizes expected during inference.
- A model trained on exactly one context size should not be assumed to generalize automatically to arbitrary larger or smaller context windows.

This is especially relevant because the available information may differ between datasets and because artifact timing may vary across acquisition settings.

## Experimental Consequence

The context length should be treated as a design variable and evaluated empirically.

A sensible ablation strategy would compare, for example:

- no additional context,
- 3-epoch context,
- 5-epoch context,
- 7-epoch context,
- with and without trigger-phase information.

This would allow a principled assessment of how much context is actually needed and whether the additional complexity is justified.

## Methodological Recommendation

The preferred design is therefore:

- a model architecture that supports variable temporal input length,
- center-region prediction,
- and training with a controlled range of context sizes.

This is preferable to either of the following extremes:

- a rigid model that can only ever process exactly one context length,
- or an apparently flexible model that is used on arbitrary inference contexts it has never seen during training.

## Thesis-Ready Core Statement

The context configuration used during training should be compatible with the intended inference scenario, but it does not need to be restricted to a single fixed number of epochs. If the architecture supports variable-length temporal input, the model can in principle be applied to different context sizes during inference. However, such flexibility is methodologically meaningful only if the model has already been trained on a representative range of context lengths. Consequently, context size should be treated as a controlled design variable rather than as an unconstrained inference-time parameter.
