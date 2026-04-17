# Training CLI Architecture and Role of Adapters

This note captures the planned architecture of a future `facet-train` command-line interface and the role of wrappers and inference adapters in the FACETpy deep-learning stack.

It is written so that parts of it can be reused later in technical documentation or in a thesis.

## Overview

The planned `facet-train` CLI provides a standardized training entry point for deep-learning models in FACETpy.

Its purpose is not to replace framework-specific model development, but to standardize:

- data preparation
- training orchestration
- logging
- checkpoint handling
- model export
- later reuse in the FACETpy correction pipeline

The central architectural principle is the separation between:

1. training wrappers
2. inference adapters

This separation allows the training workflow to remain framework-agnostic at the orchestration level, while still supporting framework-specific execution in both training and inference.

## Architectural Layers

The overall design can be described in four layers:

1. Data and chunking layer
2. Training orchestration layer
3. Framework-specific training integration
4. Pipeline-side inference integration

In practical terms, the data flow is:

```text
EEG files / ProcessingContext
    -> EEGArtifactDataset
    -> Trainer / facet-train CLI
    -> TrainableModelWrapper
    -> trained checkpoint / exported model
    -> Inference Adapter
    -> DeepLearningCorrection
    -> FACETpy pipeline
```

## Role of Training Wrappers

Training wrappers are the training-side adapters of the system.

They encapsulate all framework-specific functionality required during model optimization. This includes in particular:

- conversion from NumPy arrays to framework tensors
- execution of `train_step` and `eval_step`
- backward pass and optimizer step
- optional scheduler handling
- checkpoint saving and loading

In the current FACETpy implementation, this role is fulfilled by:

- `PyTorchModelWrapper`
- `TensorFlowModelWrapper`

Both are implemented in [wrapper.py](/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/src/facet/training/wrapper.py).

As a result, the generic trainer does not need to know how PyTorch or TensorFlow perform optimization internally. It only interacts with a unified wrapper interface.

## Role of Inference Adapters

Inference adapters are the pipeline-side execution adapters.

Their purpose is to make an exported model artifact usable inside the FACETpy correction pipeline. They encapsulate:

- checkpoint or exported-model loading
- runtime and device selection
- input tensor layout mapping
- model execution
- output mapping into a FACETpy-compatible prediction format

In the current FACETpy implementation, this role is fulfilled by:

- `PyTorchInferenceAdapter`
- `TensorFlowInferenceAdapter`

These are implemented in [deep_learning.py](/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/src/facet/correction/deep_learning.py).

They are used by `DeepLearningCorrection`, which integrates the model into the processor pipeline.

## Why Wrappers and Adapters Are Separated

The separation between wrappers and adapters is deliberate and technically justified.

Training and inference have different responsibilities:

- Training requires gradient computation, optimizer state, loss functions, scheduler logic, and callback handling.
- Inference requires deterministic forward execution, checkpoint loading, input/output layout handling, and integration into FACETpy execution modes such as chunking or channel-wise processing.

Combining both concerns into a single abstraction would couple training and production inference too tightly. The current separation improves:

- maintainability
- testability
- extensibility
- reuse of exported models

In short:

> Wrappers train models, adapters operate models in the FACETpy pipeline.

## Planned Role of `facet-train`

The `facet-train` CLI is intended as an orchestration layer around the existing training components.

Its main responsibilities are:

- loading a training configuration from YAML
- constructing datasets from FACETpy contexts
- dynamically importing a user-defined model factory
- selecting the appropriate framework wrapper
- running the generic trainer
- writing logs, checkpoints, and a resolved configuration
- exporting the trained model in an inference-compatible format
- storing enough metadata so the model can later be used by `DeepLearningCorrection`

The intended command shape is:

```bash
facet-train fit --config train.yaml
```

## Planned Data Flow with `facet-train`

The workflow can be described as follows:

1. A YAML configuration is loaded.
2. Training data are transformed into an `EEGArtifactDataset`.
3. A user-defined model is instantiated through a factory function.
4. A framework-specific wrapper is selected.
5. The generic `Trainer` performs optimization, logging, checkpointing, and early stopping.
6. A model artifact is exported, for example TorchScript or SavedModel.
7. Inference metadata are written alongside the export.
8. Later, a matching inference adapter loads the exported artifact and injects it into `DeepLearningCorrection`.

This yields the following end-to-end flow:

```text
train.yaml
    -> facet-train fit
    -> data loading
    -> EEGArtifactDataset
    -> user model factory
    -> training wrapper
    -> Trainer
    -> checkpoints + logs + resolved config
    -> exported model artifact
    -> inference metadata
    -> DeepLearningCorrection
```

## Practical Benefit for Developers and Researchers

For a developer or researcher, the intended workflow is reduced to three main tasks:

1. implement a model
2. write a training configuration
3. start training via CLI

The user should not have to reimplement:

- the training loop
- checkpoint logic
- live logging
- JSONL metric persistence
- export bookkeeping
- pipeline integration code

This reduces setup complexity and improves reproducibility across experiments.

## Scientific and Engineering Advantages

The proposed design offers several advantages:

- Reproducibility: each run is described by a configuration file, a run directory, and persistent logs.
- Framework independence at orchestration level: the same training workflow can be reused for PyTorch or TensorFlow.
- Clear separation of concerns: dataset handling, training logic, framework execution, and pipeline inference remain modular.
- Direct integration into FACETpy: exported models can be reused without ad hoc glue code.
- Extensibility: additional runtimes or export targets can be added through new wrappers or adapters.

## Limitations

The architecture also has explicit limits:

- A training checkpoint is not automatically pipeline-ready. Only an export that is supported by a matching inference adapter can be integrated into FACETpy.
- Framework agnosticism applies to orchestration and interfaces, not to the model implementation itself.
- Models with special side inputs, such as trigger embeddings or graph-based montage representations, may require specialized inference adapters.

## Explicit TODOs

The following items are still intentionally open and should be treated as explicit next steps for the CLI:

- Add a dedicated `facet-train validate-config --config ...` subcommand.
  This command should validate the training configuration, resolve dynamic factories, check training-to-inference consistency, and fail early without starting a training run.

## Thesis-Ready Paragraph

The following paragraph is intentionally phrased so that it can be reused in a thesis with minimal editing:

> To standardize the training workflow of deep-learning-based artifact correction models, a CLI-based training layer (`facet-train`) is planned for FACETpy. The architecture deliberately separates training wrappers from inference adapters. Training wrappers encapsulate framework-specific optimization logic, including tensor conversion, training and evaluation steps, checkpoint handling, and optional scheduler control, while the trainer itself remains framework-agnostic. Inference adapters, in contrast, are responsible for integrating exported model artifacts into the FACETpy pipeline, including checkpoint loading, device selection, tensor layout mapping, and conversion of model outputs into a FACETpy-compatible prediction format. This separation creates a clean boundary between model training and productive pipeline execution and thereby improves maintainability, extensibility, and reproducibility.

## Short Core Statement

If a shorter formulation is needed:

> Wrappers are responsible for training-time framework integration, whereas adapters are responsible for pipeline-time model integration.

## Suggested Figure Caption

If this architecture is later visualized in a figure, the following caption can be used:

> Planned architecture of the `facet-train` CLI, showing the separation between data preparation, framework-specific training integration, model export, and inference-side integration into the FACETpy pipeline.
