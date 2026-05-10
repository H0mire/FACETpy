# FACETpy Model-Specific Implementations

This directory contains model-specific deep-learning implementations. It is intentionally separate from `facet.correction` because experimental model code should not silently become part of the core correction API.

## Guiding Principle

FACETpy core code should define stable processing contracts. Model directories should contain assumptions that belong to one concrete model family or experiment.

Every addition to `src/facet` can affect compatibility, pipeline behavior, import stability, documentation, and future model flexibility. Therefore, code should only move into core when it is demonstrably reusable across model families.

## What Belongs In FACETpy Core

Core implementations belong in `src/facet/correction`, `src/facet/training`, or other stable package modules only when they are model-independent.

Examples of core-level responsibilities:

- `ProcessingContext` handling.
- Pipeline `Processor` lifecycle and execution modes.
- Generic deep-learning output semantics: `artifact`, `clean`, `both`.
- Generic prediction application: subtract artifact, apply clean prediction, store `estimated_noise`.
- Runtime adapter contracts for PyTorch, TensorFlow, ONNX, NumPy, or custom backends.
- Generic validation of runtime availability, checkpoints, channel execution mode, and metadata prerequisites.
- Generic training orchestration, logging, checkpointing, and config loading.
- Reusable dataset interfaces that do not encode one model architecture.

A core addition should be justified by at least one of the following:

- Multiple model families need the same behavior.
- The behavior is required for pipeline correctness or execution-mode compatibility.
- The behavior defines a stable public contract rather than an experimental choice.
- The behavior prevents duplicated correction semantics across models.

## What Belongs In A Model Folder

Model-specific implementations must live in a dedicated subdirectory under `src/facet/models/<model_id>/`.

Examples of model-specific responsibilities:

- Neural network architecture definitions.
- Model-specific context construction.
- Model-specific preprocessing or postprocessing assumptions.
- Fixed input tensor conventions such as `7 x 1 x 292`.
- Center-epoch prediction logic.
- Specific loss choices when they are part of the model recipe.
- Training factory functions for `facet-train`.
- Model-specific YAML configs.
- Model cards and experiment notes.
- Model-specific evaluation or application scripts when they are not reusable.

A model directory may depend on FACETpy core APIs, but FACETpy core should not depend on a model directory unless a temporary compatibility wrapper is explicitly justified.

## Recommended Model Directory Layout

```text
src/facet/models/<model_id>/
├── __init__.py
├── README.md
├── documentation/
│   ├── model_card.md
│   └── evaluations.md
├── model.py              # architecture if separated from training factories
├── training.py           # facet-train factories
├── processor.py          # only if a model-specific Processor is unavoidable
├── adapter.py            # preferred place for model-specific inference logic
├── training.yaml
├── inference.yaml
└── model_card.md
```

Not every model needs every file. Small models may combine `model.py` and `training.py`.

## Evaluation Documentation

Every model should follow `src/facet/models/evaluation_standard.md`.

Short, versionable documentation belongs in:

```text
src/facet/models/<model_id>/documentation/
```

Generated evaluation artifacts belong in:

```text
output/model_evaluations/<model_id>/<run_id>/
```

New evaluation scripts should use `facet.evaluation.ModelEvaluationWriter` so every run emits:

- `evaluation_manifest.json`
- `metrics.json`
- `evaluation_summary.md`

## Preferred Integration Path

The preferred path for new models is:

```text
DeepLearningCorrection + model-specific adapter
```

The adapter receives a `ProcessingContext`, builds whatever input the model requires, runs inference, and returns a `DeepLearningPrediction`.

This preserves model freedom while keeping correction application centralized.

## When A Model-Specific Processor Is Acceptable

A model-specific `Processor` is acceptable only when the generic `DeepLearningCorrection` contract is insufficient.

This should be considered temporary unless the processor introduces a broadly reusable execution mode.

Valid reasons may include:

- The model needs a correction pattern that cannot be represented by one `DeepLearningPrediction`.
- The model requires custom streaming or memory behavior not covered by existing execution modes.
- The model requires tight orchestration between context construction, inference, and reassembly.

If a model-specific processor is added, it should still reuse core correction semantics where possible and should be documented as model-specific.

## Compatibility Wrappers

Closed-beta code may keep compatibility wrappers in old locations, for example in `facet.correction`, so existing scripts do not break immediately.

Wrappers should contain no model logic. They should only re-export the relocated implementation and explain why the import path exists.

## Demo 01 Status

`demo01` contains the first seven-epoch context CNN proof-of-concept. It is frozen as a demo model and should not be used as the template for all future models.

The important architectural lesson from Demo 01 is that model-specific context construction should not automatically become a generic FACETpy correction primitive.

## Cascaded DAE Status

`cascaded_dae` contains a channel-wise cascaded denoising autoencoder inspired by the older `feature/deeplearning` PyTorch prototype.

It deliberately trains on single-channel windows. This keeps the checkpoint independent of the number of channels in a target dataset, while still allowing fixed-size windowed inference via a model-specific adapter.

## Cascaded Context DAE Status

`cascaded_context_dae` extends the cascaded autoencoder idea to seven trigger-defined context epochs.

It still trains and infers per channel, so it remains compatible with different channel counts. Unlike `cascaded_dae`, it requires trigger metadata during inference and is coupled to the configured context length.

## Related Documents

- [`docs/PROCESSOR_GUIDELINES.md`](../../../docs/PROCESSOR_GUIDELINES.md) — binding rules for any FACETpy processor (anatomy, validation, registration, testing).
- [`docs/source/development/training_cli_architecture.md`](../../../docs/source/development/training_cli_architecture.md) — wrapper-vs-adapter rationale and the `facet-train` data flow.
- [`evaluation_standard.md`](evaluation_standard.md) — required evaluation outputs and metric groups.
- [`docs/deep_learning_parallel_runpod_workflow.md`](../../../docs/deep_learning_parallel_runpod_workflow.md) — GPU fleet operation guide for parallel model training.
- [`docs/research/architecture_catalog.md`](../../../docs/research/architecture_catalog.md) — menu of deep-learning architectures available for new model agents.
- [`docs/research/dl_eeg_gradient_artifacts.pdf`](../../../docs/research/dl_eeg_gradient_artifacts.pdf) — comprehensive technical report on deep-learning architectures for gradient artifact removal.
- [`docs/model_agent_prompts.md`](../../../docs/model_agent_prompts.md) — reusable prompt template for spawning parallel model-development agents.
