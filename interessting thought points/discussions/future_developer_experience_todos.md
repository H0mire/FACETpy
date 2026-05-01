# Future Developer Experience TODOs for Deep-Learning Models

## Context

The deep-learning integration is currently still in a closed-beta research phase. At this stage, helper commands and polished onboarding workflows are not yet required. The priority remains scientific correctness, pipeline compatibility, and understanding the limitations of the first model generations.

However, if the deep-learning module is later opened to additional developers or researchers, the learning curve should be reduced deliberately. The following items are therefore recorded as future TODOs, not as current implementation requirements.

## TODOs For Later

- Add a simple `facet-train init` command that creates a new experiment folder from a template.

- Provide a small set of official model templates instead of expecting users to design everything from scratch.

- Start with templates such as `artifact-cnn-single-channel`, `artifact-unet-single-channel`, and `clean-signal-baseline`.

- Generate a standard experiment structure for new models.

```text
experiments/my_model/
├── README.md
├── model.py
├── train.yaml
├── dataset.yaml
├── inference.yaml
├── model_card.md
└── results/
```

- Define one public minimal model interface for common PyTorch models.

```python
class MyArtifactModel(torch.nn.Module):
    def forward(self, x):
        return predicted_artifact
```

- Document the expected tensor contract clearly.

```text
Input:  batch x context_epochs x channels x samples
Output: batch x channels x samples
```

- Hide adapter, export, and execution-mode details from beginner-facing documentation.

- Keep advanced concepts such as `ModelSpec`, `ArtifactLibrary`, execution modes, ONNX export, and backend adapters in an advanced section.

- Add a required `model_card.md` template for every trained model.

- The model card should describe the model goal, input shape, output type, training data, evaluation data, known limitations, and intended pipeline usage.

- Add human-readable config validation errors.

- Prefer messages such as: `This model expects 7 context epochs, but inference was configured with 5.`

- Avoid exposing low-level tensor or backend errors to researchers when the cause can be explained in domain terms.

- Add a single beginner walkthrough named approximately `Train your first FACETpy artifact model`.

- The walkthrough should cover dataset creation, training, evaluation, export, and pipeline inference on one known example dataset.

- Add a single-command happy path after the architecture stabilizes.

```bash
uv run facet-train run --config train.yaml
```

- Keep the lower-level commands available for debugging and advanced research use.

- Organize documentation into levels.

```text
Level 1: Use an existing template
Level 2: Modify a model architecture
Level 3: Add a new artifact source or dataset
Level 4: Add a new backend or adapter
```

- Ensure that future examples separate beginner-facing usage from internal implementation details.

- Add generated `inference.yaml` and `model_spec.json` files during export so researchers do not need to manually recreate inference settings.

- Consider a high-level Python API only after the CLI and config workflow are stable.

```python
from facet.training import ArtifactModelExperiment

experiment = ArtifactModelExperiment.from_config("train.yaml")
experiment.run()
```

## Rationale

These tasks can safely be postponed. They primarily improve accessibility, onboarding, and long-term maintainability. They are not required to validate the scientific approach of the current closed-beta models.

The architectural implication is that internal abstractions should remain clean enough that such helper layers can be added later, but the project should not spend implementation time on beginner tooling before the model design, evaluation protocol, and artifact-library strategy have stabilized.
