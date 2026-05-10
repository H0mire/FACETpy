# D4PM

Single-branch conditional denoising-diffusion gradient-artifact predictor,
adapted from the dual-branch D4PM paper
([arXiv 2509.14302](https://arxiv.org/abs/2509.14302)).

The reference D4PM trains two independent diffusion models (clean EEG and
artifact) and joins them at sampling time with a posterior consistency
constraint `x_clean + x_artifact = x_noisy`. Our adaptation trains only the
artifact branch as a conditional model `p(h | y)` because the Niazy
proof-fit dataset already supplies clean and artifact targets per epoch.
The data-consistency residual is folded into each reverse step:
`h0_pred += λ_dc · (y − h0_pred)`. See
[`documentation/research_notes.md`](documentation/research_notes.md) for
the full derivation and the trade-offs against the dual-branch original.

## Scope

- Input shape per channel: `(1, samples)`. Default `samples = 512`.
- Output shape: `(1, samples)` artifact estimate, subtracted by
  `DeepLearningCorrection`.
- Channel-wise inference, so the checkpoint is independent of channel
  count.
- Requires trigger metadata: epoch boundaries are derived from
  `trigger[i:i+1]`. Each native epoch is resampled to `epoch_samples`,
  passed through the diffusion sampler, then resampled back.

## Training

Build the 7-epoch context dataset first (D4PM only consumes
`noisy_center` and `artifact_center`, but the dataset script writes both):

```bash
uv run python examples/build_niazy_proof_fit_context_dataset.py \
  --artifact-bundle output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz \
  --target-epoch-samples 512 \
  --context-epochs 7 \
  --output-dir output/niazy_proof_fit_context_512
```

Then train and export:

```bash
uv run facet-train fit --config src/facet/models/d4pm/training_niazy_proof_fit.yaml
```

## Inference

```python
from facet.models.d4pm import D4PMArtifactCorrection

context = context | D4PMArtifactCorrection(
    checkpoint_path="training_output/<run>/checkpoints/last.pt",
    epoch_samples=512,
    num_steps=200,
    sample_steps=50,
    device="cuda",
)
```

The adapter loads the **state-dict checkpoint** (`.pt` file written by
the trainer's `CheckpointCallback`), reinstantiates the
`D4PMTrainingModule`, and runs DDPM reverse sampling with
data-consistency reinforcement.

## Why state-dict instead of TorchScript

Diffusion sampling is iterative Python control flow with stochastic
intermediate steps. Tracing the training-time forward (which packs noisy
and target into a 2-channel input and samples a random timestep) does
not produce a useful inference artifact. We therefore document the
state-dict checkpoint as the canonical inference artifact and keep the
torchscript export only as a smoke-test artifact required by the GPU
fleet contract.

## Smoke vs Full

| Aspect | Smoke | Full |
|---|---|---|
| Diffusion training steps T | 200 | 200 |
| Inference sampling steps | 10 | 50 |
| Encoder layers | 2 | 2 |
| `feats` / `d_model` / `d_ff` | 32 / 64 / 256 | 64 / 128 / 512 |
| Epochs | 1 | 30 |
| Batch size | 32 | 64 |

The smoke configuration is sized so that one full epoch through 24990
single-channel examples plus the iterative-sampling export step fits
comfortably under one minute on a single RTX 5090.
