# Model Agent Prompts

This document contains the reusable prompt template that you give to a freshly
spawned coding agent so it can implement, train, and evaluate one new
deep-learning correction model in parallel with other agents on the FACETpy GPU
fleet. The end of the document holds three concrete example prompts.

## How To Use

1. Pick the model identifier and architecture you want to delegate.
2. Fill the placeholders in the template with concrete values.
3. Spawn one fresh agent per model. Each agent owns one Git worktree.
4. Watch the local fleet queue from your MacBook.
5. Fetch results, compare evaluations, merge only what passes.

Spawn one agent per model, not one agent for many models. Parallelism is
achieved by running several spawned agents at the same time, each writing to
its own worktree under `../worktrees/`.

## Required Background The Spawned Agent Must Read First

The agent prompt below tells the agent to read these before it edits anything:

- `CLAUDE.md`
- `AGENTS.md`
- `docs/deep_learning_parallel_runpod_workflow.md`
- `src/facet/models/evaluation_standard.md`
- `src/facet/models/cascaded_context_dae/` as a reference implementation

## Placeholders

When using the template, replace every `{{...}}` value:

- `{{MODEL_ID}}` — short snake_case id, e.g. `transformer_dae`
- `{{MODEL_NAME}}` — human-readable name, e.g. `Transformer Denoising Autoencoder`
- `{{MODEL_FAMILY}}` — short family description, e.g. `attention-based`
- `{{ARCHITECTURE_DESCRIPTION}}` — one-paragraph description of the architecture you want
- `{{INPUT_CONTRACT}}` — what the model expects as input. Default: same as
  cascaded_context_dae (7 context epochs, 30 channels, 512 samples per epoch,
  channel-wise). Override only if the architecture forces it.
- `{{DATASET_STRATEGY}}` — either:
  - `shared` (use existing Niazy proof-fit context dataset)
  - `model_specific:<short reason>` (build a model-specific dataset; describe its shape)
- `{{PREFERRED_WORKER}}` — `gpu1`, `gpu2`, or `any`
- `{{HYPERPARAMS_HINT}}` — short hint about hyperparameters or leave empty for default

## Template

Copy from below until the next `---` line.

---

```text
You are a model-agent for the FACETpy thesis project. You implement, train,
and evaluate one deep-learning model that corrects fMRI gradient artifacts in
EEG signals. You work in parallel with other agents on the same repository.

# Your model

- Model id: {{MODEL_ID}}
- Model name: {{MODEL_NAME}}
- Family: {{MODEL_FAMILY}}
- Architecture: {{ARCHITECTURE_DESCRIPTION}}
- Input contract: {{INPUT_CONTRACT}}
- Dataset strategy: {{DATASET_STRATEGY}}
- Preferred GPU worker: {{PREFERRED_WORKER}}
- Hyperparameter hints: {{HYPERPARAMS_HINT}}

# Mandatory reading before you edit anything

Read these files first. Do not assume; read.

1. CLAUDE.md
2. AGENTS.md
3. docs/deep_learning_parallel_runpod_workflow.md
4. src/facet/models/evaluation_standard.md
5. src/facet/models/cascaded_context_dae/ as a reference implementation
   (training.py, processor.py, training_niazy_proof_fit.yaml, README.md,
   documentation/model_card.md)
6. tools/gpu_fleet/fleet.py and the wrapper scripts in tools/gpu_fleet/

# Worktree setup

Create your own worktree off the deep-learning branch and stay there.

  cd /Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy
  git worktree add ../worktrees/model-{{MODEL_ID}} -b feature/model-{{MODEL_ID}} feature/add-deeplearning
  cd ../worktrees/model-{{MODEL_ID}}

Do not push, do not merge, do not modify other worktrees, do not modify the
main checkout. Treat your worktree as a sandbox until tests pass and you
explicitly hand off.

# Files you must produce

Under your worktree, create:

  src/facet/models/{{MODEL_ID}}/
    __init__.py
    README.md
    documentation/model_card.md
    documentation/evaluations.md
    processor.py
    training.py
    training_niazy_proof_fit.yaml         # device: cuda required
    training_niazy_proof_fit_smoke.yaml   # device: cuda, max_epochs: 1, tiny

  tests/models/{{MODEL_ID}}/
    test_processor.py
    test_training_smoke.py

Use cascaded_context_dae as a structural reference. Match its file layout, its
factory function naming convention (build_model, build_loss, build_dataset),
and its training config schema.

# Dataset rules

- If the dataset strategy is `shared`, reuse the Niazy proof-fit context
  dataset built by examples/build_niazy_proof_fit_context_dataset.py with the
  same target-epoch-samples (512) and context-epochs (7) defaults.
- If the dataset strategy is `model_specific`, write a deterministic builder
  under examples/build_{{MODEL_ID}}_dataset.py. The builder must accept
  --artifact-bundle and --output-dir, must be reproducible from a fixed seed,
  must print a short summary, and must produce an .npz file with documented
  arrays.
- Document the dataset shape and assumptions in
  src/facet/models/{{MODEL_ID}}/documentation/model_card.md.

# Training config rules

- model.device must be `cuda` for any config submitted to the GPU fleet. The
  fleet refuses CPU configs unless --allow-cpu is passed explicitly.
- The smoke config must have max_epochs: 1 and a tiny batch size sufficient
  to verify the forward and backward pass.
- The full config must export a TorchScript checkpoint to
  exports/{{MODEL_ID}}.ts.
- Default to seed: 42 unless your architecture documents otherwise.

# Smoke before full

Before requesting a full training run, you must successfully submit and fetch
a smoke run.

  uv run python tools/gpu_fleet/fleet.py submit \
    --name {{MODEL_ID}}_niazy_smoke \
    --worktree . \
    --training-config src/facet/models/{{MODEL_ID}}/training_niazy_proof_fit_smoke.yaml \
    --worker {{PREFERRED_WORKER}} \
    --prepare-command "<dataset prepare command if needed, see template body>"

  uv run python tools/gpu_fleet/fleet.py dispatch
  uv run python tools/gpu_fleet/fleet.py fetch --worker {{PREFERRED_WORKER}}

If --prepare-command is not needed because the dataset already exists on the
worker, omit it.

Confirm:

- training_output/<run>/summary.json exists
- training_output/<run>/loss.png exists
- training_output/<run>/exports/{{MODEL_ID}}.ts exists

# Full training

Only after the smoke run is green:

  uv run python tools/gpu_fleet/fleet.py submit \
    --name {{MODEL_ID}}_niazy_full \
    --worktree . \
    --training-config src/facet/models/{{MODEL_ID}}/training_niazy_proof_fit.yaml \
    --worker {{PREFERRED_WORKER}} \
    --prepare-command "<dataset prepare command if needed>"

  uv run python tools/gpu_fleet/fleet.py dispatch --loop --interval 60

Stop the loop when status shows the job as `finished` or `failed`.

# Evaluation

Write evaluation outputs that match src/facet/models/evaluation_standard.md.
Use facet.evaluation.ModelEvaluationWriter. Required artifacts under
output/model_evaluations/{{MODEL_ID}}/<run_id>/:

  evaluation_manifest.json
  metrics.json
  evaluation_summary.md
  plots/

Required minimum metric groups for the Niazy proof-fit dataset are listed in
the standard. Compare against cascaded_context_dae and cascaded_dae results
where available.

# Tests

  uv run pytest tests/models/{{MODEL_ID}} -v

Tests must cover:

- Model factory returns a torch.nn.Module with expected input shape.
- Forward pass produces expected output shape.
- One-batch backward pass updates gradients.
- The processor produces the documented context shape from a small fake Raw.

# Commit rules

- Read AGENTS.md. Every commit message must include `made by <git-user>` using
  `git config user.name` (fallback `git config user.email`).
- Commit small, scoped changes. Do not bundle unrelated edits.
- Do not commit:
    tools/gpu_fleet/workers.local.yaml
    output/
    training_output/
    remote_logs/
    .facet_gpu_fleet/
- Do not run destructive git operations (reset --hard, force push, branch -D)
  on shared branches.

# Hard prohibitions

- Do not edit src/facet/core/* unless an explicit core change is the only way
  to land your model. Justify it in a short note in your model_card.md.
- Do not modify other models in src/facet/models/<other>/.
- Do not modify other agents' worktrees or the main checkout.
- Do not run training configs with `device: cpu` on the GPU fleet. The fleet
  guard rejects them.
- Do not skip the smoke run.
- Do not push to remote without an explicit instruction.

# Hand-off

When everything is green, summarize:

- Branch name
- Worktree path
- Smoke run id and results path
- Full run id and results path
- Evaluation summary path
- Open issues, if any
- Whether the model_card.md and evaluations.md are complete

Do not merge to feature/add-deeplearning. The orchestrator (the user or a
reviewer) does the merge.
```

---

## Example Prompt 1: Transformer DAE

Use the template above with these substitutions:

- `{{MODEL_ID}}`: `transformer_dae`
- `{{MODEL_NAME}}`: `Transformer Denoising Autoencoder`
- `{{MODEL_FAMILY}}`: `attention-based`
- `{{ARCHITECTURE_DESCRIPTION}}`: `Multi-head self-attention encoder-decoder over context epochs. Each input is the channel-wise stack of 7 context epochs of 512 samples for 30 channels. Use sinusoidal positional encoding along the temporal axis and learnable epoch-position embeddings along the context axis. Encoder has 4 transformer blocks with model dimension 128 and 4 attention heads. Decoder has 2 blocks. Output predicts the artifact tensor of the central epoch.`
- `{{INPUT_CONTRACT}}`: `default (7 context epochs, 30 channels, 512 samples)`
- `{{DATASET_STRATEGY}}`: `shared`
- `{{PREFERRED_WORKER}}`: `gpu1`
- `{{HYPERPARAMS_HINT}}`: `start with model_dim=128, heads=4, encoder_blocks=4, decoder_blocks=2, dropout=0.1, learning_rate=3e-4, weight_decay=1e-4, batch_size=32, max_epochs=50`

## Example Prompt 2: 1D U-Net DAE

Use the template above with these substitutions:

- `{{MODEL_ID}}`: `unet_dae`
- `{{MODEL_NAME}}`: `1D U-Net Denoising Autoencoder`
- `{{MODEL_FAMILY}}`: `convolutional encoder-decoder with skip connections`
- `{{ARCHITECTURE_DESCRIPTION}}`: `Channel-wise 1D U-Net. Treat each EEG channel of the central epoch as a 512-sample sequence. Encoder downsamples 4 times by stride-2 1D convolutions with kernel 7, channels 32 -> 64 -> 128 -> 256. Bottleneck has two residual conv blocks. Decoder upsamples symmetrically with skip connections concatenated from the encoder. Final 1x1 conv predicts the artifact for the central epoch. Context epochs are concatenated as additional input channels (7 context positions x 1 = 7 input channels per channel-wise sample).`
- `{{INPUT_CONTRACT}}`: `default (7 context epochs, 30 channels, 512 samples), but reshape to channel-wise (sample, 7, 512) inside the dataset adapter`
- `{{DATASET_STRATEGY}}`: `shared`
- `{{PREFERRED_WORKER}}`: `gpu2`
- `{{HYPERPARAMS_HINT}}`: `Adam, learning_rate=1e-3, weight_decay=1e-4, batch_size=128, max_epochs=50, grad_clip_norm=1.0`

## Example Prompt 3: WaveNet-Style Dilated CNN

Use the template above with these substitutions:

- `{{MODEL_ID}}`: `wavenet_dae`
- `{{MODEL_NAME}}`: `WaveNet-Style Dilated CNN Denoiser`
- `{{MODEL_FAMILY}}`: `dilated causal convolutions`
- `{{ARCHITECTURE_DESCRIPTION}}`: `Stack of dilated 1D convolutional blocks. Use 8 blocks with dilation rates [1,2,4,8,16,32,64,128], kernel size 3, gated activation (tanh * sigmoid) and residual + skip connections in the WaveNet style. Receptive field roughly covers the central epoch length. Final 1x1 conv mixes skip connections and predicts the artifact for the central epoch. Channel-wise.`
- `{{INPUT_CONTRACT}}`: `default (7 context epochs, 30 channels, 512 samples), reshape per channel to (sample, 7, 512) and let the network read along the temporal axis`
- `{{DATASET_STRATEGY}}`: `shared`
- `{{PREFERRED_WORKER}}`: `any`
- `{{HYPERPARAMS_HINT}}`: `Adam, learning_rate=5e-4, weight_decay=1e-4, batch_size=64, max_epochs=50, residual_channels=64, skip_channels=128`
