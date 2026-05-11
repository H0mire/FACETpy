# Agent Handoff: GPU Fleet And Deep-Learning Training Status

Date: 2026-05-10  
Branch: `feature/add-deeplearning`  
Repository: `/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy`

## Purpose

This note summarizes the current state so another agent can continue without relying on chat history. The immediate workstream is the orchestration of many model-development agents over two RunPod GPUs, plus the first proof-fit deep-learning dataset/training flow based on Niazy EEG data.

## Current User Goal

The user wants to accelerate the thesis project by letting many agents develop and train different deep-learning correction models in parallel. Two RunPod machines with NVIDIA RTX 5090 GPUs should be shared by an arbitrary number of agents through a queue rather than by assigning one GPU per agent.

Important requirements:

- Agents should work in separate Git worktrees.
- The MacBook should act as orchestrator and integration point.
- RunPods should execute queued training jobs via SSH/tmux/uv.
- More agents than GPUs must be possible; jobs wait until a GPU is free.
- If a model architecture is incompatible with the common dataset layout, that model agent may build a model-specific dataset.
- Dataset preparation must be reproducible and documented.
- Model-specific implementations should remain under `src/facet/models/<model_id>/` unless a generic FACETpy abstraction is clearly justified.

## Repository Rules To Preserve

`AGENTS.md` is present at the repository root. The important operational rule is:

- Every commit message must include `made by <git-user>`.
- Use `git config user.name` as the marker source, falling back to `git config user.email`.
- Current git user during this session was `Müller Janik Michael`, so recent commits use `made by Müller Janik Michael`.

Use `uv` for commands:

```bash
uv sync
uv run pytest
uv run facet-train fit --config <config.yaml>
```

## Recent Commit Baseline

The latest relevant commits before this handoff were:

```text
2260618 fix: harden runpod gpu smoke workflow made by Müller Janik Michael
4242fa9 feat: allow gpu jobs to prepare datasets made by Müller Janik Michael
9b39196 feat: support ssh keys for gpu fleet workers made by Müller Janik Michael
35a03aa feat: add gpu fleet queue scheduler made by Müller Janik Michael
48346e6 feat: add niazy proof fit dataset workflow made by Müller Janik Michael
```

After creating this handoff note, check `git status --short` before doing anything else.

## Main Files Added Or Modified For GPU Fleet

### Queue scheduler

`tools/gpu_fleet/fleet.py`

Provides:

```bash
python tools/gpu_fleet/fleet.py submit --name <name> --worktree <path> --training-config <config> [--prepare-command <cmd>] [--worker <gpu1|gpu2>]
python tools/gpu_fleet/fleet.py dispatch [--loop --interval 60]
python tools/gpu_fleet/fleet.py status
python tools/gpu_fleet/fleet.py fetch [--worker <gpu1|gpu2>]
python tools/gpu_fleet/fleet.py cancel <job_id>
```

Behavior:

- Queue state is local and ignored by Git: `.facet_gpu_fleet/queue.json`.
- Worker config is local and ignored by Git: `tools/gpu_fleet/workers.local.yaml`.
- Jobs can include `--prepare-command`, which runs on the worker after worktree sync and before training.
- Recent hardening added remote exit-code files so future jobs can become `finished` or `failed` instead of only `finished_unknown`.
- Older queue entries may still show `finished_unknown`; treat them as historical diagnostics.

### RunPod sync and execution scripts

`tools/gpu_fleet/sync_worktree_to_runpod.sh`

- Syncs a local worktree to a RunPod.
- Excludes generated/cached/heavy folders such as `.venv/`, `output/`, `training_output/`, `remote_logs/`, caches, docs build output, and local worker config.
- Uses `rsync --no-owner --no-group` to avoid chown failures on RunPod.
- Creates `.venv` with `uv venv --system-site-packages` when the RunPod image has system CUDA PyTorch available but the project venv cannot import torch.

`tools/gpu_fleet/run_remote_training.sh`

- Starts training in detached `tmux`.
- Uses a per-GPU lock file: `/tmp/facetpy_gpu_<gpu>.lock`.
- Base64-encodes the prepare command before sending it over SSH, so commands with spaces are preserved.
- Writes remote files under `remote_logs/`:
  - `<session>.sh`
  - `<session>.runner.sh`
  - `<session>.log`
  - `<session>.exitcode`

`tools/gpu_fleet/sync_dataset_to_runpod.sh`

- Explicitly uploads datasets/artifact bundles to a RunPod.
- Intended because normal worktree sync excludes `output/`.

`tools/gpu_fleet/bootstrap_runpod.sh`

- Installs or reuses `uv`.
- Clones/reuses repo at `/workspace/facetpy`.
- Handles RunPod PyTorch/Jupyter images with preinstalled CUDA PyTorch by using `uv venv --system-site-packages`.

`tools/gpu_fleet/remote_status.sh`

- Shows `nvidia-smi`, tmux sessions, latest training runs, and latest logs on a worker.

### Worker config

Local file, ignored by Git:

```text
tools/gpu_fleet/workers.local.yaml
```

It contains the two RunPod SSH endpoints and the SSH key. Do not commit it.

Example schema:

```yaml
workers:
  gpu1:
    ssh: root@<host-1>
    port: <port-1>
    remote_repo: /workspace/facetpy
    gpu: 0
    identity_file: /Users/janikmueller/.ssh/runpod
  gpu2:
    ssh: root@<host-2>
    port: <port-2>
    remote_repo: /workspace/facetpy
    gpu: 0
    identity_file: /Users/janikmueller/.ssh/runpod
```

## Main Documentation

`docs/deep_learning_parallel_runpod_workflow.md`

This is the canonical workflow document for the current GPU fleet approach. It describes:

- MacBook-as-orchestrator architecture.
- Agent rules.
- Worktree strategy.
- Queue scheduling.
- Dataset preparation with `--prepare-command`.
- Dataset sync.
- Smoke run.
- RunPod bootstrap.
- Fetching results.

If continuing the orchestration work, update this doc alongside code changes.

## Niazy Proof-Fit Dataset Status

Goal: for the first iteration, prove that models can fit/correct on Niazy-derived data. The user explicitly wanted Niazy as training dataset and also inference/evaluation proof source, not cross-dataset generalization yet.

Important source artifact bundle:

```text
output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz
```

The proof-fit context dataset builder:

```text
examples/build_niazy_proof_fit_context_dataset.py
```

Smoke dataset command used remotely:

```bash
uv run python examples/build_niazy_proof_fit_context_dataset.py \
  --artifact-bundle output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz \
  --target-epoch-samples 512 \
  --context-epochs 7 \
  --output-dir output/niazy_proof_fit_context_512
```

Observed successful remote dataset stats:

```text
examples: 833
shape: (833, 7, 30, 512)
native length min/mean/max: 584 / 584.0 / 605
mean clean: 215.209 uV
mean artifact: 918.985 uV
```

## Smoke Training Status

A successful CUDA smoke run was executed on `gpu1`.

Submitted with:

```bash
python tools/gpu_fleet/fleet.py submit \
  --name context_dae_niazy_smoke_cuda \
  --worktree . \
  --training-config src/facet/models/cascaded_context_dae/training_niazy_proof_fit_smoke.yaml \
  --worker gpu1 \
  --prepare-command "uv run python examples/build_niazy_proof_fit_context_dataset.py --artifact-bundle output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz --target-epoch-samples 512 --context-epochs 7 --output-dir output/niazy_proof_fit_context_512"

python tools/gpu_fleet/fleet.py dispatch
python tools/gpu_fleet/fleet.py fetch --worker gpu1
```

Remote run:

```text
session: context_dae_niazy_smoke_cuda
remote output dir: /workspace/facetpy/training_output/cascadedcontextdenoisingautoencoderniazysmoke_20260510_170458
device: cuda
max_epochs: 1
best epoch: 1
best metric: 8.233534753678294e-07
```

Fetched local results:

```text
training_output/cascadedcontextdenoisingautoencoderniazysmoke_20260510_170458/summary.json
training_output/cascadedcontextdenoisingautoencoderniazysmoke_20260510_170458/loss.png
training_output/cascadedcontextdenoisingautoencoderniazysmoke_20260510_170458/training.jsonl
training_output/cascadedcontextdenoisingautoencoderniazysmoke_20260510_170458/exports/cascaded_context_dae.ts
```

These output folders are ignored by Git.

## Known Issues Already Fixed

During smoke testing, these issues occurred and were fixed:

- `rsync` failed because RunPod rejected ownership changes. Fixed with `--no-owner --no-group`.
- Worktree sync transferred too much local/generated state. Fixed by expanding excludes in `sync_worktree_to_runpod.sh`.
- `--prepare-command` was truncated to only `uv` due SSH argument splitting. Fixed by base64-encoding the command in `run_remote_training.sh`.
- `uv run` initially could not import CUDA PyTorch because RunPod's PyTorch was installed system-wide. Fixed by creating `.venv` with `--system-site-packages` when necessary.

## Important Design Decisions

### Do not build a central GPU server yet

For two RunPods, SSH + tmux + rsync + a local queue is simpler and sufficient. A full server would add operational complexity without much benefit at the current scale.

### Use worktrees for agent isolation

Each model agent should use a separate Git worktree and mostly write under its model folder:

```text
src/facet/models/<model_id>/
tests/models/<model_id>/
```

Avoid concurrent core edits unless absolutely necessary.

### Let models own incompatible dataset preparation

If a model cannot use the shared Niazy context dataset, the agent should create a model-specific deterministic dataset builder and run it via `--prepare-command`. This is preferred over forcing all architectures into one rigid dataset shape.

### Keep generic FACETpy abstractions small

Only move code into generic FACETpy modules when multiple models need the same concept. Otherwise, keep model-specific code in the model package to avoid locking the whole framework into one architecture assumption.

## Current Model/Training Context

Existing model families in this workstream include:

- `cascaded_dae`
- `cascaded_context_dae`

The context model uses 7 context epochs and currently works with a static target epoch sample length in the proof-fit dataset. The architecture may support variable lengths only where its implementation actually avoids fixed-size layers/assumptions. Do not assume variable artifact length compatibility purely from the high-level FACETpy architecture.

The user has explicitly accepted that current models can be treated as early demo/proof models. The current priority is infrastructure and comparability, not optimizing the first demo model further.

## Recommended Immediate Next Steps

1. Check current state:

```bash
git status --short
python tools/gpu_fleet/fleet.py status
```

2. Sync the Niazy artifact bundle to `gpu2` if not already present:

```bash
FACET_GPU_FLEET_SSH_KEY=/Users/janikmueller/.ssh/runpod \
tools/gpu_fleet/sync_dataset_to_runpod.sh \
  output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz \
  root@<gpu2-host> \
  /workspace/facetpy/output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz \
  <gpu2-port>
```

3. Create CUDA variants of real training configs if current configs still say `device: cpu`. Do not run CPU configs on RunPod accidentally.

4. Submit two non-smoke jobs, one per GPU. For example:

```bash
python tools/gpu_fleet/fleet.py submit \
  --name context_dae_niazy_full_gpu1 \
  --worktree . \
  --training-config src/facet/models/cascaded_context_dae/<cuda-full-config>.yaml \
  --worker gpu1 \
  --prepare-command "uv run python examples/build_niazy_proof_fit_context_dataset.py --artifact-bundle output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz --target-epoch-samples 512 --context-epochs 7 --output-dir output/niazy_proof_fit_context_512"

python tools/gpu_fleet/fleet.py submit \
  --name dae_or_next_model_niazy_full_gpu2 \
  --worktree <model-worktree> \
  --training-config src/facet/models/<model_id>/<cuda-full-config>.yaml \
  --worker gpu2 \
  --prepare-command "<model-specific deterministic dataset build command>"

python tools/gpu_fleet/fleet.py dispatch --loop --interval 60
```

5. Fetch results:

```bash
python tools/gpu_fleet/fleet.py fetch
```

6. Evaluate every trained model using the standard result layout:

```text
output/model_evaluations/<model_id>/<run_id>/evaluation_manifest.json
output/model_evaluations/<model_id>/<run_id>/metrics.json
output/model_evaluations/<model_id>/<run_id>/evaluation_summary.md
output/model_evaluations/<model_id>/<run_id>/plots/
```

## Useful Future Improvements

These are not mandatory before the next training run, but they would reduce friction:

- Add `fleet.py logs <job_id>` to stream or tail remote logs without manual SSH.
- Add `fleet.py remote-status [--worker]` as a wrapper around `remote_status.sh`.
- Add `fleet.py sync-dataset <worker> <local> <remote>` as a wrapper around `sync_dataset_to_runpod.sh`.
- Add a queue cleanup command for old `finished_unknown` entries.
- Add job-level dataset dependencies so common datasets can be synced automatically before dispatch.
- Add an agent template for new models with required files: `training.py`, `processor.py`, training config, model card, tests, evaluation command.
- Add stronger validation that a RunPod training config uses `device: cuda` when submitted to a GPU worker.

## Thesis-Relevant Summary

The current orchestration approach separates model development from GPU allocation. Model agents work independently in Git worktrees and submit deterministic training jobs to a local queue. The MacBook synchronizes the selected worktree to an idle RunPod, optionally executes a model-specific dataset preparation command, starts training in `tmux`, and later fetches standardized artifacts. This allows many model experiments to be developed concurrently while only two physical GPUs execute jobs at a time. The design deliberately avoids a heavy central GPU server because, for two workers, SSH-based orchestration is more transparent, easier to debug, and sufficient for reproducible thesis experiments.

## What The Next Agent Should Avoid

- Do not commit `tools/gpu_fleet/workers.local.yaml`.
- Do not commit `output/`, `training_output/`, `remote_logs/`, or `.facet_gpu_fleet/`.
- Do not overwrite user changes or reset the branch.
- Do not run long training with a config that still uses `device: cpu`.
- Do not assume generated artifacts are already present on both pods; verify or sync them.
- Do not move model-specific logic into core FACETpy unless there is a clear multi-model need.
