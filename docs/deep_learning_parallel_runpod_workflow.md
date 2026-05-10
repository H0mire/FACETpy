# Parallel Deep-Learning Workflow With RunPod GPUs

This workflow uses the MacBook as the orchestrator and each RunPod instance as a single-GPU worker. Many model agents can submit jobs; the two GPUs process those jobs through a local queue.

## Architecture

```text
MacBook
├── main FACETpy checkout
├── git worktrees for model agents
├── tools/gpu_fleet/*.sh
└── collected results

RunPod GPU 1
└── /workspace/facetpy

RunPod GPU 2
└── /workspace/facetpy
```

Do not build a custom GPU server for two pods. SSH plus `tmux`, `rsync`, `uv`, FACETpy's training/evaluation output conventions, and a small local queue are enough.

## Rules For Agents

- One model agent gets one Git worktree.
- One model agent writes mostly under `src/facet/models/<model_id>/` and `tests/models/`.
- Avoid simultaneous edits to core files such as `src/facet/training`, `src/facet/correction/deep_learning.py`, and dataset builders.
- Many agents may submit jobs, but each RunPod worker executes only one job at a time.
- A worker repo must not be overwritten while a job is running; sync only happens when the dispatcher assigns a pending job to an idle worker.
- Each model must provide `training.py`, `processor.py`, `training.yaml`, `README.md`, and `documentation/model_card.md`.
- Each training run must produce `training.jsonl`, `loss.png`, `summary.json`, checkpoints, and an exported model.
- Each evaluation run must produce `evaluation_manifest.json`, `metrics.json`, `evaluation_summary.md`, and plots under `output/model_evaluations/<model_id>/<run_id>/`.
- If the shared dataset is incompatible with a model architecture, the agent may build a model-specific dataset instead of forcing the architecture to fit the dataset.
- Model-specific datasets must be deterministic, documented in the model folder, and stored under `output/<model_id_or_experiment>/` on the worker.

## Recommended Local Worktrees

```bash
mkdir -p ../worktrees

git worktree add ../worktrees/model-unet -b feature/model-unet
git worktree add ../worktrees/model-tcn -b feature/model-tcn
git worktree add ../worktrees/model-transformer -b feature/model-transformer
```

Before parallelizing, commit or intentionally snapshot the current baseline. Dirty local state makes worker results hard to reproduce.

## Worker Configuration

Create a local worker config and do not commit it:

```bash
cp tools/gpu_fleet/workers.example.yaml tools/gpu_fleet/workers.local.yaml
```

Example:

```yaml
workers:
  gpu1:
    ssh: root@<runpod-host-1>
    port: 22
    remote_repo: /workspace/facetpy
    gpu: 0
    identity_file: /path/to/private/key
  gpu2:
    ssh: root@<runpod-host-2>
    port: 22
    remote_repo: /workspace/facetpy
    gpu: 0
    identity_file: /path/to/private/key
```

## Queue-Based Scheduling

For more model agents than GPUs, use the local fleet queue:

```bash
python tools/gpu_fleet/fleet.py submit \
  --name context_dae_niazy \
  --worktree ../worktrees/model-context-dae \
  --training-config src/facet/models/cascaded_context_dae/training_niazy_proof_fit.yaml

python tools/gpu_fleet/fleet.py submit \
  --name next_architecture_niazy \
  --worktree ../worktrees/model-next-architecture \
  --training-config src/facet/models/<model_id>/training_niazy_proof_fit.yaml
```

If a model requires its own dataset layout, submit a preparation command with the job:

```bash
python tools/gpu_fleet/fleet.py submit \
  --name unet_niazy_1024 \
  --worktree ../worktrees/model-unet \
  --training-config src/facet/models/unet/training_niazy_1024.yaml \
  --prepare-command "uv run python examples/build_niazy_proof_fit_context_dataset.py --target-epoch-samples 1024 --context-epochs 9 --output-dir output/unet_niazy_1024"
```

The preparation command runs on the selected RunPod after the worktree is synced and before `facet-train` starts. This keeps architecture-specific dataset decisions inside the model-agent workflow.

Start the dispatcher from the MacBook:

```bash
python tools/gpu_fleet/fleet.py dispatch --loop --interval 60
```

The dispatcher checks the local queue, picks idle workers, syncs the assigned worktree to the selected RunPod, and starts the remote training in a detached `tmux` session. Jobs remain pending until one of the two GPUs is free.

Check status:

```bash
python tools/gpu_fleet/fleet.py status
```

Fetch results from all workers:

```bash
python tools/gpu_fleet/fleet.py fetch
```

The queue state is local and intentionally ignored by Git:

```text
.facet_gpu_fleet/queue.json
```

This design allows many agents to develop and submit model experiments without requiring every agent to own a dedicated GPU. The MacBook remains the scheduler and integration point.

## Bootstrap A RunPod

RunPod PyTorch/Jupyter images usually already include CUDA, PyTorch, and Python. The bootstrap script adds `uv`, clones the repo, syncs dependencies, and checks CUDA.

```bash
tools/gpu_fleet/bootstrap_runpod.sh root@<runpod-host> <repo-url> /workspace/facetpy <ssh-port>
```

If the pod already has the repo mounted at `/workspace/facetpy`, the script reuses it.

When the RunPod base image already provides a CUDA-enabled PyTorch installation, the worker scripts create the project `.venv` with `uv venv --system-site-packages`. This keeps the project dependencies isolated while still allowing `uv run` to import the image-provided `torch` build.

## Sync A Local Worktree To A Pod

Use this when the model agent has uncommitted changes in its worktree:

```bash
tools/gpu_fleet/sync_worktree_to_runpod.sh ../worktrees/model-unet root@<runpod-host> /workspace/facetpy <ssh-port>
```

The script excludes large/generated directories such as `output/`, `training_output/`, `.venv/`, and caches.

## Sync A Generated Dataset To A Pod

Use this when an agent builds a dataset locally and needs to provide it to a worker:

```bash
tools/gpu_fleet/sync_dataset_to_runpod.sh \
  output/<dataset_id> \
  root@<runpod-host> \
  /workspace/facetpy/output/<dataset_id> \
  <ssh-port>
```

Normal worktree sync intentionally excludes `output/`. Dataset sync is explicit so agents do not accidentally overwrite large generated artifacts or mix datasets between experiments.

Prefer remote dataset builds through `--prepare-command` when the source data and builder are available on the pod. Prefer dataset sync when the dataset was generated locally or requires local-only inputs.

## Start Training On A Pod

```bash
tools/gpu_fleet/run_remote_training.sh \
  root@<runpod-host> \
  src/facet/models/<model_id>/training.yaml \
  /workspace/facetpy \
  train_<model_id> \
  <ssh-port> \
  0
```

This starts a detached `tmux` session and uses a GPU lock file:

```text
/tmp/facetpy_gpu_0.lock
```

That prevents accidentally launching two training jobs on the same GPU from these scripts.

## Smoke Run

Use a short smoke config before starting longer experiments:

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

The smoke run verifies worktree sync, optional dataset preparation, CUDA-enabled `uv run`, checkpoint writing, `training.jsonl`, `loss.png`, and TorchScript export.

## Check Remote Status

```bash
tools/gpu_fleet/remote_status.sh root@<runpod-host> /workspace/facetpy <ssh-port>
```

For live logs:

```bash
ssh -p <ssh-port> root@<runpod-host>
tmux attach -t train_<model_id>
```

## Fetch Results

```bash
tools/gpu_fleet/fetch_runpod_results.sh root@<runpod-host> /workspace/facetpy . <ssh-port>
```

This fetches:

- `training_output/`
- `output/model_evaluations/`
- `remote_logs/`

## Two-Pod Scheduling

Suggested split:

```text
gpu1: current queued training job
gpu2: current queued training job
pending queue: all other model-agent jobs
MacBook: dispatcher, dataset inspection, evaluation aggregation, thesis documentation, code review, merge integration
```

## Minimal Experiment Checklist

For each model:

1. Create model worktree and branch.
2. Implement model package under `src/facet/models/<model_id>/`.
3. Add tests under `tests/models/`.
4. Decide whether the shared dataset is compatible with the model's input contract.
5. If needed, add a deterministic dataset builder or preparation command for the model.
6. Submit the training config to the fleet queue.
7. Let the dispatcher sync, prepare data if requested, and train on the next idle RunPod.
8. Evaluate via the standard evaluation scripts.
9. Fetch results.
10. Compare `metrics.json` and plots.
11. Merge only if tests pass and documentation is complete.
