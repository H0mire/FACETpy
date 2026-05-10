# Parallel Deep-Learning Workflow With RunPod GPUs

This workflow uses the MacBook as the orchestrator and each RunPod instance as a single-GPU worker.

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

Do not build a custom GPU server for two pods. SSH plus `tmux`, `rsync`, `uv`, and FACETpy's training/evaluation output conventions are enough.

## Rules For Agents

- One model agent gets one Git worktree.
- One model agent writes mostly under `src/facet/models/<model_id>/` and `tests/models/`.
- Avoid simultaneous edits to core files such as `src/facet/training`, `src/facet/correction/deep_learning.py`, and dataset builders.
- Each model must provide `training.py`, `processor.py`, `training.yaml`, `README.md`, and `documentation/model_card.md`.
- Each training run must produce `training.jsonl`, `loss.png`, `summary.json`, checkpoints, and an exported model.
- Each evaluation run must produce `evaluation_manifest.json`, `metrics.json`, `evaluation_summary.md`, and plots under `output/model_evaluations/<model_id>/<run_id>/`.

## Recommended Local Worktrees

```bash
mkdir -p ../worktrees

git worktree add ../worktrees/model-unet -b feature/model-unet
git worktree add ../worktrees/model-tcn -b feature/model-tcn
git worktree add ../worktrees/model-transformer -b feature/model-transformer
```

Before parallelizing, commit or intentionally snapshot the current baseline. Dirty local state makes worker results hard to reproduce.

## Bootstrap A RunPod

RunPod PyTorch/Jupyter images usually already include CUDA, PyTorch, and Python. The bootstrap script adds `uv`, clones the repo, syncs dependencies, and checks CUDA.

```bash
tools/gpu_fleet/bootstrap_runpod.sh root@<runpod-host> <repo-url> /workspace/facetpy <ssh-port>
```

If the pod already has the repo mounted at `/workspace/facetpy`, the script reuses it.

## Sync A Local Worktree To A Pod

Use this when the model agent has uncommitted changes in its worktree:

```bash
tools/gpu_fleet/sync_worktree_to_runpod.sh ../worktrees/model-unet root@<runpod-host> /workspace/facetpy <ssh-port>
```

The script excludes large/generated directories such as `output/`, `training_output/`, `.venv/`, and caches.

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
gpu1: heavier architecture experiments
  - U-Net / Wave-U-Net
  - transformer / conformer style models

gpu2: medium architecture experiments
  - TCN / WaveNet
  - improved residual context DAE

MacBook:
  - dataset inspection
  - evaluation aggregation
  - thesis documentation
  - code review / merge integration
```

## Minimal Experiment Checklist

For each model:

1. Create model worktree and branch.
2. Implement model package under `src/facet/models/<model_id>/`.
3. Add tests under `tests/models/`.
4. Sync to one RunPod.
5. Train via `facet-train`.
6. Evaluate via the standard evaluation scripts.
7. Fetch results.
8. Compare `metrics.json` and plots.
9. Merge only if tests pass and documentation is complete.
