#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/gpu_fleet/sync_worktree_to_runpod.sh <local-worktree> <ssh-target> [remote-repo-dir] [ssh-port]

Example:
  tools/gpu_fleet/sync_worktree_to_runpod.sh ../worktrees/model-unet root@1.2.3.4 /workspace/facetpy 22

Purpose:
  Rsync the current local worktree to a RunPod. This includes uncommitted model code,
  but excludes large/generated outputs and local caches.
USAGE
}

if [[ $# -lt 2 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

LOCAL_WORKTREE="$1"
SSH_TARGET="$2"
REMOTE_REPO="${3:-/workspace/facetpy}"
SSH_PORT="${4:-22}"
SSH_KEY="${FACET_GPU_FLEET_SSH_KEY:-}"
SSH_ARGS=(-p "$SSH_PORT" -o StrictHostKeyChecking=accept-new)
RSYNC_SSH="ssh -p $SSH_PORT -o StrictHostKeyChecking=accept-new"
if [[ -n "$SSH_KEY" ]]; then
  SSH_ARGS+=(-i "$SSH_KEY" -o IdentitiesOnly=yes)
  RSYNC_SSH="ssh -p $SSH_PORT -o StrictHostKeyChecking=accept-new -i $SSH_KEY -o IdentitiesOnly=yes"
fi

if [[ ! -d "$LOCAL_WORKTREE" ]]; then
  echo "Local worktree not found: $LOCAL_WORKTREE" >&2
  exit 1
fi

ssh "${SSH_ARGS[@]}" "$SSH_TARGET" "mkdir -p '$REMOTE_REPO'"
rsync -az --delete --no-owner --no-group \
  -e "$RSYNC_SSH" \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '.coverage*' \
  --exclude '.facet_gpu_fleet/' \
  --exclude '.mne-home/' \
  --exclude '.mplconfig/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '.uv-cache/' \
  --exclude '.claude/' \
  --exclude '.archiv/' \
  --exclude '.vscode/' \
  --exclude '__pycache__/' \
  --exclude 'build/' \
  --exclude 'dist/' \
  --exclude 'docs/build/' \
  --exclude 'htmlcov/' \
  --exclude 'logs/' \
  --exclude 'output/' \
  --exclude 'remote_logs/' \
  --exclude 'training_output/' \
  --exclude 'worktrees/' \
  --exclude 'export/' \
  --exclude 'ai-docs/' \
  --exclude 'quick-notes/' \
  --exclude 'facet matlab edition/' \
  --exclude 'tools/gpu_fleet/workers.local.yaml' \
  --exclude '.DS_Store' \
  "$LOCAL_WORKTREE/" "$SSH_TARGET:$REMOTE_REPO/"

ssh "${SSH_ARGS[@]}" "$SSH_TARGET" "cd '$REMOTE_REPO' && if python - <<'PY'
import torch
print(torch.__version__)
PY
then
  if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python - <<'PY'
import torch
PY
  then
    rm -rf .venv
    uv venv --system-site-packages
  fi
fi
UV_LINK_MODE=copy uv sync"
