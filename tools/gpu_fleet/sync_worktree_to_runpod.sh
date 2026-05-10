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
rsync -az --delete \
  -e "$RSYNC_SSH" \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '.pytest_cache/' \
  --exclude '__pycache__/' \
  --exclude 'htmlcov/' \
  --exclude 'output/' \
  --exclude 'training_output/' \
  --exclude '.DS_Store' \
  "$LOCAL_WORKTREE/" "$SSH_TARGET:$REMOTE_REPO/"

ssh "${SSH_ARGS[@]}" "$SSH_TARGET" "cd '$REMOTE_REPO' && uv sync"
