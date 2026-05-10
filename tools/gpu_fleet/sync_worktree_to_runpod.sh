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

if [[ ! -d "$LOCAL_WORKTREE" ]]; then
  echo "Local worktree not found: $LOCAL_WORKTREE" >&2
  exit 1
fi

ssh -p "$SSH_PORT" "$SSH_TARGET" "mkdir -p '$REMOTE_REPO'"
rsync -az --delete \
  -e "ssh -p $SSH_PORT" \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '.pytest_cache/' \
  --exclude '__pycache__/' \
  --exclude 'htmlcov/' \
  --exclude 'output/' \
  --exclude 'training_output/' \
  --exclude '.DS_Store' \
  "$LOCAL_WORKTREE/" "$SSH_TARGET:$REMOTE_REPO/"

ssh -p "$SSH_PORT" "$SSH_TARGET" "cd '$REMOTE_REPO' && uv sync"
