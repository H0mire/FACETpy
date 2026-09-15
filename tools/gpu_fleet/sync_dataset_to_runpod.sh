#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/gpu_fleet/sync_dataset_to_runpod.sh <local-path> <ssh-target> <remote-path> [ssh-port]

Example:
  tools/gpu_fleet/sync_dataset_to_runpod.sh \
    output/my_model_dataset \
    root@1.2.3.4 \
    /workspace/facetpy/output/my_model_dataset \
    22

Purpose:
  Upload a generated dataset or artifact library to a RunPod without changing code.
  Use this when a model agent builds a local dataset that is intentionally excluded
  from normal worktree sync.
USAGE
}

if [[ $# -lt 3 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

LOCAL_PATH="$1"
SSH_TARGET="$2"
REMOTE_PATH="$3"
SSH_PORT="${4:-22}"
SSH_KEY="${FACET_GPU_FLEET_SSH_KEY:-}"
SSH_ARGS=(-p "$SSH_PORT" -o StrictHostKeyChecking=accept-new)
RSYNC_SSH="ssh -p $SSH_PORT -o StrictHostKeyChecking=accept-new"
if [[ -n "$SSH_KEY" ]]; then
  SSH_ARGS+=(-i "$SSH_KEY" -o IdentitiesOnly=yes)
  RSYNC_SSH="ssh -p $SSH_PORT -o StrictHostKeyChecking=accept-new -i $SSH_KEY -o IdentitiesOnly=yes"
fi

if [[ ! -e "$LOCAL_PATH" ]]; then
  echo "Local dataset path not found: $LOCAL_PATH" >&2
  exit 1
fi

ssh "${SSH_ARGS[@]}" "$SSH_TARGET" "mkdir -p '$(dirname "$REMOTE_PATH")'"

if [[ -d "$LOCAL_PATH" ]]; then
  ssh "${SSH_ARGS[@]}" "$SSH_TARGET" "mkdir -p '$REMOTE_PATH'"
  rsync -az --delete --no-owner --no-group -e "$RSYNC_SSH" "$LOCAL_PATH/" "$SSH_TARGET:$REMOTE_PATH/"
else
  ssh "${SSH_ARGS[@]}" "$SSH_TARGET" "mkdir -p '$(dirname "$REMOTE_PATH")'"
  rsync -az --no-owner --no-group -e "$RSYNC_SSH" "$LOCAL_PATH" "$SSH_TARGET:$REMOTE_PATH"
fi
