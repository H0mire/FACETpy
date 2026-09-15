#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/gpu_fleet/fetch_runpod_results.sh <ssh-target> [remote-repo-dir] [local-root] [ssh-port]

Example:
  tools/gpu_fleet/fetch_runpod_results.sh root@1.2.3.4 /workspace/facetpy . 22

Purpose:
  Fetch training and evaluation artifacts from a RunPod into the local repo.
USAGE
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SSH_TARGET="$1"
REMOTE_REPO="${2:-/workspace/facetpy}"
LOCAL_ROOT="${3:-.}"
SSH_PORT="${4:-22}"
SSH_KEY="${FACET_GPU_FLEET_SSH_KEY:-}"
RSYNC_SSH="ssh -p $SSH_PORT -o StrictHostKeyChecking=accept-new"
if [[ -n "$SSH_KEY" ]]; then
  RSYNC_SSH="ssh -p $SSH_PORT -o StrictHostKeyChecking=accept-new -i $SSH_KEY -o IdentitiesOnly=yes"
fi

mkdir -p "$LOCAL_ROOT/training_output" "$LOCAL_ROOT/output/model_evaluations" "$LOCAL_ROOT/remote_logs"
rsync -az -e "$RSYNC_SSH" "$SSH_TARGET:$REMOTE_REPO/training_output/" "$LOCAL_ROOT/training_output/" || true
rsync -az -e "$RSYNC_SSH" "$SSH_TARGET:$REMOTE_REPO/output/model_evaluations/" "$LOCAL_ROOT/output/model_evaluations/" || true
rsync -az -e "$RSYNC_SSH" "$SSH_TARGET:$REMOTE_REPO/remote_logs/" "$LOCAL_ROOT/remote_logs/" || true
