#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/gpu_fleet/remote_status.sh <ssh-target> [remote-repo-dir] [ssh-port]

Example:
  tools/gpu_fleet/remote_status.sh root@1.2.3.4 /workspace/facetpy 22
USAGE
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SSH_TARGET="$1"
REMOTE_REPO="${2:-/workspace/facetpy}"
SSH_PORT="${3:-22}"

ssh -p "$SSH_PORT" "$SSH_TARGET" bash -s -- "$REMOTE_REPO" <<'REMOTE'
set -euo pipefail
REMOTE_REPO="$1"
echo "== nvidia-smi =="
nvidia-smi || true
echo
echo "== tmux sessions =="
tmux ls || true
echo
echo "== latest training runs =="
find "$REMOTE_REPO/training_output" -maxdepth 1 -mindepth 1 -type d -print 2>/dev/null | sort | tail -10 || true
echo
echo "== latest logs =="
find "$REMOTE_REPO/remote_logs" -type f -maxdepth 1 -print 2>/dev/null | sort | tail -10 || true
REMOTE
