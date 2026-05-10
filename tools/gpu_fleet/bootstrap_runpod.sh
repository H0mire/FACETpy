#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/gpu_fleet/bootstrap_runpod.sh <ssh-target> <repo-url> [remote-repo-dir] [ssh-port]

Example:
  tools/gpu_fleet/bootstrap_runpod.sh root@1.2.3.4 git@github.com:org/facetpy.git /workspace/facetpy 22

Purpose:
  Prepare a RunPod PyTorch/Jupyter pod for FACETpy training.
USAGE
}

if [[ $# -lt 2 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SSH_TARGET="$1"
REPO_URL="$2"
REMOTE_REPO="${3:-/workspace/facetpy}"
SSH_PORT="${4:-22}"

ssh -p "$SSH_PORT" "$SSH_TARGET" bash -s -- "$REPO_URL" "$REMOTE_REPO" <<'REMOTE'
set -euo pipefail
REPO_URL="$1"
REMOTE_REPO="$2"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

mkdir -p "$(dirname "$REMOTE_REPO")"
if [[ ! -d "$REMOTE_REPO/.git" ]]; then
  git clone "$REPO_URL" "$REMOTE_REPO"
fi

cd "$REMOTE_REPO"
uv sync
python - <<'PY'
try:
    import torch
    print('torch:', torch.__version__)
    print('cuda_available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('gpu:', torch.cuda.get_device_name(0))
except Exception as exc:
    print('torch_check_failed:', repr(exc))
PY
REMOTE
