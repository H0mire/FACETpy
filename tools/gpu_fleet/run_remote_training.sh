#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/gpu_fleet/run_remote_training.sh <ssh-target> <config-path> [remote-repo-dir] [session-name] [ssh-port] [gpu-id] [prepare-command]

Example:
  tools/gpu_fleet/run_remote_training.sh root@1.2.3.4 src/facet/models/cascaded_context_dae/training.yaml /workspace/facetpy train_context 22 0

Purpose:
  Start a FACETpy training job inside tmux on a RunPod. One lock per GPU prevents
  accidental double-booking.
USAGE
}

if [[ $# -lt 2 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SSH_TARGET="$1"
CONFIG_PATH="$2"
REMOTE_REPO="${3:-/workspace/facetpy}"
SESSION_NAME="${4:-facet_train}"
SSH_PORT="${5:-22}"
GPU_ID="${6:-0}"
PREPARE_COMMAND=""
if [[ $# -gt 6 ]]; then
  shift 6
  PREPARE_COMMAND="$*"
fi
PREPARE_COMMAND_B64="$(printf '%s' "$PREPARE_COMMAND" | base64 | tr -d '\n')"
SSH_KEY="${FACET_GPU_FLEET_SSH_KEY:-}"
SSH_ARGS=(-p "$SSH_PORT" -o StrictHostKeyChecking=accept-new)
if [[ -n "$SSH_KEY" ]]; then
  SSH_ARGS+=(-i "$SSH_KEY" -o IdentitiesOnly=yes)
fi

ssh "${SSH_ARGS[@]}" "$SSH_TARGET" bash -s -- "$REMOTE_REPO" "$CONFIG_PATH" "$SESSION_NAME" "$GPU_ID" "$PREPARE_COMMAND_B64" <<'REMOTE'
set -euo pipefail
REMOTE_REPO="$1"
CONFIG_PATH="$2"
SESSION_NAME="$3"
GPU_ID="$4"
PREPARE_COMMAND_B64="$5"
PREPARE_COMMAND="$(printf '%s' "$PREPARE_COMMAND_B64" | base64 -d)"
LOCK_FILE="/tmp/facetpy_gpu_${GPU_ID}.lock"
LOG_DIR="$REMOTE_REPO/remote_logs"
RUN_SCRIPT="$LOG_DIR/${SESSION_NAME}.sh"
RUNNER_SCRIPT="$LOG_DIR/${SESSION_NAME}.runner.sh"
EXITCODE_FILE="$LOG_DIR/${SESSION_NAME}.exitcode"
mkdir -p "$LOG_DIR"

cd "$REMOTE_REPO"
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  printf "cd %q\n" "$REMOTE_REPO"
  echo 'echo "== sync deps =="'
  echo 'UV_LINK_MODE=copy uv sync'
  echo 'echo "== verify torch =="'
  echo 'uv run python tools/gpu_fleet/check_torch.py'
  if [[ -n "$PREPARE_COMMAND" ]]; then
    echo 'echo "== prepare =="'
    printf '%s\n' "$PREPARE_COMMAND"
  fi
  echo 'echo "== train =="'
  printf 'CUDA_VISIBLE_DEVICES=%q uv run facet-train fit --config %q\n' "$GPU_ID" "$CONFIG_PATH"
} > "$RUN_SCRIPT"
chmod +x "$RUN_SCRIPT"

{
  echo '#!/usr/bin/env bash'
  echo 'set -o pipefail'
  printf "cd %q\n" "$REMOTE_REPO"
  printf "flock -n %q bash %q 2>&1 | tee %q\n" "$LOCK_FILE" "$RUN_SCRIPT" "$LOG_DIR/${SESSION_NAME}.log"
  echo 'code=${PIPESTATUS[0]}'
  printf 'echo "$code" > %q\n' "$EXITCODE_FILE"
  echo 'exit "$code"'
} > "$RUNNER_SCRIPT"
chmod +x "$RUNNER_SCRIPT"
rm -f "$EXITCODE_FILE"

tmux new-session -d -s "$SESSION_NAME" \
  "bash '$RUNNER_SCRIPT'"

echo "Started tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "Log: $LOG_DIR/${SESSION_NAME}.log"
echo "Run script: $RUN_SCRIPT"
echo "Exit code: $EXITCODE_FILE"
REMOTE
