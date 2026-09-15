#!/usr/bin/env bash
# Train several deployment editions at once on one GPU.
#
# One run occupies ~1.3-2.4 GB of a 24.5 GB card and drives it to roughly a
# quarter of its capacity: the models are small (0.14-16.6 M parameters) and the
# sequences short (512 samples), so the kernels do not fill the device. Running
# them one after another leaves most of the card idle for hours.
#
# Concurrency is safe for the *results*. Contention changes how long a run takes,
# not what it computes -- the seed, the data and the order of batches are
# unaffected. Wall-clock per run is not one of the numbers this project reports,
# so there is nothing to protect by serialising.
#
# Usage:  bash run_deployment_editions_parallel.sh [-j N] family [family ...]
#         CONFIG_SUFFIX=_long bash run_deployment_editions_parallel.sh ...
#
# CONFIG_SUFFIX picks a config variant, e.g. `_long` for the raised epoch ceiling.
set -uo pipefail

JOBS=4
while getopts "j:" opt; do
  case $opt in
    j) JOBS=$OPTARG ;;
    *) echo "usage: $0 [-j N] family ..." >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
[ "$#" -eq 0 ] && { echo "usage: $0 [-j N] family ..." >&2; exit 2; }

cd "$(dirname "$0")/../.." || exit 1
LOGS=output/deployment_editions/logs
mkdir -p "$LOGS"

# Without this each of N processes asks for all 96 cores and they thrash each
# other on the CPU side of batch assembly.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

run_one() {
  family="$1"
  suffix="${CONFIG_SUFFIX:-}"
  config="src/facet/models/${family}_deployment_edition/training_niazy_proof_fit${suffix}.yaml"
  logs=output/deployment_editions/logs
  # The suffix belongs in the log name. Two variants of one family running at
  # once otherwise write to the same file, truncate each other, and the "Best
  # metric" grep reads whatever survived -- which is how a finished run once
  # reported rc=0 with no metric at all.
  label="${family}${suffix}"
  if [ ! -f "$config" ]; then
    echo "SKIP  $label (kein Config unter $config)" >> "$logs/queue.log"
    return 0
  fi
  echo "=== $(date -u +%H:%M:%S)  START $label" >> "$logs/queue.log"
  python -m facet.training.cli fit --config "$config" > "$logs/${label}.log" 2>&1
  status=$?
  best=$(grep -m1 "Best metric:" "$logs/${label}.log" | tr -d '\r')
  echo "=== $(date -u +%H:%M:%S)  DONE  $label rc=$status  $best" >> "$logs/queue.log"
}
export -f run_one
export CONFIG_SUFFIX="${CONFIG_SUFFIX:-}"

printf '%s\n' "$@" | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}
echo "=== $(date -u +%H:%M:%S)  parallel queue finished" >> "$LOGS/queue.log"
