#!/usr/bin/env bash
# Train every deployment edition, one after the other, on one GPU.
#
# Sequential on purpose. Two 4090-sized runs fit in memory but contend for it,
# and the point of this queue is a set of comparable runs, not the fastest wall
# clock: a run that was slowed down by its neighbour is not the same experiment
# as one that was not.
#
# Usage:  bash tools/pipeline_demo/run_deployment_editions.sh [family ...]
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1
LOGS=output/deployment_editions/logs
mkdir -p "$LOGS"

ALL=(ic_unet vit_spectrogram denoise_mamba st_gnn demucs conv_tasnet sepformer
     nested_gan cascaded_dae cascaded_context_dae dpae dhct_gan_v2 dhct_gan)
FAMILIES=("${@:-${ALL[@]}}")
[ "$#" -gt 0 ] && FAMILIES=("$@")

for family in "${FAMILIES[@]}"; do
  config="src/facet/models/${family}_deployment_edition/training_niazy_proof_fit.yaml"
  if [ ! -f "$config" ]; then
    echo "SKIP  $family (no config at $config)" | tee -a "$LOGS/queue.log"
    continue
  fi
  echo "=== $(date -u +%H:%M:%S)  START $family" | tee -a "$LOGS/queue.log"
  python -m facet.training.cli fit --config "$config" > "$LOGS/${family}.log" 2>&1
  status=$?
  best=$(grep -m1 "Best metric:" "$LOGS/${family}.log" | tr -d '\r')
  echo "=== $(date -u +%H:%M:%S)  DONE  $family rc=$status  $best" | tee -a "$LOGS/queue.log"
done

echo "=== queue finished" | tee -a "$LOGS/queue.log"
