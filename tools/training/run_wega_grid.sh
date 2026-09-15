#!/usr/bin/env bash
# Starte das Weg-A-Gitter: vier Familien, je 27 Punkte, je ein MIG-Slice.
#
# Wie run_wega_pod.sh holt es die MIG-UUIDs vom Pod, weil CUDA von jeder
# physischen Karte nur die erste Instanz zeigt und der vierte Slice sonst
# unerreichbar bleibt.
#
# Usage:  bash tools/training/run_wega_grid.sh <ssh-alias> [phase]
set -uo pipefail

POD="${1:?usage: run_wega_grid.sh <ssh-alias> [phase]}"
PHASE="${2:-screen}"
REMOTE=/workspace/facetpy

MIG=()
while IFS= read -r z; do [ -n "$z" ] && MIG+=("$z"); done \
  < <(ssh -n -o BatchMode=yes "$POD" "nvidia-smi -L | grep -o 'MIG-[0-9a-f-]*'")
[ "${#MIG[@]}" -ge 4 ] || { echo "nur ${#MIG[@]} MIG-Instanzen" >&2; exit 1; }

FAMILIEN=(nested_gan vit_spectrogram demucs dhct_gan)
for i in "${!FAMILIEN[@]}"; do
  fam="${FAMILIEN[$i]}"
  ssh -n -o BatchMode=yes "$POD" "cd $REMOTE && mkdir -p logs grids && \
    CUDA_VISIBLE_DEVICES=${MIG[$i]} setsid nohup python -u tools/training/grid_search_run7.py $PHASE \
      --family $fam --dataset wega --device cuda --pipeline-device cuda \
      --out-root grids/wega > logs/grid_wega_${fam}_${PHASE}.log 2>&1 < /dev/null" > /dev/null 2>&1 &
  echo "== $fam -> ${MIG[$i]:0:22}…"
  sleep 5
done

sleep 25
N=$(ssh -n -o BatchMode=yes "$POD" "ps aux | grep -c '[g]rid_search_run7'" 2>/dev/null | tail -1)
echo "gestartet: $N Prozess(e)"
