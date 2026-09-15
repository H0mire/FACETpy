#!/usr/bin/env bash
# Starte die Gridsuche für eine Familie auf einem Pod, abgekoppelt vom Terminal.
#
# Eine Familie je Pod, sequenziell. Zwei Läufe passen auf eine 4090, aber sie
# konkurrieren um den Speicher -- und ein Lauf, den sein Nachbar ausgebremst hat,
# ist nicht dasselbe Experiment wie einer ohne Nachbarn. Das war schon in run 7
# die Begründung für die sequenzielle Warteschlange.
#
# nohup + setsid, damit die Suche eine abbrechende SSH-Verbindung überlebt.
#
# Usage:  bash tools/training/run_grid_pod.sh <ssh-alias> <familie> [phase] [weitere Argumente]
set -euo pipefail

POD="${1:?usage: run_grid_pod.sh <ssh-alias> <familie> [phase]}"
FAMILY="${2:?usage: run_grid_pod.sh <ssh-alias> <familie> [phase]}"
PHASE="${3:-screen}"
shift 3 2>/dev/null || shift 2
REMOTE=/workspace/facetpy
LOG="logs/grid_${FAMILY}_${PHASE}.log"

# Das ssh wird lokal abgekoppelt und sein Rückkanal verworfen. Grund: ein
# `setsid nohup ... &` über ssh startet den Prozess auf dem Pod zuverlässig,
# aber die ssh-Sitzung kehrt trotz aller Umleitungen nicht zurück -- sie wartet
# auf das Schliessen des Kanals. Wer hier auf sie wartet, wartet für immer,
# und eine Startschleife über vier Pods bleibt beim ersten stehen. Ob der Start
# geklappt hat, wird deshalb mit einer *zweiten*, kurzen Verbindung geprüft.
ssh -n -o BatchMode=yes "$POD" "cd $REMOTE && mkdir -p logs grids && \
  setsid nohup python -u tools/training/grid_search_run7.py $PHASE \
    --family $FAMILY --device cuda --pipeline-device cuda \
    --out-root grids/run8 $* > $LOG 2>&1 < /dev/null" > /dev/null 2>&1 &
sleep 8
LAEUFT=$(ssh -n -o BatchMode=yes "$POD" "ps aux | grep -c '[g]rid_search_run7'" 2>/dev/null | tail -1)
echo "gestartet: $LAEUFT Prozess(e) auf $POD"

echo "== $POD läuft $FAMILY/$PHASE"
echo "   Fortschritt:  ssh $POD 'tail -f $REMOTE/$LOG'"
echo "   Ergebnis:     $REMOTE/grids/run8/grid_${FAMILY}_${PHASE}.json"
