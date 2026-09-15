#!/usr/bin/env bash
# Sichert laufend, stoppt jeden Pod einzeln, sobald er nachweislich fertig ist.
#
# **Gestoppt, nicht terminiert.** `stop` beendet die GPU-Abrechnung und laesst das
# Volume stehen; `remove` loescht es. Solange nur die besten drei Punkte je
# Familie als Gewichte heruntergeladen sind, waere Terminieren unumkehrbarer
# Datenverlust. Terminieren bleibt eine Handbewegung des Nutzers.
#
# **Warum nicht einfach "keine Prozesse mehr".** Ein Pod, dessen Lauf abgestuerzt
# ist, zeigt ebenfalls keine Prozesse -- und wuerde gestoppt, bevor jemand merkt,
# dass die letzten Punkte fehlen. Deshalb drei Bedingungen, und bei Zweifel wird
# gemeldet statt gestoppt:
#
#   1. zweimal im Abstand von 10 Minuten keine Suchprozesse
#   2. die Ergebnisdatei enthaelt die erwartete Zahl Punkte
#   3. die lokale Kopie hat ebenso viele Zeilen wie die auf dem Pod
#
# Usage:  bash tools/gpu_fleet/stop_when_done.sh [erwartete_punkte]
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
ERWARTET="${1:-27}"
KARTE=/tmp/podkarte.txt

zeilen_auf_pod() {   # $1 alias -> Summe der Zeilen ueber alle Ergebnisdateien
  ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$1" "cd /workspace/facetpy 2>/dev/null && python3 -c \"
import json,glob
n=0
for f in glob.glob('grids/*/grid_*_screen.json'):
    try: n += len(json.load(open(f))['zeilen'])
    except Exception: pass
print(n)\"" 2>/dev/null | tail -1
}
zeilen_lokal() {     # $1 alias
  python3 - "$1" <<'PY'
import json, sys
from pathlib import Path
n = 0
for f in Path(f"output/run8_fetched/{sys.argv[1]}").rglob("grid_*_screen.json"):
    try: n += len(json.load(open(f))["zeilen"])
    except Exception: pass
print(n)
PY
}
prozesse() { ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$1" \
  "ps aux | grep -cE '[g]rid_search_run7|[f]acet.training.cli'" 2>/dev/null | tail -1; }

declare -a STILL=()
for runde in $(seq 1 120); do   # bis zu 40 Stunden
  offen=0
  while IFS=' ' read -r alias id; do
    [ -z "${alias:-}" ] && continue
    status=$(runpodctl pod get "$id" 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin).get('runtimeStatus','?'))" 2>/dev/null)
    [ "$status" != "running" ] && continue
    offen=$((offen+1))
    bash tools/gpu_fleet/fetch_run8_results.sh "$alias" meta > /dev/null 2>&1
    n=$(prozesse "$alias"); n=${n:-1}
    if [ "$n" -gt 0 ]; then STILL[$offen]=0; continue; fi

    # Bedingung 1: zweite Beobachtung, zehn Minuten spaeter
    if [ "${STILL[$offen]:-0}" -eq 0 ]; then
      STILL[$offen]=1
      echo "$(date '+%H:%M')  $alias: still -- pruefe in 10 min erneut"
      continue
    fi

    auf=$(zeilen_auf_pod "$alias"); lok=$(zeilen_lokal "$alias")
    if [ "${auf:-0}" -lt "$ERWARTET" ]; then
      echo "!! $alias: nur ${auf:-0} Punkte statt $ERWARTET -- NICHT gestoppt, bitte ansehen"
      continue
    fi
    bash tools/gpu_fleet/fetch_run8_results.sh "$alias" all > /dev/null 2>&1
    lok=$(zeilen_lokal "$alias")
    if [ "${lok:-0}" -ne "${auf:-0}" ]; then
      echo "!! $alias: lokal $lok Zeilen, auf dem Pod $auf -- NICHT gestoppt"
      continue
    fi
    echo "== $alias ($id): $auf Punkte, lokal vollstaendig -> stoppe"
    runpodctl pod stop "$id" 2>&1 | tail -1
  done < "$KARTE"

  echo "$(date '+%d.%m. %H:%M')  Runde $runde: $offen Pods laufen, $(du -sh output/run8_fetched 2>/dev/null | cut -f1) gesichert"
  [ "$offen" -eq 0 ] && { echo "alle Pods gestoppt"; break; }
  sleep 600
done
echo "=== WAECHTER FERTIG"
