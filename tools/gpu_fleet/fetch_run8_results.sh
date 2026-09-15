#!/usr/bin/env bash
# Hole die Ergebnisse eines Pods ab -- die Metadaten laufend, die Gewichte zum Schluss.
#
# **Warum zweistufig.** Auf einem Pod stehen rund 4,8 GB Checkpoints gegen 4,9 MB
# Metadaten. Das Unersetzliche -- aufgeloeste Konfigurationen, Lernkurven,
# Ergebnistabellen, Protokolle -- ist ein Tausendstel des Volumens. In run 7 kam
# von den Pods nur `exports/` zurueck; die aufgeloesten Konfigurationen fehlten,
# und damit liess sich hinterher nicht mehr aus der Datei belegen, auf welchem
# Datensatz ein Modell trainiert wurde. Die Namenskonvention war eindeutig, aber
# eine Namenskonvention ist kein Nachweis.
#
# Modus `meta` ist billig genug, um ihn jede halbe Stunde zu fahren.
#
# Usage:  bash tools/gpu_fleet/fetch_run8_results.sh <ssh-alias> [meta|all]
set -uo pipefail

POD="${1:?usage: fetch_run8_results.sh <ssh-alias> [meta|weights|all]}"
MODUS="${2:-meta}"
cd "$(dirname "$0")/../.." || exit 1
ZIEL="output/run8_fetched/$POD"
REMOTE=/workspace/facetpy
mkdir -p "$ZIEL"

hole_meta() {
  # Ueber tar und ssh in einem Rutsch: 350 Einzeldateien per scp waeren 350
  # Handshakes ueber eine 182-ms-Strecke, also Minuten statt Sekunden.
  echo "== $POD: Metadaten"
  ssh -n -o BatchMode=yes "$POD" "cd $REMOTE && tar czf - \
      --ignore-failed-read \
      \$(find grids training_output logs configs -type f \
         \\( -name '*.json' -o -name '*.jsonl' -o -name '*.yaml' -o -name '*.log' \
            -o -name '*.png' -o -name '*.csv' \\) 2>/dev/null) 2>/dev/null" \
    | tar xzf - -C "$ZIEL" 2>/dev/null
  echo "   $(find "$ZIEL" -type f | wc -l | tr -d ' ') Dateien, $(du -sh "$ZIEL" | cut -f1)"
}

hole_alles() {
  # **tar, nicht rsync.** Die Gitter-Pods haben kein rsync installiert, und
  # `rsync ... 2>/dev/null` meldet dann Erfolg, ohne eine Datei zu kopieren --
  # dieselbe Falle, die in tools/infra/setup_runpod.sh schon vermerkt ist und in
  # die ich am 13.09. trotzdem gelaufen bin: gridpod2 wurde gestoppt, nachdem
  # der Abzug "erfolgreich" null Checkpoints geholt hatte.
  echo "== $POD: alles (inkl. Gewichte)"
  local vorher
  vorher=$(find "$ZIEL" -type f 2>/dev/null | wc -l | tr -d ' ')
  ssh -n -o BatchMode=yes "$POD" "cd $REMOTE && tar cf - --ignore-failed-read       grids training_output logs configs 2>/dev/null"     | tar xf - -C "$ZIEL" 2>/dev/null
  local nachher
  nachher=$(find "$ZIEL" -type f 2>/dev/null | wc -l | tr -d ' ')
  echo "   $nachher Dateien (+$((nachher - vorher))), $(du -sh "$ZIEL" | cut -f1), "        "$(find "$ZIEL" -name '*.pt' | wc -l | tr -d ' ') Checkpoints"
}

case "$MODUS" in
  meta)    hole_meta ;;
  weights) hole_alles ;;
  all)     hole_alles ;;
  *) echo "unbekannter Modus $MODUS" >&2; exit 1 ;;
esac
