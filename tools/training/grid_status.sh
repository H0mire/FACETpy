#!/usr/bin/env bash
# Der Stand aller vier Gitter auf einen Blick.
#
# Usage:  bash tools/training/grid_status.sh
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

for zeile in "gridpod1 nested_gan" "gridpod2 vit_spectrogram" "gridpod3 dhct_gan" "gridpod4 demucs"; do
  set -- $zeile
  pod="$1"; family="$2"
  echo "── $pod / $family"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$pod" "
    cd /workspace/facetpy 2>/dev/null || { echo '  nicht eingerichtet'; exit 0; }
    # Die Klammer verhindert, dass grep sich selbst findet -- pgrep kann das nicht.
    lauf=\$(ps aux | grep '[g]rid_search_run7' | wc -l | tr -d ' ')
    echo \"  Prozesse: \$lauf\"
    for f in grids/run8/grid_${family}_*.json; do
      [ -f \"\$f\" ] || continue
      python - \"\$f\" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
z = d['zeilen']
ok = [r for r in z if r.get('status') == 'ok']
print(f\"  {sys.argv[1].split('/')[-1]}: {len(z)} gelaufen, {len(ok)} gewertet\")
for r in sorted(ok, key=lambda r: r['ga_rest_uv'])[:3]:
    print(f\"    {r['tag']:34s} GA-Rest {r['ga_rest_uv']:6.2f} µV  Naht {r['naht_ratio']:.2f}  val_loss {r.get('val_loss', float('nan')):8.4f}\")
schlecht = [r for r in z if r.get('status') not in ('ok', None)]
for r in schlecht[:3]:
    print(f\"    {r['tag']:34s} {r['status']}: {str(r.get('fehler'))[:90]}\")
PY
    done
    tail -2 logs/grid_${family}_*.log 2>/dev/null | sed 's/^/  /' | tail -4
  " 2>&1 | grep -v "^Warning: Permanently added"
done
