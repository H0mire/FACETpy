#!/usr/bin/env bash
# Starte die vier Weg-A-Läufe auf einem Pod mit vier MIG-Slices.
#
# **Warum die Geräte nicht einfach 0..3 heissen.** Die drei physischen Karten
# tragen vier MIG-Instanzen, und CUDA zeigt von jeder Karte nur die *erste*. Der
# vierte Slice sitzt als zweite Instanz auf GPU 2 und ist nur ueber seine
# MIG-UUID ansprechbar. Wer hier stumpf CUDA_VISIBLE_DEVICES=0,1,2,3 setzt,
# bekommt fuer den vierten Lauf kein Geraet -- oder, schlimmer, dasselbe wie fuer
# den ersten.
#
# **Der Datensatz kommt von der lokalen Platte, nicht aus /workspace.** Das
# Volume ist ein Netzlaufwerk (MooseFS ueber FUSE, gemessen 837 MB/s gegen
# 5,0 GB/s lokal), und jeder der vier Prozesse liest beim Start 1,33 GB ein.
# /workspace bleibt der persistente Ablageort, /data der Arbeitsort.
#
# Usage:  bash tools/training/run_wega_pod.sh <ssh-alias>
set -uo pipefail

POD="${1:?usage: run_wega_pod.sh <ssh-alias>}"
REMOTE=/workspace/facetpy
LOKAL=/data/weg_a_farm_v10_locked_1ch/weg_a_spatiotemporal_dataset.npz

# Die MIG-UUIDs vom Pod holen statt sie einzutragen: sie aendern sich mit jedem
# neuen Pod, und eine veraltete UUID ist ein stiller Fehlstart.
# mapfile gibt es erst ab bash 4; macOS liefert 3.2. Deshalb per Schleife.
MIG=()
while IFS= read -r zeile; do
  [ -n "$zeile" ] && MIG+=("$zeile")
done < <(ssh -n -o BatchMode=yes "$POD" "nvidia-smi -L | grep -o 'MIG-[0-9a-f-]*'")
if [ "${#MIG[@]}" -lt 4 ]; then
  echo "nur ${#MIG[@]} MIG-Instanzen gefunden, erwartet wurden 4" >&2; exit 1
fi
echo "MIG-Instanzen: ${#MIG[@]}"

FAMILIEN=(nested_gan vit_spectrogram demucs dhct_gan)
for i in "${!FAMILIEN[@]}"; do
  fam="${FAMILIEN[$i]}"
  uuid="${MIG[$i]}"
  echo "== $fam auf ${uuid:0:20}…"
  ssh -n -o BatchMode=yes "$POD" "cd $REMOTE && mkdir -p logs && \
    python - <<PY
import yaml
from pathlib import Path
p = Path('configs/run8_wega_$fam.yaml')
kopf, _, rest = p.read_text().partition('model:')
c = yaml.safe_load('model:' + rest)
c['data']['kwargs']['path'] = '$LOKAL'
c['training']['output_dir'] = '$REMOTE/training_output'
Path('configs/run8_wega_${fam}.resolved.yaml').write_text(kopf + yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
PY
    CUDA_VISIBLE_DEVICES=$uuid setsid nohup python -u -m facet.training.cli fit \
      --config configs/run8_wega_${fam}.resolved.yaml \
      > logs/wega_${fam}.log 2>&1 < /dev/null" > /dev/null 2>&1 &
  sleep 6
done

sleep 20
echo
LAEUFT=$(ssh -n -o BatchMode=yes "$POD" "ps aux | grep -c '[f]acet.training.cli'" 2>/dev/null | tail -1)
echo "gestartet: $LAEUFT Prozess(e) auf $POD"
echo "Fortschritt:  ssh $POD 'tail -f $REMOTE/logs/wega_<familie>.log'"
