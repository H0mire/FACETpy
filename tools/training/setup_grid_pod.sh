#!/usr/bin/env bash
# Bringe einen leeren Pod auf den Stand, auf dem die Gridsuche läuft.
#
# Baut auf tools/infra/setup_runpod.sh auf und ergänzt, was run 8 zusätzlich
# braucht: die Werkzeuge unter tools/, die Aufnahme selbst und ein Verzeichnis
# für die Gitterläufe. Ohne die EDF gibt es keine Pipeline-Bewertung -- und die
# ist der ganze Punkt dieser Suche, siehe docs/research/run_8_gridsuche_hyperparameter.md §4.
#
# Kein venv: die Pods bringen torch 2.4.1+cu124 im Systeminterpreter mit, und ein
# eigenes venv würde entweder ein zweites torch ziehen (mehrere GB auf einer
# 20-GB-Platte) oder über --system-site-packages ohnehin dasselbe benutzen.
# Stattdessen eine .pth-Datei auf src/, so wie in run 7.
#
# tar über ssh statt rsync: ein frischer Pod hat kein rsync, und der Fehlschlag
# ist still, wenn stderr verworfen wird.
#
# Usage:  bash tools/training/setup_grid_pod.sh <ssh-alias>
set -euo pipefail

POD="${1:?usage: setup_grid_pod.sh <ssh-alias>}"
cd "$(dirname "$0")/../.." || exit 1
SSH="ssh -o BatchMode=yes $POD"
REMOTE=/workspace/facetpy

echo "== $POD: Umgebung"
$SSH 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch;print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())"
df -h /workspace | tail -1'

echo "== $POD: Abhängigkeiten (Versionen an run 7 angeglichen)"
$SSH 'pip install -q --no-input "numpy==2.1.3" "scipy>=1.15" "mne==1.10.2" mne-bids neurokit2 \
        loguru rich matplotlib pandas scikit-learn pyyaml edfio 2>&1 | grep -viE "^\[notice\]" || true'

echo "== $POD: Verzeichnisse"
$SSH "mkdir -p $REMOTE/src $REMOTE/tools $REMOTE/examples/datasets \
              $REMOTE/output/niazy_proof_fit_context_512 $REMOTE/grids $REMOTE/logs"

echo "== $POD: Quellcode"
# --no-same-owner: das Archiv trägt die uid/gid dieses Macs (501/50), GNU tar als
# root will sie ehren und bricht mit "Cannot change ownership to uid 501" ab.
tar czf - --exclude "__pycache__" -C src facet | $SSH "tar xzf - --no-same-owner -C $REMOTE/src"
tar czf - --exclude "__pycache__" -C . tools | $SSH "tar xzf - --no-same-owner -C $REMOTE"
tar czf - -C output/niazy_proof_fit_context_512 \
      niazy_proof_fit_context_dataset_metadata.json holdout_v1_indices.json \
  | $SSH "tar xzf - --no-same-owner -C $REMOTE/output/niazy_proof_fit_context_512"

echo "== $POD: die Aufnahme (22 MB, ohne sie keine Bewertung)"
tar czf - -C examples/datasets NiazyFMRI.edf | $SSH "tar xzf - --no-same-owner -C $REMOTE/examples/datasets"

echo "== $POD: facet auf den Importpfad"
$SSH "echo $REMOTE/src > \$(python -c 'import site;print(site.getsitepackages()[0])')/facetpy.pth"
$SSH "cd $REMOTE && python -c 'import facet.training.cli, mne, yaml; print(\"facet importierbar\")'"

echo "== $POD fertig."
echo "   Datensatz jetzt mit: bash tools/infra/transfer_to_pod.sh $POD output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz"
