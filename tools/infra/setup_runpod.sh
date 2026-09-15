#!/usr/bin/env bash
# Bring a fresh RunPod pod to the point where it can run facet-train.
#
# Written down because doing it by hand three times produced three different
# failures: the wrong SSH key looked like a dead pod, a missing `rsync` made
# `rsync ... 2>/dev/null` report success while copying nothing, and `pip install`
# left numpy at a different major version than the other pods.
#
# Usage:  bash tools/infra/setup_runpod.sh <ssh-alias>
#
# The alias must already exist in ~/.ssh/config with IdentityFile ~/.ssh/runpod.
# The dataset is transferred separately with transfer_to_pod.sh -- runpodctl is
# ~25x faster than rsync over the same link and needs a code handshake.
set -euo pipefail

POD="${1:?usage: setup_runpod.sh <ssh-alias>}"
cd "$(dirname "$0")/../.." || exit 1
SSH="ssh -o BatchMode=yes $POD"

echo "== $POD: Umgebung prüfen"
$SSH 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; python -c "import torch;print(\"torch\", torch.__version__, torch.cuda.is_available())"'

echo "== $POD: Abhängigkeiten (Versionen an die anderen Pods angeglichen)"
$SSH 'pip install -q --no-input "numpy==2.1.3" "scipy>=1.15" "mne==1.10.2" mne-bids neurokit2 \
        loguru rich matplotlib pandas scikit-learn 2>&1 | grep -viE "^\[notice\]" || true'

echo "== $POD: Verzeichnisse"
$SSH 'mkdir -p /workspace/facetpy/src /workspace/facetpy/tools/pipeline_demo \
              /workspace/facetpy/output/niazy_proof_fit_context_512'

# tar, not rsync: a fresh pod has no rsync, and the failure is silent when
# stderr is discarded.
#
# --no-same-owner because the archive carries this Mac's uid/gid (501/50) and
# GNU tar as root tries to honour them: "Cannot change ownership to uid 501" and
# the extraction aborts. It only bit on some pods, which is worse than always.
echo "== $POD: Quellcode"
tar czf - --exclude "__pycache__" -C src facet 2>/dev/null | $SSH 'tar xzf - --no-same-owner -C /workspace/facetpy/src'
tar czf - -C tools/pipeline_demo run_deployment_editions.sh run_deployment_editions_parallel.sh 2>/dev/null \
  | $SSH 'tar xzf - --no-same-owner -C /workspace/facetpy/tools/pipeline_demo'
tar czf - -C output/niazy_proof_fit_context_512 niazy_proof_fit_context_dataset_metadata.json holdout_v1_indices.json 2>/dev/null \
  | $SSH 'tar xzf - --no-same-owner -C /workspace/facetpy/output/niazy_proof_fit_context_512'

echo "== $POD: facet auf den Importpfad"
$SSH 'echo /workspace/facetpy/src > $(python -c "import site;print(site.getsitepackages()[0])")/facetpy.pth'
$SSH 'cd /workspace/facetpy && python -c "import facet.training.cli; print(\"facet importierbar\")"'

echo "== $POD fertig. Datensatz jetzt mit: bash tools/infra/transfer_to_pod.sh $POD <datei>"
