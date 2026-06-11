# Run 4 — Fleet-Training der `_paper_accurate_edition`-Modelle

> **Art dieses Runs:** Ops-/Trainings-Run auf der GPU-Fleet (nicht eine
> Forschungsrichtung wie `run_3_plan.md`). Ziel ist, die 12 neuen
> paper-getreuen Modell-Editionen unter denselben Bedingungen wie der letzte
> Fleet-Run zu trainieren, zu exportieren, zu evaluieren — und **A/B gegen die
> Originale** aus dem Run vom 10.05.2026 zu stellen.
>
> **Vorlage / Tooling:** `docs/deep_learning_parallel_runpod_workflow.md`
> (Operator-Guide), `tools/gpu_fleet/fleet.py` (lokaler Scheduler).

---

## 0. TL;DR (Kickoff in 6 Schritten)

1. **CUDA-Full-Configs erzeugen** — jede Edition hat aktuell nur ein
   `device: cpu`-Smoke-Config (`max_epochs: 1`). Das ist die eigentliche
   Vorbereitungsarbeit (→ §3.1).
2. **Pods provisionieren** und `tools/gpu_fleet/workers.local.yaml`
   aktualisieren (alte RunPod-Hosts sind tot; aktuell nur `gpu1` eingetragen).
3. **Editions committen/snapshotten** (sie sind derzeit untracked) →
   reproduzierbarer Sync.
4. **Bootstrap** beider Pods, **Dataset** einmal pro Pod bauen
   (`--prepare-command`).
5. **Submit Smoke → Full** je Edition, `dispatch --loop`, `fetch`.
6. **Evaluieren** und gegen die Original-Editionen vergleichen (`metrics.json`).

---

## 1. Rückblick: der letzte Fleet-Run (10.05.2026)

Aus `.facet_gpu_fleet/queue.json` rekonstruiert (40 Jobs, 2 GPUs, ~16:55–22:59):

- **Muster:** pro Modell zuerst `<model>_niazy_smoke`, dann
  `<model>_niazy_full`. Smoke verifiziert Sync + Dataset-Build + CUDA-`uv run`
  + Checkpoint/`loss.png`/TorchScript-Export, bevor das Full-Training läuft.
- **Topologie:** `gpu1` + `gpu2` (RunPod, single-GPU pro Pod), MacBook als
  Dispatcher. Jobs warten `pending`, bis ein Worker frei ist.
- **Trainiert (finished):** alle 12 paper-basierten Modelle (conv_tasnet, demucs,
  sepformer, d4pm, dpae, ic_unet, st_gnn, vit_spectrogram, dhct_gan,
  dhct_gan_v2, denoise_mamba, nested_gan) sowie die beiden Prototypen
  (cascaded_dae, cascaded_context_dae) als Smoke.
- **Flaky / Lehren fürs nächste Mal:**
  - `d4pm` brauchte **3 Smoke-Retries** (`_v2`/`_v3`/`_v4`) bevor es lief.
  - `nested_gan` brauchte **1 Retry** (`_v2`).
  - `dhct_gan` Smoke einmal `cancelled`, dann ok.
  - 2 Jobs blieben `pending` und liefen nie:
    `cascaded_dae_niazy_full`, `cascaded_context_dae_niazy_full`.

> **Konsequenz:** Retry-Budget für `d4pm` und `nested_gan` einplanen; GAN-Smokes
> nicht voreilig abbrechen.

---

## 2. Ausgangslage der neuen Editionen

12 paper-getreue Editionen unter `src/facet/models/<id>_paper_accurate_edition/`
(je 6 Dateien) plus 3 reine `NOTE.md`-Ordner für die paperlosen Prototypen
(`cascaded_dae`, `cascaded_context_dae`, `demo01` → **kein Fleet-Job**).

Lokal verifiziert: 81 Tests grün in 6,7 s (CPU), alle 12 Editionen importieren
kollisionsfrei (eindeutige Processor-Namen `<id>_paper_accurate_correction`).

**Was für die Fleet noch fehlt / zu beachten ist:**

| Punkt | Status heute | Aktion |
|---|---|---|
| CUDA-Full-Config je Edition | ❌ existiert nicht | §3.1 — pro Edition erzeugen |
| CUDA-Smoke-Config | ⚠️ nur `device: cpu` | §3.1 — `device: cuda`-Smoke ergänzen |
| Editionen committed? | ❌ untracked | §3.3 — committen/snapshotten |
| Worker-Config | ⚠️ nur `gpu1`, Host vermutlich tot | §3.2 — neu provisionieren |
| Dataset auf Pod | ❌ | §3.4 — `--prepare-command` |

---

## 3. Vorbereitungs-Checkliste

### 3.1 CUDA-Configs erzeugen (Kern-Vorbereitung)

Jede Edition braucht zwei Configs analog zu den Originalen
(`src/facet/models/<id>/training_niazy_proof_fit{,_smoke}.yaml`):

1. **`training_niazy_proof_fit.yaml`** (Full, `device: cuda`, paper-skalige
   `model.kwargs`, realistisches `max_epochs`, kein `max_examples`-Cap).
2. **`training_niazy_proof_fit_cuda_smoke.yaml`** (1 Epoche, `device: cuda`,
   kleines `max_examples`) — der bestehende `_smoke.yaml` bleibt als
   **CPU-Smoke** unverändert (M4-tauglich).

**Schnellster, korrekter Weg:** Original-Full-Config der jeweiligen
Edition kopieren und **nur die drei Factory-Pfade** auf das
`_paper_accurate_edition`-Modul umbiegen, dann `model.kwargs` auf die
paper-getreuen Parameter aus `documentation/paper_accuracy_review.md`
anpassen. Die `data`/`training`/`export`-Blöcke bleiben weitgehend gleich.

```yaml
model:
  framework: pytorch
  factory: facet.models.<id>_paper_accurate_edition.training:build_model
  kwargs: { ... paper-getreue Werte ... }
  loss_factory: facet.models.<id>_paper_accurate_edition.training:build_loss
  loss_kwargs: { ... }
  device: cuda
data:
  dataset_factory: facet.models.<id>_paper_accurate_edition.training:build_dataset
  kwargs:
    path: ./output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz
  eeg_only: true
training:
  model_name: <Id>PaperAccurate
  chunk_size: 512
  target_type: artifact   # bzw. clean — je Edition prüfen!
  max_epochs: <full>
  device-relevante Felder ...
export:
  enabled: true
  format: torchscript     # Export via torch.jit.trace (siehe §6)
```

> ⚠️ `vit_spectrogram`: `build_loss` bekommt vom Trainer **keine** Modell-Dims
> injiziert — STFT/Patch/Mask-Geometrie muss in `loss_kwargs` **identisch** zu
> `model.kwargs` gesetzt werden, sonst trainiert es gegen die falschen Patches.
> Zur Laufzeit `model.masked_index == loss.masked_index` prüfen.

> ⚠️ `dhct_gan_v2` referenziert im CPU-Smoke das `context_64`-Dataset; die
> übrigen Editionen `context_512`. Vor dem Submit den `path` je Config
> verifizieren.

### 3.2 Pods provisionieren & Worker-Config

RunPod-Instanzen sind ephemer — die Hosts vom 10.05. sind mit hoher
Wahrscheinlichkeit weg. Aktuell ist nur **`gpu1`** in `workers.local.yaml`.

1. 1–2 RunPod-PyTorch/CUDA-Pods starten.
2. `tools/gpu_fleet/workers.local.yaml` mit den neuen `ssh`/`port`/
   `identity_file` für `gpu1` (+ optional `gpu2`) aktualisieren (nicht committen).
3. Bootstrap je Pod:

```bash
tools/gpu_fleet/bootstrap_runpod.sh root@<host> https://github.com/H0mire/FACETpy.git /workspace/facetpy <port>
```

   Bei CUDA-Image-Torch nutzt der Worker `uv venv --system-site-packages`.
   **`st_gnn`** braucht `torch-geometric` passend zur CUDA-Torch-Version —
   nach Bootstrap mit `tools/gpu_fleet/check_torch.py` und einem PyG-Import-Test
   verifizieren (Provisioning-Risiko, siehe §5).

### 3.3 Editionen reproduzierbar machen

Die Editionen sind untracked. `sync_worktree_to_runpod.sh` (rsync) synct auch
untracked Dateien, aber für Reproduzierbarkeit:

- **Empfohlen:** Editionen + neue Configs auf `feature/add-deeplearning`
  committen (nur eigene Dateien stagen, **kein** `git add -A` —
  `examples/`-Änderungen gehören einem anderen Agenten). *Commit nur nach
  ausdrücklicher Freigabe.*
- **Alternative:** uncommitted `--worktree .` synchronisieren und den
  Commit-Hash/`git stash`-Snapshot im Run-Log notieren.

### 3.4 Dataset

Einmal pro Pod via `--prepare-command` (läuft nach Sync, vor `facet-train`):

```
uv run python examples/dataset_building/build_niazy_proof_fit_context_dataset.py \
  --artifact-bundle output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz \
  --target-epoch-samples 512 --context-epochs 7 \
  --output-dir output/niazy_proof_fit_context_512
```

Für `dhct_gan_v2` (falls Full auf `context_64` bleibt) zusätzlich mit
`--target-epoch-samples 64 --output-dir output/niazy_proof_fit_context_64`.
Das Artifact-Bundle liegt lokal (77 MB) und kann sonst per
`sync_dataset_to_runpod.sh` hochgeladen werden.

---

## 4. Job-Matrix (12 Editionen)

`<edn>` = `<id>_paper_accurate_edition`. Pro Edition: erst Smoke, dann Full.
Retry-Budget aus dem letzten Run abgeleitet.

| # | Modell | Dataset | Retry-Budget | Sonderfall (§5) |
|---|---|---|---|---|
| 1 | conv_tasnet | 512/7 | niedrig | linearer Encoder + Sigmoid-Maske |
| 2 | demucs | 512/7 | niedrig | length-agnostic forward; Resampling-Trick |
| 3 | sepformer | 512/7 | niedrig | kompakter als Paper (dokumentiert) |
| 4 | **d4pm** | 512/7 | **hoch (≥3)** | Diffusion; im letzten Run flaky |
| 5 | dpae | 512/7 | niedrig | symmetrische Fusion |
| 6 | ic_unet | 512/7 | mittel | reines Sensor-Level-U-Net (ICA raus) |
| 7 | **st_gnn** | 512/7 | mittel | **torch-geometric auf CUDA prüfen** |
| 8 | vit_spectrogram | 512/7 | mittel | **Loss-Geometrie an Modell koppeln** |
| 9 | dhct_gan | 512/7 | mittel | Multi-Disc nur teilweise realisiert |
| 10 | dhct_gan_v2 | 64 (prüfen) | mittel | LSGAN; Multi-Disc teilweise |
| 11 | denoise_mamba | 512/7 | niedrig | nur `jit.trace`-exportierbar (ok) |
| 12 | **nested_gan** | 512/7 | **hoch (≥1)** | Primärpaper fehlt → Restormer-treu |

---

## 5. Modell-spezifische Risiken & Sonderfälle

Aus der unabhängigen Verifikation der Editionen (Verify-Stage):

- **dhct_gan / dhct_gan_v2 — Multi-Discriminator/Gating nur teilweise:** Unter
  dem Standard-`facet-train`-Kontrakt ruft der Wrapper
  `loss_fn(model(x), target)`, wobei `model(x)` nur den Artefakt-Head liefert.
  Folge: Clean-/Gate-Heads bekommen **keinen Gradienten**, und zwei der drei
  Diskriminatoren sehen identische Paare (redundant). Für den Fleet-Run mit
  Standard-Wrapper ist das **erwartet und dokumentiert**. Die volle GAN-Rezeptur
  (separate Generator-Betas 0.5/0.9, echte 3-Disc-Aufteilung) bräuchte eine
  **eigene Trainings-Schleife** außerhalb des Wrappers → **bewusst out of scope**
  für diesen Run (als Folge-Run vormerken).
- **denoise_mamba — nicht `jit.script`-bar:** Export läuft über `jit.trace`
  (`cli.py` nutzt trace), verifiziert round-trip-fähig. Keine Aktion, nur im
  Log vermerken (gilt auch fürs Original).
- **vit_spectrogram — Geometrie-Kopplung:** siehe §3.1-Warnung.
- **st_gnn — PyG/CUDA:** torch-geometric muss zur CUDA-Torch-Version des
  Pod-Images passen; sonst Import-/Kernel-Fehler erst zur Trainingszeit.
- **d4pm / nested_gan — flaky:** großzügiges Smoke-Retry-Budget; Smokes nicht
  abbrechen.
- **nested_gan — Primärpaper paywalled & fehlt:** Edition ist dem
  **Restormer-Backbone** treu (MDTA/GDFN); GAN-Nesting nicht gegen Quelle
  verifizierbar. Ergebnisse als „Restormer-backbone faithful" labeln.

---

## 6. Runbook (exakte Befehle)

```bash
# 0) Worker-Config aktualisieren (lokal, nicht committen)
$EDITOR tools/gpu_fleet/workers.local.yaml

# 1) Bootstrap je Pod
tools/gpu_fleet/bootstrap_runpod.sh root@<host1> https://github.com/H0mire/FACETpy.git /workspace/facetpy <port1>
# (optional gpu2)

# 2) Smoke (CUDA) submitten — Beispiel conv_tasnet
python tools/gpu_fleet/fleet.py submit \
  --name conv_tasnet_pa_niazy_smoke \
  --worktree . \
  --worker gpu1 \
  --training-config src/facet/models/conv_tasnet_paper_accurate_edition/training_niazy_proof_fit_cuda_smoke.yaml \
  --prepare-command "uv run python examples/dataset_building/build_niazy_proof_fit_context_dataset.py --artifact-bundle output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz --target-epoch-samples 512 --context-epochs 7 --output-dir output/niazy_proof_fit_context_512"

# 3) Full submitten (nach grünem Smoke)
python tools/gpu_fleet/fleet.py submit \
  --name conv_tasnet_pa_niazy_full \
  --worktree . \
  --training-config src/facet/models/conv_tasnet_paper_accurate_edition/training_niazy_proof_fit.yaml

#    → für alle 12 Editionen wiederholen (Smoke + Full).

# 4) Dispatcher starten (verteilt pending Jobs auf freie Worker)
python tools/gpu_fleet/fleet.py dispatch --loop --interval 60

# 5) Status / Live-Logs
python tools/gpu_fleet/fleet.py status
ssh -p <port> root@<host> 'tmux attach -t <session>'

# 6) Ergebnisse holen
python tools/gpu_fleet/fleet.py fetch            # alle Worker
tools/gpu_fleet/fetch_runpod_results.sh root@<host> /workspace/facetpy . <port>
```

Pro Trainingslauf entstehen (Konvention): `training.jsonl`, `loss.png`,
`summary.json`, Checkpoints, exportiertes TorchScript-Modell.

---

## 7. Evaluation & A/B-Vergleich

1. Pro Edition Eval-Script laufen lassen (vorhandene Originale haben
   `evaluate.py` / `examples/model_evaluation/evaluate_<id>.py` als Vorlage);
   Output über `facet.evaluation.ModelEvaluationWriter` nach
   `output/model_evaluations/<id>_paper_accurate_edition/<run_id>/`
   (`evaluation_manifest.json`, `metrics.json`, `evaluation_summary.md`, Plots).
2. **A/B gegen die Originale** (Run vom 10.05.): pro Modell
   `flat_metrics` aus `metrics.json` der Edition vs. Original
   gegenüberstellen. Vergleichsregel aus `evaluation_standard.md`: nicht an
   *einer* Metrik festmachen — supervised-synthetisch **und** real-proxy
   (trigger-locked RMS) **und** Spike-Preservation **und** Laufzeit/Memory.
3. Leitfrage: *Bringt die Paper-Treue messbar etwas* gegenüber der
   ursprünglichen Edition — pro Modell ein klares besser/gleich/schlechter.

---

## 8. Deliverables

- [ ] 12 × `training_niazy_proof_fit.yaml` (CUDA, paper-skaliert) + CUDA-Smoke.
- [ ] `workers.local.yaml` mit lebenden Hosts (lokal).
- [ ] 12 × Full-Training `finished` + Export + `summary.json`.
- [ ] 12 × `output/model_evaluations/<edn>/<run_id>/` Standard-Run-Dateien.
- [ ] A/B-Vergleichstabelle Edition vs. Original (eine Markdown-Tabelle).
- [ ] Aktualisierter `.facet_gpu_fleet/queue.json`-Stand (bleibt gitignored).
- [ ] Kurzes Run-Log (Commit-Hash/Snapshot, Pods, Fehler/Retries).

---

## 9. Offene Entscheidungen vor Kickoff

1. **1 oder 2 GPUs?** Letzter Run lief faktisch fast komplett auf `gpu2`;
   24 Jobs (12×Smoke+Full) sind mit 2 Pods in ~einem Abend machbar.
2. **Editionen vorher committen?** (empfohlen, §3.3) — braucht ausdrückliche
   Freigabe wegen Commit-Policy.
3. **`target_type` je Edition** (`artifact` vs. `clean`) verifizieren, bevor das
   Full-Config eingefroren wird.
4. **dhct_gan/v2 voller GAN-Loop** — in diesem Run *nicht* (Standard-Wrapper),
   als Folge-Run vormerken? (§5)
5. **Nachzügler aus Run-3-Welt:** die zwei `pending` cascaded-`full`-Jobs aus
   dem letzten Run sind paperlos — bewusst weglassen.
