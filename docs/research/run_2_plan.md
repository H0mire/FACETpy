# Run 2 — Plan & Erkenntnisse aus Run 1

Companion zu [`thesis_results_report.md`](thesis_results_report.md) (was passiert ist)
und [`architecture_catalog.md`](architecture_catalog.md) (welche Modellfamilien).
Dieses Dokument: **was im zweiten Lauf konkret anders gemacht werden muss**.

Branch of record: `feature/proof_fit_consolidated`
Datensatz: `output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz`
(833 Beispiele × 7 Kontext-Epochen × 30 Kanäle × 512 Samples, sfreq 4096 Hz)

---

## 1. Stand nach Run 1

12 Deep-Learning-Modelle wurden parallel auf zwei RunPod-GPUs trainiert und gegen
den Niazy Proof-Fit-Datensatz evaluiert. Daraus:

| Bucket | Modelle | Status |
|---|---|---|
| Top-3 (Audio-Familie) | demucs, conv_tasnet, sepformer | Stabil, +19–31 dB SNR. **Nicht erneut trainieren**, nur ggf. Holdout-Re-Eval. |
| Plateau-Tier | nested_gan, denoise_mamba, ic_unet, st_gnn | Konvergiert auf ~+11–13 dB. Architektur funktioniert; ggf. Tuning. |
| **Defekte** | vit_spectrogram, dpae, dhct_gan, dhct_gan_v2 | Diagnostizierte Bugs, siehe §2. |
| Confounded | d4pm | Nur 32 Beispiele evaluiert (Diffusions-Inferenzkosten). Holdout-Re-Eval nötig. |

Im Folgenden bedeutet **Run 2** den Re-Training-Pass für die defekten Modelle plus
einen gemeinsamen Holdout-Re-Eval-Pass für alle 12.

---

## 2. Modell-spezifische Bugs (Run 1) und Fixes für Run 2

### 2.1 `vit_spectrogram` — Dead-Clamp Output-Collapse

**Symptom.** `val_loss` ist über alle 13 Epochen bit-identisch
`2.3923365652483917e-07`. `train_loss` fällt nach Epoche 1 auf denselben Wert.
Best epoch = 1. Effektives SNR +11.60 dB nur, weil bei der Eval Fallback-Logik
greift; das Modell selbst lernt nichts.

**Wurzelursache.** [`src/facet/models/vit_spectrogram/training.py:358-360`](../../src/facet/models/vit_spectrogram/training.py:358):

```python
predicted_patches = self.decoder_head(tokens)
pred_log_mag = self._unpatchify(predicted_patches)
pred_magnitude = torch.expm1(pred_log_mag).clamp(min=0.0)
```

`expm1(x)` ist für `x < 0` negativ; `clamp(min=0.0)` saturiert dann auf 0 und
nullt den Gradient (`∂clamp/∂x = 0` im gesättigten Bereich). Sobald in Epoche 1
ein paar Decoder-Outputs unter 0 wandern → gradient-tot → Bias-Drift zieht den
Rest hinterher → Modell konvergiert auf "Magnitude = 0 überall" → `iSTFT(0)=0` →
`Loss = mean(clean²) = 2.39e-7` ist ein Konstanter.

**Fix (Run 2).** Ersetze in [training.py:360](../../src/facet/models/vit_spectrogram/training.py:360):

```python
# alt: pred_magnitude = torch.expm1(pred_log_mag).clamp(min=0.0)
pred_magnitude = torch.nn.functional.softplus(pred_log_mag)
```

`softplus` ist überall differenzierbar und positiv, ohne Sättigungs-Plateau.
Alternativ (falls man Magnitude > 0 strikt erzwingen will): Residual zur
Input-Magnitude lernen:

```python
input_magnitude = magnitude  # vor dem Patchifizieren bereits berechnet
pred_magnitude = input_magnitude * torch.exp(pred_log_mag)
```

**Sanity-Check nach Fix.** Nach 1 Epoche darf `val_loss` *nicht* mehr
bit-identisch zu späteren Epochen sein. Bei korrektem Training: `val_loss`
ändert sich um mind. 6 signifikante Stellen pro Epoche.

---

### 2.2 `dpae` — BatchNorm-Running-Stats Divergenz

**Symptom.** `train_loss` glatt (5e-7 bis 5e-6). `val_loss` springt
zwischen `4.3e-7` und `6.0e-4` — vier Größenordnungen in benachbarten Epochen.
SNR +7.48 dB, der niedrigste aller diskriminativen Modelle.

**Wurzelursache.** [`src/facet/models/dpae/training.py:122-127`](../../src/facet/models/dpae/training.py:122):

```python
self.fusion = nn.Sequential(
    nn.BatchNorm1d(fused),  # ← Problem
    nn.Conv1d(fused, fused, kernel_size=1),
    nn.SELU(inplace=True),
)
```

`BatchNorm1d` benutzt im `eval()`-Modus die getrackten Running-Stats. Bei
per-Kanal Daten mit batch_size=128 sind die Batch-Statistiken von Batch zu
Batch hoch volatil (einzelne demeaned EEG-Kanäle), und die Running-Stats
(momentum=0.1) tracken diese Schwankungen statt zu konvergieren. Im Eval-Modus
liegen die Running-Stats dann nicht zur aktuellen Val-Batch-Verteilung → BN
verzerrt die Feature-Distribution → Modell-Output weicht stark vom
Trainings-Regime ab → Loss schießt um Größenordnungen hoch (oder fällt zufällig
auf "Best", was zu der trügerischen `best_epoch=16`-Markierung führt).

**Fix (Run 2).** Eine der Optionen, in Reihenfolge bevorzugt:

```python
# Option A (empfohlen): GroupNorm — kein train/eval-Unterschied
self.fusion = nn.Sequential(
    nn.GroupNorm(num_groups=8, num_channels=fused),
    nn.Conv1d(fused, fused, kernel_size=1),
    nn.SELU(inplace=True),
)

# Option B: LayerNorm über Kanaldimension
# (transpose nötig wegen Conv-Layout)

# Option C (Minimal-Eingriff): BN-Momentum absenken
nn.BatchNorm1d(fused, momentum=0.01)
```

**Sanity-Check.** Nach Fix darf `val_loss` sich pro Epoche nicht mehr um
mehr als eine Größenordnung bewegen, sobald `train_loss` stabil im
sub-`1e-5`-Bereich ist.

---

### 2.3 `dhct_gan` (v1) — Input-Contract + GAN-Discriminator-Dominanz

**Symptom.** SNR **−7.13 dB** (Verschlechterung), `val_loss` monoton 0.20 → 0.67.
Best epoch = 1.

**Wurzelursache (2 unabhängige Bugs).**

1. **Input-Contract**: v1 nutzt nur `noisy_center` (eine Epoche), nicht den
   7-Epochen-Kontext. v2 hat diesen Teil gefixt → +1.69 dB (immer noch
   schlecht, siehe §2.4).

2. **GAN-Dynamik** in [`src/facet/models/dhct_gan/training.py:348-453`](../../src/facet/models/dhct_gan/training.py:348):
   ```python
   generator_loss = recon_artifact + α * recon_consistency
   if beta_adv > 0.0:
       adv_loss = BCE_with_logits(D(pred), 1)
       generator_loss = generator_loss + beta_adv * adv_loss
   ```
   - Diskriminator hat eigenen Adam-Optimizer im selben Schritt (1:1 G/D-Ratio)
   - BatchNorm im Diskriminator destabilisiert, wenn G die Output-Verteilung
     verschiebt
   - `monitor: loss` (=generator total) ist *kein* verlässliches Stopping-Signal:
     wenn D besser wird, wächst `adv_loss` mechanisch, selbst wenn die L1-Fidelity
     gleich bleibt

**Fix (Run 2).** v1 wird in Run 2 **nicht** wiederbelebt — v2 (mit korrigiertem
Input-Contract) ist die Referenzimplementierung. Stattdessen v2 mit folgenden
Änderungen erneut trainieren (siehe §2.4).

---

### 2.4 `dhct_gan_v2` — GAN-Discriminator-Dominanz (Input-Contract ist ok)

**Symptom.** SNR +1.69 dB. `val_loss` oszilliert 0.13–0.77, mit Trend nach oben
über 34 Epochen.

**Wurzelursache.** Identische GAN-Dynamik wie v1. Der Input-Contract-Fix hat
den −7.13 → +1.69 dB Sprung ermöglicht, aber die Diskriminator-Dominanz und
das Loss-Monitor-Problem sind unverändert.

**Fix (Run 2).** Mehrere Optionen, gestaffelt nach Aufwand:

**Option A — Adversarial ganz raus** (empfohlen, ein einzeiliger Config-Change):

In [`src/facet/models/dhct_gan_v2/training_niazy_proof_fit.yaml`](../../src/facet/models/dhct_gan_v2/training_niazy_proof_fit.yaml):
```yaml
loss_kwargs:
  alpha_consistency: 0.5
  beta_adv: 0.0       # alt: 0.1
```
Damit ist es ein reiner CNN+Transformer-Regressor mit L1-Recon + Consistency.
Erwartung: konkurrenzfähig zu IC-U-Net / Mamba (~+11–13 dB).

**Option B — Stabilisiertes GAN** (mehr Aufwand, bessere Inputs für Thesis):
- BCE → Hinge-Loss oder LS-GAN (`L_D = E[(D(real)-1)²] + E[D(fake)²]`)
- Spectral Normalization am Diskriminator
- 5:1 G:D-Update-Ratio (D-Update nur jeden 5. Schritt)
- D's `BatchNorm1d` → `InstanceNorm1d` oder `GroupNorm`
- Early-Stopping auf separate Validation-SNR-Metrik, nicht auf `generator_loss`

**Empfehlung für Run 2.** Option A. Falls Zeit bleibt: zusätzlich Option B als
separates Modell `dhct_gan_v3` (eigener Order, eigene Run-ID), damit die
Adversarial-Wirkung sauber isoliert vergleichbar wird.

---

### 2.5 `d4pm` — Sample-Size Confound (kein Bug, aber blockierend für Vergleich)

**Symptom.** SNR +3.21 dB, aber nur 32 Beispiele × 4 Kanäle evaluiert. Andere
Modelle haben 833 oder 4998 Beispiele.

**Wurzelursache.** Diffusions-Inferenz ist zu teuer für den vollen Validation-Split
unter dem Run-1-Zeitbudget.

**Fix (Run 2).** Im einheitlichen Holdout-Re-Eval (siehe §5.1) auch d4pm auf
demselben Test-Split evaluieren. Dafür Inference-Budget ~1–2 h GPU einplanen.
Code-Änderung nicht nötig, nur Eval-Run.

---

## 3. Cross-cutting Lessons — auf alle Run-2-Implementierungen anwenden

### 3.1 Input-Contract: `noisy_context` vs `noisy_center`

Der Niazy-Datensatz liefert `noisy_context` der Form `(833, 7, 30, 512)` und
`noisy_center` der Form `(833, 30, 512)`. Modelle, die nur den zentralen
Epoch sehen (`noisy_center` oder der mittlere Kontext-Index), sind durch
**isolierten Single-Epoch-Input** stark benachteiligt — DHCT-GAN v1 erreichte
damit −7.13 dB statt +1.69 dB (9 dB Swing).

**Regel für Run 2.** Jeder neue oder reaktivierte Agent **muss explizit
deklarieren**, ob das Modell Single-Epoch- oder Multi-Epoch-Input nutzt, und
warum. Single-Epoch nur dann, wenn die Architektur prinzipiell keinen Kontext
nutzen kann (begründen!).

**Plateau-Tier-Audit (Run 2 Task).** denoise_mamba, ic_unet, st_gnn, dpae
liegen alle bei ~+7–12 dB. Hypothese: einige davon nutzen ebenfalls nur den
Center-Epoch. Vor weiterem Tuning prüfen, ob ein 7-Epoch-Input verfügbar wäre.

### 3.2 Aktivierungs-Sättigung (Dead-Gradient Traps)

`clamp`, `relu`, `expm1+clamp`, `softmax` mit großem Temperatur-Gefälle —
alles, was im Forward-Pfad einen flachen Bereich erzeugt, kann den Gradient
auf große Teile des Output-Raums vollständig nullen. Klassisches Symptom:
**bit-identische Loss-Werte über Epochen hinweg**.

**Regel für Run 2.** Wenn Magnitudes oder strikt positive Werte gebraucht
werden: `softplus` (überall differenzierbar, asymptotisch ≈ identity), nicht
`relu`/`clamp`. Wenn ein Wertebereich erzwungen werden muss: `sigmoid` oder
`tanh` mit linearer Skalierung, nicht `clamp`.

**Sanity-Check.** Nach 2 Epochen Training: `set(val_loss_values)` muss
mindestens 2 verschiedene Werte enthalten. Falls nicht, sofort Forward-Pfad
debuggen — das Modell ist dead.

### 3.3 BatchNorm bei per-Kanal Daten

Mit `batch_size=128` einzelnen demeaned EEG-Kanälen sind BatchNorm-Statistiken
inhärent volatil. Result: train/eval-Mismatch → unbrauchbare Val-Kurve.

**Regel für Run 2.** Default für neue Modelle ist **GroupNorm** (oder
LayerNorm). BatchNorm nur dort, wo Batches groß *und* statistisch homogen
sind (z.B. Diskriminator auf vollen Multi-Channel-Spektrogrammen). Wenn BN
nötig ist: `momentum=0.01` statt Default `0.1`.

### 3.4 GAN-Dynamik

GAN-Training ist im EEG-Denoising-Setting hochfragil:
- Ein einziger Diskriminator-Update-Schritt pro Generator-Schritt führt zur
  D-Dominanz
- BCE-Verlust am Diskriminator hat unbeschränkte Gradienten
- `generator_total_loss` ist kein verlässliches Konvergenz-Signal

**Regel für Run 2.**
- Adversarial-Komponente ist **opt-in**, nicht default. Starte ohne, addiere
  nur wenn ein reiner Regressor nachweislich an einer Ceiling klebt
- Wenn adversarial: Hinge oder LS-GAN, Spectral-Norm, ≥3:1 G/D-Ratio
- Early-Stopping immer auf eine separate Validation-Metrik (SNR oder
  rein-rekonstruktiver Loss), nicht auf das Total-Loss-Aggregat

### 3.5 TorchScript Device-Baking

Beim Tracen via `torch.jit.trace` kann ein `tensor.to(device)`-Aufruf im
Forward-Pfad die Device-Konstante in den Graph baken — der exportierte
Adapter ist dann CUDA-locked. Siehe Fix-Pattern in Commit `4184443` im
`worktrees/model-vit_spectrogram`.

**Regel für Run 2.** In `forward()` keine `.to(...)`-Aufrufe. Geräte-Transfers
erfolgen am Adapter-Layer (`DeepLearningModelAdapter`), nicht in der nn.Module.

### 3.6 Early-Stopping-Signal

Im Run 1 hat `monitor: loss` für GANs zu 16/34 Epochen monoton steigender
Loss geführt, bevor Patience triggerte. Für andere Modelle war das ok.

**Regel für Run 2.** Für neue/refaktorierte Modelle:
- Validation-Verlust **immer** loggen (`val_every_n_epochs: 1`)
- Early-Stopping monitor: `val_loss` (nicht `loss`)
- Bei GANs zusätzlich: eigene Validation-SNR-Metrik berechnen und auf die
  early-stoppen, nicht auf den Trainings-Loss

---

## 4. Infrastruktur-Status

### 4.1 In Run 1 stabilisiert (Run 2 kann darauf bauen)

- **Zentrale Queue über alle Worktrees**: `tools/gpu_fleet/fleet.py` resolviert
  `REPO_ROOT` via `git rev-parse --git-common-dir`. Agenten in
  `worktrees/model-<id>/` schreiben in dieselbe Queue.
- **Per-Job `uv sync` + Torch-Check**: `tools/gpu_fleet/run_remote_training.sh`
  synct vor dem Training und ruft `tools/gpu_fleet/check_torch.py` auf
  (Exit-Code-Verifikation, dass torch.cuda verfügbar).
- **Dispatcher-Console-Logging**: zeitstempelte Status-Übergänge, Heartbeats,
  `listening`-Indikator.
- **Robuste tmux-Session-Detection**: `tmux ls -F #{session_name}` als
  single-quoted String (nicht als separate argv).
- **`preferred_worker=any/none/""` normalisiert**: Submit-Time-Validation,
  Dispatcher überspringt unbekannte Worker mit Warning statt SystemExit.
- **`output/model_evaluations/` whitelisted** in `.gitignore`
  (`output/*` + `!output/model_evaluations`).
- **HANDOFF.md pro Modell-Ordner** (`src/facet/models/<id>/HANDOFF.md`), nicht
  Repo-Root.

### 4.2 Noch offen / empfohlen für Run 2

- **Symlink-Fix für `training_output`** (in Run 1 deferred): Worktrees haben
  eigenes `training_output/`, daher manueller Copy-Step im
  ModelEvaluationWriter nötig. Symlink `worktrees/<id>/training_output → ../../training_output`
  würde das automatisieren.
- **Cross-Model Holdout-Split** (siehe §5.1).
- **Architektur-Catalog-Update**: nach Run 2 die neuen v3/fixed-Varianten
  ergänzen.

---

## 5. Evaluation-Methodik für Run 2

### 5.1 Einheitlicher Holdout-Split

**Problem (aus thesis_results_report §5).** In Run 1 wurden die Modelle auf
unterschiedlich großen Test-Splits evaluiert (833, 4998, 32 Beispiele). Die
absoluten SNR-Zahlen sind nicht direkt vergleichbar.

**Lösung für Run 2.** Vor jedem Re-Eval:

1. Einen **gemeinsamen Holdout-Split** definieren (z.B. `seed=42`,
   `val_ratio=0.2`, dieselbe Indexliste serialisiert nach
   `output/niazy_proof_fit_context_512/holdout_indices.json`).
2. Jedes Modell auf genau diesem Split evaluieren — auch Modelle, die nicht
   re-trainiert werden müssen (`demucs`, `conv_tasnet`, `sepformer`,
   `nested_gan`, `ic_unet`, `denoise_mamba`, `st_gnn`).
3. `evaluation_manifest.json` enthält ab Run 2 das Feld `holdout_split_hash`,
   um Vergleichbarkeit zu attestieren.

### 5.2 AAS-Fidelity-Ceiling

**Problem (aus thesis_results_report §5).** Der "clean" Target ist
AAS-korrigiert, nicht echter Ground Truth. Die Metriken messen Fidelity zur
AAS-Schätzung, nicht physikalisches Denoising darüber hinaus.

**Lösung für Run 2 (Reporting).** Im Run-2-Report:
- Klarstellen, dass `+30 dB SNR` "vs AAS-Target" bedeutet, nicht "vs ideale
  saubere EEG"
- Wenn möglich: einen Sub-Test auf nicht-AAS-korrigierten Synthetic-Daten
  (cascaded_dae-Baseline-Set) ergänzen, um die Ceiling zu illustrieren

### 5.3 Loss-Plot-QA als Pflicht-Schritt

Run 1 hätte die ViT- und DPAE-Bugs sofort gezeigt, wenn die Loss-Plots vor
Akzeptanz inspiziert worden wären. Daher:

**Regel für Run 2.** Bevor ein Modell als "done" markiert wird:
1. `plots/training_loss.png` öffnen
2. Prüfen: Sinkt `val_loss` monoton? Wenn nein → Debug.
3. Prüfen: Variiert `val_loss` zwischen Epochen? Wenn nein → Dead-Activation-Check.
4. Prüfen: Differenz `train_loss − val_loss` plausibel? Wenn `val_loss < train_loss`
   um Größenordnungen → BN-Issue prüfen.

Diese Checks gehören in die agent-prompt Phase 9 (Hand-Off).

---

## 6. Konkreter Run-2-Plan

### 6.1 Reihenfolge

1. **Holdout-Split fixieren** und serialisieren (1 Skript, ~30 min).
2. **Code-Fixes für die 4 defekten Modelle** (siehe §2.1–§2.4):
   - vit_spectrogram: `softplus` statt `expm1+clamp`
   - dpae: BN → GroupNorm
   - dhct_gan_v2: `beta_adv: 0.0` in YAML
   - dhct_gan v1: nicht re-trainieren, deprecated markieren
3. **Re-Training** dieser 3 Modelle (vit, dpae, dhct_gan_v2) parallel auf
   gpu1/gpu2 via Fleet-Submit. Erwartete Dauer: <2 h pro Modell.
4. **Holdout-Re-Eval** aller 12 Modelle (inkl. der 3 frisch trainierten) auf
   dem gemeinsamen Split. d4pm braucht ~1–2 h, Rest jeweils ~10 min.
5. **Updated INDEX.md** und thesis_results_report mit Run-2-Zahlen.

### 6.2 Welche Agenten spawnen

Nur **3 Re-Training-Agenten**:

- `model-vit_spectrogram-v2` — Fix §2.1 anwenden, neu trainieren, neu evaluieren
- `model-dpae-v2` — Fix §2.2 anwenden, neu trainieren, neu evaluieren
- `model-dhct_gan_v3` — Fix §2.4 Option A (beta_adv=0) als eigenes Modell

Plus **1 Eval-Agent**:

- `eval-holdout-rerun` — gemeinsamer Holdout-Split, alle 12 Modelle auf
  diesem Split re-evaluieren, INDEX.md updaten.

Agent-Prompts folgen dem Template in [`docs/model_agent_prompts.md`](../model_agent_prompts.md).

**Pflicht-Reading vor Code-Änderungen** (in dieser Reihenfolge):

1. Dieses Dokument (§2 für den spezifischen Fix, §3 für Cross-cutting-Regeln)
2. [`src/facet/models/README.md`](../../src/facet/models/README.md) — Integration-Pattern
3. **`src/facet/models/<id>/HANDOFF.md`** — Hand-off-Notiz des Vor-Agenten. Enthält die ursprüngliche Hypothese, was im Run 1 versucht wurde, welche Metriken erreicht wurden und (für die defekten Modelle) erste Fail-Mode-Beobachtungen. **Pflicht zu lesen vor jedem Code-Change** — der vorherige Agent hat dort oft schon Konfig-Entscheidungen begründet, die der neue Agent nicht blind überschreiben soll.
4. **`src/facet/models/<id>/README.md`** und **`src/facet/models/<id>/documentation/`** (falls vorhanden) — Architekturbeschreibung und Design-Rationale. Wer den Fix anwendet, muss verstanden haben, *warum* der Vor-Agent die jetzige Struktur gewählt hat, damit er nicht aus Versehen den intendierten Mechanismus zerstört.
5. [`thesis_results_report.md`](thesis_results_report.md) §4 (Fail-Mode-Diskussion über alle Modelle hinweg)

**Pflicht-Update nach Code-Änderungen.** Der Run-2-Agent aktualisiert (nicht nur ergänzt) die Dokumentation des Modells:

- `src/facet/models/<id>/HANDOFF.md`: neue Sektion `## Run 2 Update` mit identifiziertem Bug, angewandtem Fix, neuen Metriken und einem expliziten Vergleich Run-1 vs Run-2
- `src/facet/models/<id>/README.md`: Architektur-Bullet aktualisieren, falls der Fix die Forward-Pfad-Semantik verändert (z.B. `softplus` statt `clamp`, `GroupNorm` statt `BatchNorm`)
- `src/facet/models/<id>/documentation/` (falls vorhanden): wenn der Bug einer Design-Annahme widersprach, diese Annahme korrigieren

Ziel: nach Run 2 enthält die Modell-Dokumentation eine vollständige Historie der Fehler-Diagnose. Ein hypothetischer Run-3-Agent soll die Fehlentscheidungen aus Run 1 nicht wiederholen können.

### 6.3 Nicht erneut trainieren

`demucs`, `conv_tasnet`, `sepformer`, `nested_gan`, `denoise_mamba`,
`ic_unet`, `st_gnn` — diese 7 Modelle haben in Run 1 ohne diagnostizierten
Bug konvergiert. Sie werden in Run 2 nur **re-evaluiert** auf dem
gemeinsamen Holdout-Split, nicht re-trainiert.

`dhct_gan` (v1) — als deprecated markieren, kein Re-Training. Das v3 ersetzt
die Familie.

### 6.4 Optionale Folge-Schritte (Run 3 oder Stretch-Goals)

Nur wenn nach Run 2 noch Budget ist:

- **Plateau-Tier Input-Audit** (§3.1): denoise_mamba und ic_unet auf
  konsistenten Multi-Epoch-Input umstellen, falls sie aktuell nur Center
  sehen. Erwarteter Upside: +2–5 dB.
- **Stabilisiertes GAN** (Option B aus §2.4): dhct_gan_v3 als
  Hinge-Loss + Spectral-Norm + 5:1-Ratio Variante, separat zur reinen
  Regressor-Version trainieren.
- **Ensemble**: gewichtetes Mittel der Top-3 (demucs + conv_tasnet +
  sepformer). Erwartet konsistenter als das Maximum-Modell allein.

---

## 7. Sanity-Checklist (vor jedem "Modell fertig"-Hand-Off in Run 2)

Pflicht-Punkte, abzuhaken in HANDOFF.md:

**Training-Qualität:**
- [ ] `val_loss` variiert über Epochen hinweg (nicht bit-identisch)
- [ ] `val_loss` ist im selben Größenordnung-Bereich wie `train_loss`
  (kein 10×-Spike)
- [ ] `val_loss` sinkt über die ersten 5 Epochen
- [ ] `best_epoch` ist nicht Epoche 1 (außer bei trivialen Tasks)
- [ ] Loss-Plot visuell inspiziert (`plots/training_loss.png`)

**Evaluation:**
- [ ] Modell auf Holdout evaluiert (nicht auf Random-Split)
- [ ] `evaluation_manifest.json` enthält `holdout_split_hash`
- [ ] TorchScript-Export auf CPU UND CUDA getestet (Device-Baking-Check)
- [ ] Input-Contract dokumentiert: Single-Epoch oder Multi-Epoch (Begründung)

**Dokumentation (siehe §6.2 Pflicht-Update):**
- [ ] HANDOFF.md des Vor-Agenten gelesen, *vor* Code-Changes
- [ ] README.md und documentation/ des Modells gelesen, Design-Rationale verstanden
- [ ] HANDOFF.md mit Sektion `## Run 2 Update` ergänzt: Bug, Fix, Vorher/Nachher-Metriken
- [ ] README.md aktualisiert, falls Forward-Pfad-Semantik verändert wurde
- [ ] documentation/ aktualisiert, falls eine Design-Annahme widerlegt wurde
