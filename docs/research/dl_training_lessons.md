# Deep-Learning Training — stehende Lehren & Checklisten

Zeitlose Engineering-Regeln aus Run 1 (ursprünglich `run_2_plan.md` §3/§5/§7).
Gelten für **jedes** FACETpy-DL-Modell — Originale wie `_paper_accurate_edition`.
Aktueller Plan-Überblick: [`README.md`](README.md).

---

## 1. Cross-cutting Lessons

### 1.1 Input-Contract: `noisy_context` vs `noisy_center`
Der Niazy-Datensatz liefert `noisy_context` `(N, 7, 30, 512)` und `noisy_center`
`(N, 30, 512)`. Modelle, die nur den zentralen Epoch sehen, sind stark
benachteiligt — DHCT-GAN v1 erreichte mit Single-Epoch −7.13 dB statt +1.69 dB
(9 dB Swing). **Regel:** jedes Modell muss explizit deklarieren, ob es Single-
oder Multi-Epoch-Input nutzt, und warum. Single-Epoch nur, wenn die Architektur
prinzipiell keinen Kontext nutzen kann.

### 1.2 Aktivierungs-Sättigung (Dead-Gradient-Traps)
`clamp`, `relu`, `expm1+clamp`, `softmax` mit großem Gefälle — alles, was im
Forward-Pfad einen flachen Bereich erzeugt, kann den Gradient auf großen Teilen
des Output-Raums nullen. Klassisches Symptom: **bit-identische Loss-Werte über
Epochen** (genau der vit_spectrogram-Bug: `expm1(x).clamp(min=0)` → gradient-tot
→ Magnitude-Kollaps auf 0). **Regel:** strikt positive Werte via `softplus`
(überall differenzierbar), nicht `relu`/`clamp`; erzwungener Wertebereich via
`sigmoid`/`tanh` + lineare Skalierung, nicht `clamp`.

### 1.3 BatchNorm bei per-Kanal-Daten
Mit `batch_size=128` einzelnen demeaned EEG-Kanälen sind BN-Statistiken volatil →
train/eval-Mismatch → unbrauchbare Val-Kurve (der dpae-Bug: Val-Loss sprang über
4 Größenordnungen). **Regel:** Default **GroupNorm** (oder LayerNorm). BN nur bei
großen, statistisch homogenen Batches; wenn nötig `momentum=0.01` statt `0.1`.

### 1.4 GAN-Dynamik
Im EEG-Denoising hochfragil: 1:1-G/D-Ratio → D-Dominanz; BCE am Diskriminator
hat unbeschränkte Gradienten; `generator_total_loss` ist kein verlässliches
Konvergenz-Signal. **Regel:** Adversarial-Komponente ist **opt-in**, nicht
Default. Wenn adversarial: Hinge- oder LS-GAN, Spectral-Norm, ≥3:1 G/D-Ratio,
Early-Stopping auf eine separate Validierungs-Metrik (SNR oder rein-rekonstruktiv).

### 1.5 TorchScript Device-Baking
`tensor.to(device)` im `forward()` kann beim `torch.jit.trace` die Device-Konstante
in den Graph backen → CUDA-locked Export. **Regel:** keine `.to(...)`-Aufrufe in
`forward()`; Geräte-Transfers am Adapter-Layer (`DeepLearningModelAdapter`).

### 1.6 Early-Stopping-Signal
`monitor: loss` hat für GANs zu 16/34 Epochen steigendem Loss geführt, bevor
Patience triggerte. **Regel:** Val-Loss immer loggen (`val_every_n_epochs: 1`),
Early-Stopping auf `val_loss` (nicht `loss`); bei GANs auf eine eigene
Validierungs-SNR-Metrik.

---

## 2. Evaluations-Methodik

- **Einheitlicher Holdout-Split.** Gemeinsamer, serialisierter Split (`seed=42`,
  `val_ratio=0.2`, Indexliste nach `holdout_indices.json`); jedes Modell auf
  *genau* diesem Split evaluieren; `evaluation_manifest.json` trägt
  `holdout_split_hash`. (In Run 1 wurden Modelle auf 833/4998/32 Beispielen
  evaluiert → nicht vergleichbar.)
- **AAS-Fidelity-Ceiling.** „+30 dB SNR" heißt „vs AAS-Target", nicht „vs ideale
  saubere EEG". Im Report klarstellen; wo möglich einen Sub-Test auf
  nicht-AAS-Daten ergänzen. (Die Entkopplung davon ist jetzt eigener Arc:
  [`run_3_decoupled_dataset_weg_a.md`](run_3_decoupled_dataset_weg_a.md) /
  [`run_5_self_supervised_weg_b.md`](run_5_self_supervised_weg_b.md).)
- **Loss-Plot-QA als Pflicht.** Vor „done": Sinkt `val_loss` monoton? Variiert er
  über Epochen (sonst Dead-Activation)? Ist `train−val` plausibel (sonst BN-Issue)?

---

## 3. Sanity-Checkliste (vor jedem „Modell fertig")

**Training-Qualität**
- [ ] `val_loss` variiert über Epochen (nicht bit-identisch)
- [ ] `val_loss` in derselben Größenordnung wie `train_loss` (kein 10×-Spike)
- [ ] `val_loss` sinkt über die ersten 5 Epochen
- [ ] `best_epoch` ist nicht Epoche 1 (außer bei trivialen Tasks)
- [ ] Loss-Plot visuell inspiziert

**Evaluation**
- [ ] auf gemeinsamem Holdout evaluiert (nicht Random-Split)
- [ ] `evaluation_manifest.json` enthält `holdout_split_hash`
- [ ] TorchScript-Export auf CPU **und** CUDA getestet (Device-Baking)
- [ ] Input-Contract dokumentiert (Single- vs Multi-Epoch, begründet)
