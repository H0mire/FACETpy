# Run 3 — Beating AAS for Spike Preservation

Companion zu [`run_2_plan.md`](run_2_plan.md), [`thesis_results_report.md`](thesis_results_report.md)
und [`architecture_catalog.md`](architecture_catalog.md).

**Status der Vorgänger-Phasen** (für Klarheit):
- **Run 1** (ausgeführt): die ursprünglichen 12 parallelen Modell-Trainings + Evaluationen
- **Unified-Holdout-Re-Evaluation** (ausgeführt): die bestehenden Run-1-Modelle wurden auf einem einheitlichen 166-Window-Holdout re-evaluiert; **kein Re-Training**
- **Run 2** (geplant, noch nicht ausgeführt): Bugfixes für die 4 defekten Modelle aus Run 1 (vit_spectrogram, dpae, dhct_gan v1/v2)
- **Run 3** (dieser Plan, noch nicht ausgeführt): siehe unten

**Frage**: Kann ein Deep-Learning-Modell **AAS schlagen**, gemessen an der
*klinisch relevanten Metrik*: Erhaltung epileptischer Spikes, die sich unter
fMRT-Gradient-Artefakten verstecken?

Branch-Vorschlag: `feature/spike_preservation_run3`
Erwartete Dauer: 3-4 Wochen fokussierter Arbeit.

---

## 1. Warum dieser Run?

Aus Run 1 (Architektur-Exploration) und der nachfolgenden
Unified-Holdout-Re-Evaluation wissen wir:
- **Demucs** ist die mit Abstand stärkste Architektur (+31.30 dB auf
  Unified-Holdout, gemessen)
- Das **AAS-Fidelity-Ceiling** begrenzt die Bedeutung dieser Zahl —
  Demucs lernt aktuell, AAS zu approximieren, nicht AAS zu schlagen
- Die ursprüngliche **klinische Motivation** (Spike-Detektion in simultanem
  EEG-fMRT) wird mit der bisherigen Methodik nicht direkt adressiert

Run 3 schließt diese Lücke:

| Run 1 + Unified-Holdout (ausgeführt) | Run 3 (geplant) |
|---|---|
| Architektur-Exploration über 12 Modelle | Vertiefung der Top-Architektur (Demucs) |
| Bulk-SNR auf AAS-Target | Spike-Detection-Metriken auf Ground-Truth-Spikes |
| Single-Channel pro Modell | Multichannel-Kontext |
| Generic L1-Loss | Spike-aware Loss |
| Comparison gegen andere DL-Modelle | Comparison gegen AAS direkt |

---

## 2. Hypothesen zum Testen

**Wichtige Vorbemerkung zum AAS-Vergleich**: AAS subtrahiert strukturell nur
die **periodische Komponente** des Signals. Epileptische Spikes sind
non-periodisch (treten nicht phasensynchron zum MR-Trigger auf) und können
daher **nicht ins Template gelangen** — sie werden durch AAS-Subtraktion
zu **0 %** beschädigt. Spikes bleiben *vollständig* erhalten. Die klinische
Schwäche von AAS bei Spike-Detektion liegt einzig in den **morphologischen
Eigenschaften der Restartefakte**: residuelle Artefakt-Komponenten (durch
Bewegung, Drift, Trigger-Jitter, non-stationäre Gradient-Variationen) haben
**transiente, breitbandige, sharp-rise-Formen** die epileptischen Spikes
strukturell ähnlich sind. Das verursacht **False Positives in der
Spike-Detektion**. Die Run-3-Hypothesen sind entsprechend formuliert.

**H1 · Target-Ceiling lösbar.** Mit synthetischem Datensatz (echtes
Ground-Truth-EEG + injizierte Spikes + simuliertes Artefakt) entfällt das
AAS-Ceiling. Demucs sollte auf solchem Target höhere Korrektur-Qualität
erreichen als auf dem Niazy-AAS-Target — speziell bei der Reduktion
residueller Artefakt-Komponenten, die im AAS-Output verbleiben.

**H2 · Multichannel hilft spezifisch für Restartefakt-Reduktion und
Spike-Differenzierung.** Räumliche Topographie ist der Schlüssel-Diskriminator
zwischen fokal-dipolarem Spike (echtes EEG-Signal) und gradient-artigem
Restartefakt (artefakt-Quelle). Multichannel-Demucs sollte Single-Channel
auf der Restartefakt-vs-Spike-Verwechselbarkeit übertreffen, weil es die
unterschiedlichen räumlichen Footprints ausnutzen kann. Spike-Amplitude und
-Erhaltung sind in beiden Methoden gleich.

**H3 · Spike-aware Loss verhindert DL-spezifisches Spike-Mit-Korrigieren.**
Anders als AAS, das Spikes strukturell *nicht* erfassen kann, hat DL ein
spezifisches Risiko: wenn das Trainings-Target Spikes als "zu korrigierende
Abweichung" behandelt, lernt das Modell sie zu entfernen. Eine spike-aware
Loss-Komponente verhindert das. Dies ist ein DL-Bug-Schutz, kein DL-Vorteil
über AAS.

**H4 · DL schlägt AAS bei der Restartefakt-Spike-Diskriminierbarkeit.**
Die korrekte Vergleichsmetrik ist nicht "wieviele Spikes erkenne ich nach
Korrektur" (Spike-Erhaltung ist bei AAS strukturell 100% und bei DL via
spike-aware Loss absichert), sondern "wie häufig produziert ein
Spike-Detektor False Positives auf dem korrigierten Signal". Auf dieser
Metrik kann DL strukturell besser sein, weil es non-stationäre
Artefakt-Variationen modellieren kann statt sie als residuum übrig zu
lassen.

---

## 3. Phasen-Plan

### Phase A · Dataset-Erweiterung (Woche 1)

**Ziel**: einen neuen Datensatz `niazy_spike_preservation_v1.npz`, der
clean EEG + bekannte Spikes + simuliertes fMRT-Artefakt enthält.

**Datenquellen:**
- **Spike-Library**: TUH EEG Spike Corpus oder PhysioNet-CHB-MIT, kuratiert
  auf isolierte interictale Spikes (~500-1000 Templates)
- **Clean EEG**: aus dem Niazy-AAS-Output entnommen — auch wenn
  AAS-bounded, dient hier nur als Träger für injizierte Spikes
- **Artefakt-Templates**: aus dem Niazy-Artefakt-Library
  (`output/artifact_libraries/niazy_aas_2x_direct/`) bereits vorhanden

**Build-Skript**: `examples/build_spike_preservation_dataset.py`

**Pipeline**:
```
1. Lade clean EEG (N Beispiele × 30 Kanäle × T_native)
2. Sample für jedes Beispiel: 0-3 zufällige Spike-Events
   - Spike-Template aus Library
   - Random-Onset im Sample (~70% innerhalb der Center-Epoche)
   - Random-Amplitude im physiologischen Bereich (30-150 μV)
   - Topografie: dipolar projiziert auf 30-Kanal-Setup über
     forward-model (z.B. mne.simulate_evoked + dipole position)
3. Addiere Spike(s) zum clean EEG → clean_with_spikes
4. Sample MRT-Artefakt-Template aus Niazy-Library
5. Addiere Artefakt → noisy_with_spikes
6. Erzeuge spike_labels: (N, n_ch, T) binary mask
   - 1 in Spike-Region (±50 ms um Spike-Peak)
   - 0 sonst
7. Speichere alle Arrays + Sfreq + Trigger als .npz
```

**Output-Schema** (`output/spike_preservation_v1/spike_preservation_v1.npz`):
```python
noisy_with_spikes_context:    (N, 7, 30, 512)  float32  # Input
clean_with_spikes_context:    (N, 7, 30, 512)  float32  # Target
clean_with_spikes_center:     (N, 30, 512)     float32  # Eval-Target
noisy_with_spikes_center:     (N, 30, 512)     float32
artifact_only_context:        (N, 7, 30, 512)  float32  # Pures Artefakt
spike_labels_context:         (N, 7, 30, 512)  bool     # Spike-Mask
spike_labels_center:          (N, 30, 512)     bool
spike_metadata:               (N,) object              # n_spikes, onsets, channels, amplitudes
sfreq:                        (1,) float32
```

**Validierungs-Check**:
- 30 % der Beispiele haben mindestens 1 Spike
- Spike-Amplitude im Center-Epoch median ~80 μV
- Topographien plausibel (dipolar, Max an passender Elektrode)
- Artefakt-Statistik matched Niazy-Recording

**Aufwand**: 5 Tage. Risiko: Spike-Library-Qualität (kuratierte annotierte
Spikes können knapp sein → Fallback: parametrische Spike-Generierung
via Gauss-Modulierter Half-Cycle nach Lourenço et al. 2015).

---

### Phase B · Multichannel-Demucs (Woche 2)

**Ziel**: `src/facet/models/demucs_mc/` mit Cross-Channel-Bridge.

**Architektur (Variante B aus Diskussion)**:

```python
class MultichannelDemucs(nn.Module):
    def __init__(self, n_channels=30, ...):
        # Per-Channel-Encoder (shared weights)
        self.encoder = nn.ModuleList([...])  # gleiche wie Single-Channel
        # NEU: Cross-Channel Self-Attention nach jedem Encoder-Level
        self.channel_attn = nn.ModuleList([
            CrossChannelAttention(C_i, n_heads=4)
            for C_i in encoder_channels
        ])
        # BiLSTM unverändert
        self.bilstm = nn.LSTM(C_bottleneck, C_bottleneck, num_layers=2,
                              bidirectional=True)
        # Per-Channel-Decoder (shared weights)
        self.decoder = nn.ModuleList([...])

    def forward(self, x):
        # x: (B, 30, T_input)
        B, C, T = x.shape

        # Encoder: process each channel separately, share weights
        # Then cross-channel attention at each level
        skips = []
        h = x.unsqueeze(2)  # (B, 30, 1, T)
        for i, (enc, attn) in enumerate(zip(self.encoder, self.channel_attn)):
            # Reshape to (B*30, 1, T_i) for shared-weight conv
            h_flat = h.view(B * C, *h.shape[2:])
            h_enc = enc(h_flat)
            # Reshape back to (B, 30, C_i, T_i)
            h = h_enc.view(B, C, *h_enc.shape[1:])
            # Cross-channel attention: per timestep, attend over channels
            h = attn(h)
            skips.append(h)

        # BiLSTM at bottleneck: per-channel
        h_lstm = []
        for c in range(C):
            seq = h[:, c].transpose(1, 2)  # (B, T_bottleneck, C_features)
            h_lstm.append(self.bilstm(seq)[0].transpose(1, 2))
        h = torch.stack(h_lstm, dim=1)

        # Decoder mirror with skip-adds
        for i in reversed(range(len(self.decoder))):
            h = h + skips[i]  # skip
            h_flat = h.view(B * C, *h.shape[2:])
            h_dec = self.decoder[i](h_flat)
            h = h_dec.view(B, C, *h_dec.shape[1:])

        return h.squeeze(2)  # (B, 30, T)
```

**Cross-Channel-Attention**: Standard-MultiHeadAttention über die
Channel-Dimension. Treats the 30 EEG channels als "Sequenz" für Attention,
ähnlich wie eine Vision-Transformer-Patch-Attention.

**Parameter-Budget-Vorabschätzung**:
- Single-Channel Demucs (gemessen in Run 1): 16.6 M Parameter
- Multichannel-Demucs: Encoder/Decoder bleiben unverändert wegen Shared
  Weights; die Self-Attention-Layer über die Channel-Dimension ergänzen
  zusätzliche Parameter. Die genaue Größenordnung hängt von der gewählten
  Head-Anzahl, Layer-Anzahl und Positions-Encoding-Strategie ab und sollte
  vor Implementation per `torchinfo` o.ä. konkret berechnet werden.
  Die Erwartung ist: **Multichannel-Demucs bleibt in der gleichen
  Größenordnung wie Single-Channel-Demucs**, also vermutlich unter 25 M,
  damit das Modell weiterhin auf einem einzelnen GPU mit 24 GB VRAM
  trainierbar ist.

**Tests**:
- `tests/models/demucs_mc/test_forward.py`: shape check, gradient flow
- `tests/models/demucs_mc/test_channel_invariance.py`: gleicher Output bei
  Channel-Permutation (sollte equivariant sein, nicht channel-spezifisch
  gebrochen)

**Aufwand**: 5 Tage.

---

### Phase C · Spike-Aware Loss (parallel zu Phase B, 2-3 Tage)

**Ziel**: Loss-Funktion in `src/facet/models/demucs_mc/training.py`, die
Spike-Regionen höher gewichtet.

**Formel**:
```python
def spike_aware_loss(pred, target, spike_mask, lambda_spike=10.0):
    """
    pred:       (B, 30, T) — Model output
    target:     (B, 30, T) — Clean with spikes preserved
    spike_mask: (B, 30, T) bool — 1 where spike is present

    Loss = L1_recon + lambda_spike * L1_spike_region
    """
    l1_full = F.l1_loss(pred, target, reduction='mean')

    # Per-element L1, then mask + weight
    per_elem = F.l1_loss(pred, target, reduction='none')
    mask_float = spike_mask.float()
    spike_loss = (per_elem * mask_float).sum() / (mask_float.sum() + 1e-8)

    return l1_full + lambda_spike * spike_loss
```

**Auch erwägen**:
- **Focal-Loss-Variante**: höhere Strafe für *vergessene* Spikes (hohe
  prediction error in Spike-Region) als für falsche Spikes (irrtümliches
  Hinzufügen)
- **Topografische Konsistenz-Loss**: Spike-Field muss dipolar sein
  (Korrelations-Penalty zwischen Nachbarkanälen)

**Hyperparameter-Sweep**: `lambda_spike ∈ {1, 5, 10, 25, 50, 100}` mit
einem kleineren Schnellrunden-Setup (max_examples=2048, 10 Epochen).

**Aufwand**: 3 Tage inkl. Sweep.

---

### Phase D · Evaluation-Framework (Woche 3)

**Ziel**: neue Metriken in `tools/eval_unified_holdout.py` und
`tools/aggregate_unified_holdout.py`, plus eine direkte AAS-Vergleichs-Run.

**Neue Metriken** (in `compute_metrics()`):

```python
def compute_spike_metrics(pred_clean, target_clean, spike_labels):
    """
    pred_clean, target_clean: (N, 30, T)
    spike_labels: (N, 30, T) bool — Ground-Truth-Spike-Region
    """
    # Spike-Region und Umfeld
    in_spike = spike_labels
    in_non_spike = ~spike_labels
    in_neighborhood = expand_mask(spike_labels, margin_samples=200)  # ±50ms @ 4kHz
    in_neighborhood_only = in_neighborhood & ~in_spike

    # 1. Spike-Amplitude-Preservation (Ziel: 1.0)
    # Per Spike: max(|pred|) vs max(|target|). AAS schützt rare events
    # strukturell durch die Mittelung — Spikes sind non-periodisch und
    # können nicht ins Template gelangen. Erwartung daher: AAS liefert
    # Werte sehr nahe an 1.0. DL muss das via Loss-Design absichern,
    # sonst besteht das Risiko, dass Spikes als "zu korrigierende
    # Variation" mit-gelernt werden.
    spike_amp_ratio = compute_per_spike_amplitude_ratio(
        pred_clean, target_clean, spike_labels)

    # 2. Spike-Hintergrund-SNR im Umfeld (HAUPTMETRIK für DL vs AAS)
    # Wie sauber ist das EEG im ±50ms-Umfeld des Spikes? AAS lässt hier
    # residuelle Artefakte zurück, DL kann sie potenziell besser
    # entfernen. Genau das ist der wahre DL-Vorteil.
    spike_neighborhood_snr = snr_db(
        target_clean[in_neighborhood_only],
        pred_clean[in_neighborhood_only] - target_clean[in_neighborhood_only])

    # 3. Spike-zu-Background-Kontrast (cSNR) — das eigentliche
    # klinische Erkennbarkeitsmaß. Spike-Peak vs RMS des Restartefakts
    # im Spike-Umfeld.
    spike_contrast_db = compute_spike_contrast_db(
        pred_clean, target_clean, spike_labels, in_neighborhood_only)

    # 4. Spike-Detection-Sensitivität bei festem False-Positive-Rate
    # Standard-Spike-Detector (Amplitude+Shape) auf korrigiertem Signal,
    # bei festem FPR-Niveau (z.B. 5%). Erwartung: bei rein stationären
    # Aufnahmen liefern AAS und DL ähnlich hohe Recall-Werte. Der
    # Unterschied dürfte erst bei non-stationären Komponenten sichtbar
    # werden, wenn AAS-Restartefakte die Detektor-Schwelle stören.
    spike_recall_at_5pct_fpr = compute_spike_recall(
        pred_clean, target_clean, spike_labels, fpr=0.05)

    # 5. Spike-Latency-Drift (Ziel: 0)
    latency_drift_ms = compute_peak_latency_drift(
        pred_clean, target_clean, spike_labels)

    # 6. Spike-Morphologie-Treue: Korrelation pred vs target im Spike-Fenster
    spike_morphology_corr = compute_spike_morphology_correlation(
        pred_clean, target_clean, spike_labels)

    # 7. Non-Spike-Region SNR — generelle Bulk-Korrektur außerhalb Spikes
    non_spike_snr = snr_db(target_clean[in_non_spike],
                           pred_clean[in_non_spike] - target_clean[in_non_spike])

    return {
        "spike_amplitude_ratio": spike_amp_ratio,             # Ziel: 1.0
        "spike_neighborhood_snr_db": spike_neighborhood_snr,  # HAUPTMETRIK
        "spike_contrast_db": spike_contrast_db,               # HAUPTMETRIK
        "spike_recall_at_5pct_fpr": spike_recall_at_5pct_fpr,
        "spike_peak_latency_drift_ms": latency_drift_ms,      # Ziel: 0
        "spike_morphology_corr": spike_morphology_corr,
        "non_spike_snr_db": non_spike_snr,
    }
```

**Update am Aggregator**: neue Spalten in `UNIFIED_HOLDOUT.md` und
`INDEX.md` für die Spike-Metriken.

**Welche Metrik ist die AAS-vs-DL-Headline?**

Die zentrale klinische Frage ist: **wie oft produziert ein
Spike-Detektor False Positives auf dem korrigierten Signal?** Daher die
wichtigste Metrik:

```python
def compute_false_positive_rate(pred_clean, spike_labels, detector):
    """
    Apply a standard amplitude+shape-based spike detector to the
    corrected signal. Count False Positives (detected "spikes" outside
    ground-truth spike_labels regions) per minute.
    """
    detected = detector(pred_clean)
    # False Positive: detector flagged something where no spike was injected
    fp_mask = detected & ~expand_mask(spike_labels, margin_samples=50)
    fp_per_minute = fp_mask.sum() * (sfreq * 60) / pred_clean.size
    return fp_per_minute
```

Diese Metrik kombiniert Architektur-Performance und klinische Relevanz —
sie ist genau das Maß, das in der Klinik zählt. AAS hat hier inhärent ein
Problem, weil seine Restartefakte spike-ähnliche Morphologie haben. DL
hat hier theoretisch einen Vorteil — *falls* die Restartefakte nach DL
nicht nur kleiner sind, sondern auch **strukturell anders** aussehen als
Spikes.

Sekundär relevante Metriken:
- `spike_amplitude_ratio` (Sanity-Check für Spike-Erhaltung; beide
  Methoden sollten ~1.0 erreichen)
- `non_spike_snr_db` (Bulk-Korrektur-Qualität außerhalb Spikes)
- `spike_morphology_corr` (Spike-Form-Treue im Spike-Fenster)

**Aufwand**: 4 Tage inkl. Implementation und Validierung der
Spike-Detector-Schwelle.

---

### Phase E · AAS Apples-to-Apples Comparison (Woche 3, parallel)

**Ziel**: AAS auf dem identischen Spike-Preservation-Datensatz laufen lassen,
gleiche Metriken berechnen, in die Tabelle einreihen.

**Skript**: `tools/eval_aas_on_spike_preservation.py`

```python
# Pseudocode
for window in spike_preservation_dataset:
    raw = window_to_mne_raw(window.noisy_with_spikes)
    pipeline = Pipeline([
        TriggerDetector(...),
        AASCorrection(window_size=30),  # Standard FACETpy AAS
    ])
    corrected = pipeline.run(raw)
    metrics = compute_spike_metrics(
        corrected, window.clean_with_spikes,
        window.spike_labels)
```

Outputs unter `output/model_evaluations/aas_baseline/spike_preservation_v1/`.

**Erwartung**: AAS wird in *Bulk*-SNR vermutlich konkurrenzfähig zur
Demucs-Multichannel-Variante sein (oder etwas besser für nicht-Spike-Regionen),
aber **deutlich schwächer in Spike-Recall** — das ist die These.

**Aufwand**: 2 Tage.

---

### Phase F · Comparison-Run + Reporting (Woche 4)

**Ziel**: vollständige Vergleichstabelle in
`docs/research/spike_preservation_results.md`.

**Vergleichs-Setup**:

| Modell | Training-Daten | Test-Daten | Metriken |
|---|---|---|---|
| AAS | (kein Training) | `spike_preservation_v1` Holdout | alle Unified-Holdout-Metriken aus Run 1 + die neuen Spike-Metriken |
| Single-Channel Demucs (Run-1-Baseline, im Unified-Holdout re-evaluiert) | Niazy-Proof-Fit | `spike_preservation_v1` Holdout | dito |
| **Multichannel Demucs + Standard-L1** | `spike_preservation_v1` Train | `spike_preservation_v1` Holdout | dito |
| **Multichannel Demucs + Spike-Aware Loss** | `spike_preservation_v1` Train | `spike_preservation_v1` Holdout | dito |

**Generierte Outputs**:
- `output/model_evaluations/demucs_mc/spike_preservation_v1_l1/`
- `output/model_evaluations/demucs_mc/spike_preservation_v1_spikeaware/`
- `output/model_evaluations/aas_baseline/spike_preservation_v1/`
- `docs/reports/2026-06-XX_spike_preservation_report/` (analog zu
  Status-Update mit Figuren + HTML/PDF)

**Aufwand**: 5 Tage inkl. Report-Writing.

---

## 4. Deliverables (am Ende von Run 3)

| Artefakt | Pfad |
|---|---|
| Spike-Preservation-Datensatz | `output/spike_preservation_v1/` |
| Multichannel-Demucs-Modell | `src/facet/models/demucs_mc/` |
| Trained Checkpoints (3 Varianten) | `training_output/demucs_mc_*/` |
| Spike-Aware-Loss-Implementation | `src/facet/models/demucs_mc/training.py` |
| Neue Spike-Metriken | `tools/eval_unified_holdout.py` (erweitert) |
| AAS-Baseline-Eval | `output/model_evaluations/aas_baseline/spike_preservation_v1/` |
| Per-Modell-Holdout-Evals | `output/model_evaluations/demucs_mc/*` |
| Vergleichsreport | `docs/research/spike_preservation_results.md` |
| Status-Update für Betreuer | `docs/reports/2026-06-XX_spike_preservation/` |

---

## 5. Risikoanalyse

| Risiko | Wahrscheinlichkeit | Auswirkung | Mitigation |
|---|---|---|---|
| Spike-Library zu klein / nicht annotiert | mittel | hoch | Fallback auf parametrisch generierte Spikes (Gauss-modulierter Half-Cycle) |
| Synthetisches Artefakt unrealistisch | mittel | mittel | Validiere Spektrum vs echte Niazy-Aufnahmen + diskutiere im Caveat |
| Cross-Channel-Attention bringt nichts | gering | mittel | Ablation-Study: mit/ohne Bridge → Differenz quantifizieren |
| AAS schlägt DL auch bei Spikes | gering | hoch | Wenn ja → echtes wissenschaftliches Resultat, Thesis erzählt warum |
| TorchScript-Device-Baking wieder ein Problem | mittel | gering | run_2_plan §3.5 — fix bei Export |
| Training auf Spike-Aware-Loss instabil | mittel | mittel | Lambda-Sweep + Mixed-Loss (L1 + λ·spike), starte mit kleinem λ |

---

## 6. Was *nicht* in Run 3 gehört

Diese Themen sind eindeutig **out-of-scope** für Run 3, damit das Projekt
einen klaren Endpunkt hat:

- ❌ Echte klinische Validierung an Patientendaten (braucht Ethik-Antrag,
  separate PhD-Studie)
- ❌ Real-time-Inferenz-Optimierung (TensorRT etc.)
- ❌ BCG-Artefakt-Korrektur (separate Architektur-Frage)
- ❌ Cross-Subject- und Cross-Scanner-Generalisierung (braucht Multi-Site-Daten)
- ❌ Hybrid- oder Transformer-Demucs (HT-Demucs v4) — nicht Thesis-relevant
- ❌ Weitere Architekturen aus dem Catalog (D4PM, ST-GNN etc. retrainieren)

---

## 7. Decision Points vor Kickoff

Diese Fragen sollten **vor Phase A** mit dem Betreuer geklärt werden:

1. **Welche Spike-Library?** TUH EEG Spike Corpus (frei, gut annotiert) vs
   eigene klinische Daten (besser kuratiert, aber Ethik). Mein Vorschlag:
   TUH starten, falls verfügbar.

2. **Welcher Spike-Typ?** Nur interictale "klassische" Spikes (Dauer 20-70 ms)
   oder auch sharp waves (70-200 ms)? Mein Vorschlag: erstmal klassische Spikes,
   sharp waves als Stretch-Goal.

3. **Wie viele Spikes pro Window?** 0-3 ist meine Default-Annahme. Höhere
   Dichte simuliert ictale Phasen, niedrigere ist realistischer für
   Interictal-Recording. → Mit Betreuer abstimmen.

4. **Synthetic Spikes oder Real Spikes als Library?** Real-extracted ist
   realistischer, parametric ist kontrollierter (Amplitude/Dauer als
   variable Hyperparameter). Beide haben Vor- und Nachteile. → Beide
   parallel evaluieren?

5. **Soll die Thesis Single- oder Multichannel-Demucs als primären
   Beitrag positionieren?** Single-Channel ist bereits Top-1 in Run 1
   und im Unified-Holdout — ein klares, gemessenes Resultat. Multichannel
   wäre der nächste Schritt zur "echten" Antwort auf die klinische
   Frage, aber noch ungemessen. → Strategie-Entscheidung.

6. **Wie streng muss die AAS-Vergleichbarkeit sein?** Soll man auch
   AAS-Varianten (Iterative AAS, OBS) als Baselines reinnehmen, oder reicht
   Vanilla-AAS?

---

## 8. Erwartete Aussage am Ende

**Best case** (alle Hypothesen bestätigt):
> *"Multichannel Demucs mit Spike-aware Loss reduziert auf einem
> synthetischen Datensatz mit injizierten Spikes die Anzahl der False
> Positives in der automatisierten Spike-Detektion um X % gegenüber AAS
> — bei gleichbleibender Spike-Amplituden-Erhaltung (~1.0 in beiden
> Methoden) und gleichbleibendem True-Positive-Rate. Die
> Restartefakt-RMS-Amplitude reduziert sich um Y dB. Damit ist die
> Architektur klinisch relevant überlegen, weil sie die
> Verwechslungs-Wahrscheinlichkeit zwischen Restartefakt und Spike
> systematisch reduziert."*

**Realistic case** (H1+H2 ja, H3 als Schutz wirksam):
> *"Auf der Bulk-SNR (außerhalb Spikes) ist Demucs vergleichbar mit oder
> besser als AAS. Auf der False-Positive-Rate bei Spike-Detektion
> schlägt Multichannel-Demucs AAS um Y %. Die Spike-Amplituden-Erhaltung
> ist in beiden Methoden ~1.0; die spike-aware Loss-Komponente diente
> vor allem dem Schutz vor versehentlicher Spike-Mit-Korrektur durch das
> DL-Modell, nicht als Vorteil über AAS."*

**Worst case** (H4 falsch — DL erreicht nur AAS-Niveau):
> *"Auf der False-Positive-Rate ist Multichannel-Demucs vergleichbar mit
> AAS, ohne signifikante Verbesserung. DL bietet weiterhin
> Inferenz-Geschwindigkeit und Trigger-Robustheit. Detaillierte Analyse
> zeigt: die residuellen Artefakte nach AAS haben in diesem (stationären)
> Datensatz keine ausreichende Spike-Ähnlichkeit, um den DL-Vorteil
> sichtbar zu machen. Possible Implication: AAS ist für rein-stationäre
> Recordings nahe-optimal; der wahre DL-Hebel liegt bei nicht-stationären
> Aufnahmen (Bewegung, Drift), die in der zukünftigen Forschung
> untersucht werden sollten."*

Alle drei Fälle sind **publikationswürdig**, weil sie eine wissenschaftliche
Frage präzise beantworten — der Worst-Case wäre tatsächlich der
interessanteste, weil er die Limitierungen aktueller DL-Korrektur
quantitativ aufzeigen würde.

---

## 9. Kickoff-Checkliste

Bevor Phase A startet:

- [ ] Decision Points §7 mit Betreuer geklärt
- [ ] Branch `feature/spike_preservation_run3` aus
  `feature/proof_fit_consolidated` erstellt
- [ ] TUH EEG Spike Corpus heruntergeladen (oder Alternative)
- [ ] Niazy-Artefakt-Library noch verfügbar
  (`output/artifact_libraries/niazy_aas_2x_direct/`)
- [ ] GPU-Fleet aktiv und für Multi-Day-Training verfügbar
- [ ] `docs/research/run_3_plan.md` (dieses Dokument) mit Betreuer
  abgestimmt

---

## Anhang A · Architektur-Diagramm Multichannel-Demucs

```
                  Input: (B, 30, 3584)
                          │
            ┌─────────────┴─────────────┐
            │   Per-Channel-Encoder     │  ← shared weights
            │   (Conv1d × 4 Stages)     │
            └─────────────┬─────────────┘
                          │ (B, 30, C_i, T_i)
            ┌─────────────┴─────────────┐
            │  Cross-Channel-Attention  │  ← NEU
            │  (MHA über 30er-Dim)      │
            └─────────────┬─────────────┘
                          │ (B, 30, C_i, T_i)
              [skip-out to decoder]
                          │
                  Bottleneck Stage
                          │
            ┌─────────────┴─────────────┐
            │      BiLSTM × 2 Layer     │  ← per-channel, shared
            └─────────────┬─────────────┘
                          │
            ┌─────────────┴─────────────┐
            │   Per-Channel-Decoder     │  ← shared weights
            │   (ConvT1d × 4 Stages)    │
            │   + Skip-Adds             │
            └─────────────┬─────────────┘
                          │
                  Output: (B, 30, 3584)
```

## Anhang B · Spike-Aware-Loss-Schema

```
Standard L1 (full):           L1_full = mean(|pred − target|)        [low weight]
Spike-region L1:              L1_spike = mean(|pred − target|)       [high weight λ]
                                          where spike_mask == True

Topo-Consistency (optional):  L_topo = corr_penalty(neighbor channels in spike regions)

Total:                        L = L1_full + λ·L1_spike + μ·L_topo
                              (typisch: λ ∈ [10, 50], μ ∈ [0, 1])
```

---

*Erstellt: 2026-05-12 · Run 3 Owner: Janik M. Müller · Status: Plan,
noch nicht im Lauf · Reviewers: Thesis-Betreuung*
