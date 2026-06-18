# Run 3 — Von AAS entkoppelter Trainingsdatensatz (Weg A)

> **Kontext:** Ergebnis der Datensatz-Diskussion (2026-06-11). Dies ist **Weg A**
> der AAS-Entkopplung: ein semi-synthetischer, räumlich-zeitlicher
> Referenz-Datensatz mit *unabhängiger* clean-Quelle. **Weg B**
> (self-supervised, ganz ohne Target) ist als nächster Run in
> [`run_5_self_supervised_weg_b.md`](run_5_self_supervised_weg_b.md)
> dokumentiert.
>
> Verwandt: [`run_6_beat_aas_spike_preservation.md`](run_6_beat_aas_spike_preservation.md) (Spike-Preservation gegen AAS),
> `src/facet/models/evaluation_standard.md`.

---

## 0. TL;DR

1. **Problem:** der proof-fit-Datensatz ist per Konstruktion an AAS gekoppelt
   (`noisy = corrected(AAS) + artifact(AAS)`) → die Decke des Modells *ist* AAS.
2. **Entkopplung (zwei Hebel, beide in diesem Run):**
   - **Artefakt vollständiger machen:** Artefakt = **AAS + subtrahierte
     PCA/OBS-Komponenten (`n_components=4`)**, nicht AAS allein.
   - **clean unabhängig machen:** das Artefakt-Template auf eine *unabhängige*
     clean-Quelle legen (synthetisch zuerst), Target = das *wahre* clean.
3. **Struktur:** ein Beispiel pro (Zielkanal `c`, Epoche `e`); Input =
   reale Referenz **(7 Epochen × 3 Kanäle × S)**, Target = Artefakt von `(c,e)`.
4. **Augment:** Kern-Shift (continuous-window) + Hintergrund-Mix (clean-swap) +
   Amplituden-Jitter + TR-/Längen-Jitter & Rauschen (Auswahl 1/2/4).
5. **Scope:** Builder + Dataset-Klasse + Shift-Transform + billiger CPU-Test;
   single-recording proof-fit zuerst.

---

## 1. Warum wir aktuell an AAS hängen

`build_niazy_proof_fit_context_dataset.py` baut alles aus dem AAS-Bundle:
`noisy = corrected(AAS) + artifact(AAS)`. Damit ist die Zerlegung exakt erfüllt,
aber **zirkulär**: das beste lernbare Ziel ist „gib aus, was AAS ausgegeben
hätte". AAS' Fehler — Residual (Epoch-zu-Epoch-Variabilität, Bewegung,
He-Pumpe, Sub-sample-Rest) und Over-subtraction (geglättete Spikes/Alpha) —
tauchen im Target nie als Fehler auf. → Modell imitiert AAS, kann es nicht
schlagen. (So auch im docstring des Builders vermerkt.)

## 2. Hebel 1 — Artefakt = AAS + PCA/OBS (`n_components=4`)

**Begründung.** AAS subtrahiert ein gemitteltes Template und lässt damit die
**slice-zu-slice-Residualstruktur** stehen. Der OBS-/PCA-Schritt
(`PCACorrection`, MATLAB `DoPCA`/`FitOBS`-treu) modelliert genau diese
Residual-Varianz über die ersten Hauptkomponenten. **AAS + PCA** ist daher eine
*vollständigere* Schätzung des echten Gradientenartefakts als AAS allein — und
genau die Residualstruktur, an der AAS scheitert, kommt so ins Artefakt-Template
(und damit ins Training).

**Wie (mechanisch).** Die Pipeline akkumuliert Artefaktschätzungen über
`context.accumulate_noise(...)`: AAS addiert seine Schätzung, `PCACorrection`
addiert ihre OBS-Rekonstruktion ([pca.py:171](../../src/facet/correction/pca.py)).
`context.get_estimated_noise()` liefert daher **AAS + PCA summiert**. Konkret:

```python
# Erweiterung von examples/dataset_building/extract_niazy_artifact_signal.py:
# nach AAS einen OBS-Schritt einziehen, dann das kombinierte estimated_noise exportieren.
steps = [
    ...,
    AASCorrection(...),                       # füllt estimated_noise mit AAS
    PCACorrection(n_components=4, hp_freq=300.0),  # OBS auf >300 Hz: nur HF-Residual, kein Hirn
    ...,
]
artifact_combined = context.get_estimated_noise()   # = AAS + PCA(4)
corrected         = original - artifact_combined
# als neues Bundle z.B. output/artifact_libraries/niazy_aas_pca4_direct/...npz
```

**Config (festgelegt): `PCACorrection(n_components=4, hp_freq=300.0)`.**
`n_components=4` ist die klassische Niazy-OBS-Zahl. **`hp_freq=300`** ist der
FACET-Beispiel-Cutoff: der hohe Cutoff hält das EEG-Band aus dem OBS-Basisraum
heraus, sodass der OBS **nur das HF-Residual** des Artefakts modelliert und
garantiert **kein Hirnsignal** entfernt (sicherste Variante laut
`PCACorrection`-docstring). Das ins Artefakt-Template aufgenommene OBS-Residual
ist also bewusst der hochfrequente Anteil, den AAS liegen lässt — das im
Bundle-Metadaten festhalten.

> **Volle Bandbreite — kein 70-Hz-Low-pass.** Das Modell soll das Artefakt *in
> seiner Gesamtheit* lernen. Wir setzen **keinen** nachgelagerten 70-Hz-Low-pass
> voraus — nicht jeder filtert so, und unsere DL-Modelle sollen gerade *höhere*
> EEG-Frequenzen erhalten als der klassische FACET-Cutoff. Also: Datensatz **und**
> Eval auf voller Nyquist-Bandbreite bauen, **kein** `LowPassFilter` im
> Datenaufbau-Pfad; der >300-Hz-OBS-Anteil wird **behalten und gelernt**, nicht
> verworfen.
>
> Technisch (verifiziert in `pca.py`): bei `hp_freq=300` wird das gesamte
> Akquisitionssignal zuerst mit 300 Hz hochpassgefiltert, und `fitted_artifact`
> wird aus diesen gefilterten Epochen rekonstruiert → der OBS-Beitrag ist genau das
> **>300-Hz-Residual**. Die breitbandige Artefakt-„Body" liefert weiterhin **AAS**.
> Also: `Template = AAS (breitbandig) + OBS (>300-Hz-Residual)`. Das **in-band**-Residual
> (<300 Hz) lassen wir bewusst stehen, weil dort das Hirn sitzt — ein tieferer
> OBS-Cutoff brächte Brain-Removal-Risiko. „Gesamtheit" heißt hier: voller
> AAS-Body über das ganze Band **plus** das HF-Residual, das ein 70-Hz-Pfad
> wegwerfen würde.

> Hinweis zur Restkopplung: das Template stammt weiterhin aus der Pipeline
> (AAS+OBS). Die Zirkularität bricht trotzdem, weil dieses Template (Hebel 2) auf
> **unabhängiges** clean gelegt wird — die AAS-Fehler entstehen beim *Subtrahieren
> vom selben Signal*, nicht im Template selbst. Für volle Entkopplung später:
> GA simulieren/messen (Weg C, nicht dieser Run).

## 3. Hebel 2 — unabhängige clean-Quelle

`noisy = clean_true + artifact_template`, **Target = `clean_true`** (bzw. das
Template). `clean_true` ist *nicht* AAS-korrigiert, sondern umschaltbar:

- `clean_source = synthetic` — synthetisches EEG mit realistischem Spektrum +
  Spikes; Generatoren existieren (`generate_synthetic_fmri_artifact_source.py`,
  `build_synthetic_spike_artifact_*`, `generate_synthetic_spike_source_dataset.py`).
  **Erste Variante.**
- `clean_source = external` — outside-scanner-EEG derselben Probanden/Montage
  oder sauberer Korpus auf die Montage gemappt (stärkste Variante, falls
  Daten vorhanden).
- `clean_source = aas_corrected` — alter, gekoppelter Modus; nur als Baseline.

> **Bandbreite — Artefakt ≠ clean.** Die „kein 70-Hz-Low-pass"-Regel (§2) gilt
> für die **Artefakt-Berechnung** und das finale `noisy`, **nicht** für die
> clean-Herstellung. Die clean behält die Bandbreite ihrer Quelle (echtes/synth.
> EEG hat von Natur aus kaum >70-Hz-Leistung) — wir lowpassen sie nicht künstlich,
> spritzen aber auch kein künstliches HF rein; oberhalb des EEG-Bandes ist `noisy`
> dann ≈ reines Artefakt. **Konsistenz:** `clean_true` und `artifact_template`
> werden addiert → gleiches `fs`/Zeitgrid/Epochenlänge; die clean **nicht** heimlich
> downsamplen (impliziter Low-pass).

Optional die Fehlermodi explizit dazumischen, an denen AAS scheitert
(Template-Jitter pro Epoche, bewegungsmodulierte Amplitude, He-Pumpe, BCG) →
das Modell lernt zu entfernen, was AAS liegen lässt.

## 4. Datensatz-Struktur (1 Artefakt ← 3 Kanäle × 7 Epochen)

- Beispiel = (Zielkanal `c`, Epoche `e`). Input **(7, 3, S)** = Zielkanal + 2
  kNN-Montage-Nachbarn × 7 aufeinanderfolgende Epochen. Target = Artefakt von
  `(c, e)` als `(1, S)`.
- Erhält die **reale Cross-Channel-Signatur** (gleichzeitiges Kanaltripel) und
  die **Cross-Epoch-Periodik** des Artefakts.
- kNN über die Montage (gleiche Logik wie `st_gnn` `knn_k`).

## 5. Augmentationen (Auswahl 1 / 2 / 4 + Kern)

- **Kern (immer):** korrekter Sub-sample-/Window-Shift aus dem **kontinuierlichen**
  Signal (Fenster bei `trigger + base_offset + δ` neu schneiden, dann resampeln);
  **nicht** der zirkuläre `np.roll` des aktuellen `TriggerJitter`. δ ganzzahlig
  und optional fraktional (sinc/`resample_poly`).
- **1 — Hintergrund-Mix (clean-swap):** realen Artefakt-Block
  `artifact[{c,n1,n2}, e-3..e+3]` + realen clean-Block `clean[{c,n1,n2}, e'-3..e'+3]`
  (`e' ≠ e`); **nur die Epochen-Paarung** wird randomisiert, nie Kanal-Identitäten,
  nie pro-Kanal — sonst stirbt die räumliche Signatur. Schaltbar via
  `background_mix_prob`.
- **2 — Amplituden-Jitter:** global *und* per-Kanal (Gain-Drift).
- **4 — TR-/Längen-Jitter + Messrauschen:** native Epochenlänge leicht variieren
  (resample 512·(1±ε)) + kleines additives Rauschen/Baseline-Drift.
- **Weggelassen:** räumliche Nachbar-Augmentation (3), `SignFlip`,
  `ChannelDropout` (letztere zerstören die Referenz).

Referenz und Target bekommen jede Augmentation **konsistent** (gleicher δ,
gleicher Mix-Block, gleicher Scale).

## 6. Bauplan / Scope

1. **Bundle:** `extract_niazy_artifact_signal.py` um den AAS→`PCACorrection(4)`-Pfad
   erweitern → neues `niazy_aas_pca4_direct`-Bundle (Artefakt = AAS+PCA, plus
   `corrected`, `triggers`, `sfreq`, `artifact_to_trigger_offset`, `ch_names`,
   Montage/Positionen für kNN).
2. **Builder:** `build_spatiotemporal_reference_dataset.py` → Output `(N,7,3,S)` +
   **guard-band** (Epoche ± `max_shift` echte Samples) für den online-Shift;
   `clean_source`-Schalter; Metadaten + Warnhinweis.
3. **Dataset-Klasse:** `NPZSpatioTemporalDataset` (mit `__len__/__getitem__/
   train_val_split/input_shape/target_shape/...`, kompatibel zum `facet-train`-Kontrakt).
4. **Transform:** `WindowShift` (continuous-window crop), `BackgroundMix`,
   `AmplitudeJitter`, `LengthJitterNoise`.
5. **Test:** billiger CPU-Smoke (winziges synthetisches Bundle, Shapes, ein paar
   Schritte, Shift/Mix-Invarianten als pure Funktionen).
6. **Spike-Injektions-Modus (Fundament für run_6):** optionaler clean-Modus, der
   bekannte Spikes (Library oder parametrisch) ins `clean_true` injiziert und eine
   `spike_labels`-Maske mitführt. Das ist die Datenbasis der Spike-Preservation-
   Evaluation in [`run_6_beat_aas_spike_preservation.md`](run_6_beat_aas_spike_preservation.md)
   — run_6 baut **keinen** eigenen Datensatz mehr, sondern nutzt diesen Modus.

Single-recording proof-fit zuerst; Multi-Recording später (eigentliche
Generalisierung).

## 6a. Stand der Umsetzung (umgesetzt 2026-06-18)

Alle sechs Bausteine aus §6 sind gebaut, getestet (13 CPU-Tests, ~4 s) und
**end-to-end auf der echten Niazy-Aufnahme** validiert:

| # | Artefakt | Datei |
|---|---|---|
| 1 | AAS+PCA(4)-Bundle-Extractor | [`tools/dataset_building/extract_niazy_aas_pca4_artifact.py`](../../tools/dataset_building/extract_niazy_aas_pca4_artifact.py) |
| 2 | Builder (Bibliothek) | [`src/facet/training/spatiotemporal_builder.py`](../../src/facet/training/spatiotemporal_builder.py) |
| 2 | Builder-CLI | [`tools/dataset_building/build_spatiotemporal_reference_dataset.py`](../../tools/dataset_building/build_spatiotemporal_reference_dataset.py) |
| 3 | `NPZSpatioTemporalDataset` | [`src/facet/training/dataset.py`](../../src/facet/training/dataset.py) |
| 4 | `WindowShift`/`BackgroundMix`/`AmplitudeJitter`/`LengthJitterNoise` | [`src/facet/training/dataset.py`](../../src/facet/training/dataset.py) |
| 5 | CPU-Smoke-Test | [`tests/test_spatiotemporal_reference_dataset.py`](../../tests/test_spatiotemporal_reference_dataset.py) |
| 6 | Spike-Injektions-Modus (`--inject-spikes`, `spike_labels`) | builder + dataset (`get_spike_labels`) |

**Echter Lauf (Faktor 2 → 4096 Hz):** 840 Trigger, 30 EEG-Kanäle, kombiniertes
Artefakt mean |art| ≈ 736 µV; daraus ein Weg-A-Set `(N, 7, 3, 512)` mit
guard-band 32 (`synthetic` clean) sowie eine Spike-Variante (`--inject-spikes`).

**Bewusste Abweichungen vom §6-Wortlaut:**
- Der Extractor ist eine **neue** Datei unter `tools/` statt einer Erweiterung
  von `examples/dataset_building/extract_niazy_artifact_signal.py` (die
  `examples/` gehören einem anderen Agenten und bleiben unangetastet).
- kNN-Positionen werden **nicht** ins Bundle geschrieben, sondern downstream aus
  `ch_names` über die `standard_1005`-Montage rekonstruiert (Alias-Map T3→T7 …,
  identisch zu `st_gnn`). Das Bundle bleibt damit format-gleich zu den
  AAS-Bundles.
- Das Dataset speichert **kein** `noisy_context` — `noisy = clean + artifact`
  wird pro Item rekonstruiert (spart ⅓ Platz/RAM und garantiert die Invariante).

## 7. Evaluation auch entkoppeln

Nur gegen „passt zu AAS" zu messen kann „besser als AAS" nicht zeigen. Nötig:
(i) unabhängiges clean-Target für synthetische Sets; (ii) für echte Aufnahmen
AAS-unabhängige Proxys (trigger-locked Residual-RMS, Spike-Preservation gegen
bekannte Spikes, Spektralmetriken) — teils in `evaluation_standard.md`.

## 8. Offene Punkte / Übergang

- OBS-Band festgelegt: `hp_freq=300` (HF-Residual, kein Hirn). **Datensatz + Eval
  voll-bandbreitig, kein 70-Hz-Low-pass** — das Artefakt wird in Gänze gelernt.
- Builder/Eval-Pipelines dürfen **keinen** `LowPassFilter` enthalten (sonst ginge
  genau der gelernte HF-Artefaktanteil verloren).
- **Epochen-Auflösung — geklärt (2026-06-18):** native Epoche ≈ 795 Samples
  (≈0,194 s bei 840 Triggern). Bei `core_samples=512` ist die effektive
  Abtastrate ≈ 2640 Hz → **Nyquist ≈ 1320 Hz**, also deutlich über 300 Hz: das
  >300-Hz-OBS-Residual bleibt erhalten, die Resample-Auflösung ist **nicht** der
  heimliche Low-pass. Wer mehr HF will (bis 2048 Hz beim 4096-Hz-Bundle), erhöht
  `core_samples` (≥ native) oder den Upsample-Faktor.
- `clean_source=external`: gibt es outside-scanner-/clean-EEG derselben Montage?
  (weiterhin offen — Builder unterstützt `--clean-source external --external-clean`.)
- Danach **Weg B** (self-supervised, ganz AAS-frei) als nächster Run —
  [`run_5_self_supervised_weg_b.md`](run_5_self_supervised_weg_b.md). Die hier
  gebaute (7×3)-Referenz und das entkoppelte Eval sind die Voraussetzung dafür.
