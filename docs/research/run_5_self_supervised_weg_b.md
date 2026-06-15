# Run 5 — Self-supervised, AAS-freie Artefaktschätzung (Weg B)

> **Status: NÄCHSTER Run nach Weg A.** Dies ist **Weg B** der AAS-Entkopplung
> aus der Datensatz-Diskussion (2026-06-11): Training **ganz ohne clean-Target**.
> Setzt auf der in Weg A gebauten (7 Epochen × 3 Kanäle)-Referenz und dem
> entkoppelten Evaluations-Setup auf —
> [`run_3_decoupled_dataset_weg_a.md`](run_3_decoupled_dataset_weg_a.md).

---

## 0. TL;DR

Kein clean-Target, kein AAS-Target. Die Supervision kommt allein aus der
**Struktur der Daten**: das Gradientenartefakt ist über Epochen und
Nachbarkanäle stark **geteilt/korreliert**, das EEG **nicht**. Daraus lässt sich
das Artefakt trennen, ohne dass je „die Wahrheit" als Label vorkommt → die Decke
ist nicht mehr AAS.

## 1. Warum überhaupt Weg B, wenn es Weg A gibt?

Weg A entkoppelt über eine *unabhängige clean-Quelle* (synthetisch/extern) — gut,
aber: synthetisches clean ist nur so gut wie das EEG-Modell, und externes
clean-EEG derselben Montage ist oft nicht verfügbar. Weg B braucht **gar kein**
clean — er trainiert direkt auf den **echten verrauschten** EEG-fMRI-Daten und
kann damit reale Residual-/Spike-Effekte erfassen, die kein synthetisches Set
abbildet. A und B sind komplementär: A liefert das messbare, entkoppelte Eval,
B liefert ein Modell, das nie an einer Methode (AAS) oder einem Surrogat klebt.

## 2. Prinzip: geteilt vs. nicht geteilt

- **Artefakt:** deterministisch durch die MR-Sequenz → über aufeinanderfolgende
  Epochen quasi-identisch, über Nachbarkanäle räumlich glatt/korreliert.
- **EEG:** stochastisch, über Epochen *nicht* wiederholt, andere räumliche
  Quellprojektion.

Ein Prädiktor, der nur den **über Epochen/Kanäle reproduzierbaren** Anteil
ausgeben darf, lernt zwangsläufig das Artefakt — das EEG ist der Teil, der sich
*nicht* aus den anderen Epochen/Kanälen vorhersagen lässt. Genau dafür ist die
(7×3)-Referenz aus Weg A gebaut.

## 3. Konkrete Trainingsziele (Kandidaten)

1. **Noise2Noise über Epochen.** Zwei verschiedene Epochen `e1, e2` desselben
   Kanals tragen ~dasselbe Artefakt, aber *unabhängiges* EEG. Das Modell sagt aus
   der Referenz von `e1` das *noisy* von `e2` (oder dessen Artefaktanteil) voraus.
   Da das EEG zwischen `e1` und `e2` unkorreliert ist, minimiert der Erwartungswert
   genau den geteilten Anteil = Artefakt (Lehmann/Noise2Noise-Argument).
2. **Cross-Channel-/Cross-Epoch-Prädiktion + Konsistenz.** Artefakt des Zielkanals
   aus den Nachbarkanälen und Nachbar-Epochen schätzen; erzwingen, dass die
   Schätzung über verschiedene Referenz-Teilmengen konsistent ist (die räumliche
   Signatur soll stabil sein, das EEG-Residual nicht).
3. **Noise2Self / Maskierung.** Eine Epoche/ein Kanal wird maskiert und aus dem
   Rest rekonstruiert; nur der vorhersagbare (geteilte) Anteil kann getroffen
   werden → Artefakt.
4. **Periodizitäts-/Low-rank-Prior.** Das Artefakt liegt in einem niedrigrangigen,
   trigger-periodischen Unterraum (vgl. OBS/AAS-Template); ein Strukturterm
   (Rank-/Periodizitäts-Penalty) trennt es vom breitbandigen EEG — ganz ohne
   Label.

## 4. Architektur-Fit

Direkt nutzbar aus den `_paper_accurate_edition`-Modellen:
- **`st_gnn`** — räumlicher Graph über die 3+ Nachbarkanäle (Cross-Channel-Teil).
- **`conv_tasnet` / `demucs`** — Source-Separation-Köpfe (geteilt vs. nicht
  geteilt = zwei „Quellen").
- **`cascaded_context_dae`** — 7-Epochen-Kontext-Prädiktion ist schon die
  natürliche Form für Ziel (1)/(3).

Kein neuer facet-train-Loss-Kontraktbruch nötig: die Self-Supervision lässt sich
als (input, pseudo-target)-Paare im Dataset realisieren (z.B. `e1`→`e2`), sodass
der bestehende `loss_fn(model(x), target)`-Pfad genügt.

## 5. Risiken & Gegenmaßnahmen

- **EEG-Leck** (Modell sagt auch EEG vorher): Referenz- und Zielepoche zeitlich
  genug trennen; Noise2Noise-Paare mit unkorreliertem EEG wählen.
- **Kollaps** (Modell gibt 0/Mittelwert): Strukturterm + Validierung gegen das
  *entkoppelte* Eval aus Weg A.
- **Stabilität** (gerade bei GAN-/Diffusionsmodellen): zuerst die einfachen
  Regressionsziele (1)/(3), GAN/Diffusion erst danach.
- **Bewegungs-/nicht-stationäres Artefakt:** verletzt „über Epochen identisch";
  daher kurze Epochen-Fenster und/oder bewegungsrobuste Referenzauswahl.

## 6. Bezug zu Weg A

- **Datenstruktur:** identische (7×3)-Referenz + guard-band aus Weg A.
- **Evaluation:** das in Weg A aufgebaute *AAS-unabhängige* Eval ist die einzige
  Möglichkeit, „besser als AAS" für ein label-freies Modell überhaupt zu zeigen.
- **Reihenfolge:** Weg A zuerst (entkoppeltes Set + Eval + Baselines), dann
  Weg B als der eigentlich AAS-freie Schritt.

## 7. Deliverables (wenn der Run startet)

- Dataset-Modus, der self-supervised-Paare erzeugt (`e1`→`e2`, Maskierung).
- 1–2 self-supervised-Trainingsziele auf einem leichten Modell (z.B.
  `cascaded_context_dae` / `st_gnn`) als Machbarkeit.
- Vergleich gegen AAS und gegen die Weg-A-Modelle auf dem entkoppelten Eval.
