# Hinweise für zukünftige Agenten: FACET MATLAB Edition + FACETpy

Kurz-Runbook für diese Umgebung (Repo: `facetpy`, Branch-Präfix: `codex/`).

## Relevante Pfade
- MATLAB-Beispiel: `facet matlab edition/facet_matlab/src/CleanExJanik.m`
- FACETpy-Äquivalent: `examples/cleanexjanik_parity_pipeline.py`
- Datensatz: `examples/datasets/NiazyFMRI.set`
- MATLAB-Referenz-EDFs (für Stufenvergleich):
  - `examples/datasets/matlab_with_alignment.edf`
  - `examples/datasets/matlab_only_aas.edf`
  - `examples/datasets/matlab_with_pca.edf`
  - `examples/datasets/matlab_only_lowpass.edf`
  - `examples/datasets/matlab_with_anc.edf`

## Ausführung in dieser Umgebung
1. Abhängigkeiten installieren:
   - `poetry install`
2. C-Extension bauen (wichtig für ANC-Performance):
   - `HOME=/tmp MNE_HOME=/tmp poetry run build-fastranc`
3. Parity-Pipeline starten (klassisches Logging + channel-sequential):
   - `HOME=/tmp MNE_HOME=/tmp poetry run python examples/cleanexjanik_parity_pipeline.py`

Erwartete Outputs:
- `output/cleanexjanik_parity/corrected_cleanexjanik_equivalent.edf`
- `output/cleanexjanik_parity/comparison.json`
- `output/cleanexjanik_parity/findings.md`
- Logs: `output/cleanexjanik_parity/logs/`

## Bekannte Stolperfallen
- `NiazyFMRI.set` hat `Status` als EEG-Kanal, nicht als STIM.
  - Deshalb vor Trigger-Detektion zwingend remappen: `Status -> STIM`.
- Für ANC-Parität muss `artifact_length` nach Resampling in nativen Sample-Einheiten konsistent sein.
  - Fix ist bereits implementiert in `src/facet/preprocessing/resampling.py`.
  - Regressionstests: `tests/test_preprocessing.py` (`TestResampling`).
- MATLAB und FACETpy sind nicht vollständig 1:1:
  - Trigger-Heuristiken (`FindMissingTriggers`) und Pre-Filter-Details können abweichen.
  - Daher immer stufenweise vergleichen (Alignment/AAS/PCA/Lowpass/ANC), nicht nur Endsignal.

## MATLAB lokal ausführen (optional)
Nicht zwingend nötig für den aktuellen Vergleich, da Referenz-EDFs bereits unter `examples/datasets/` liegen.
Falls MATLAB-Ausführung nötig ist, CleanExJanik aus `facet matlab edition/facet_matlab/src/` starten und danach EDF-Exporte mit den FACETpy-Zwischenständen vergleichen.
