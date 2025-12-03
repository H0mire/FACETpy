# Vergleich der Evaluation: FACETpy vs. FACET Matlab Edition

Dieses Dokument vergleicht die Implementierung der Evaluationsmetriken in `facetpy` (Python) und der originalen `facet matlab edition` (Matlab).

## Zusammenfassung

| Metrik | Matlab Implementation | Python Implementation | Status |
|--------|----------------------|-----------------------|--------|
| **SNR** | `snr_residual` | `SNRCalculator` | Äquivalent |
| **Legacy SNR** | (Implizit in `Eval.m`) | `LegacySNRCalculator` | Äquivalent |
| **RMS Improvement** | `rms_correction` | `RMSCalculator` | Äquivalent |
| **RMS Residual** | `rms_residual` | `RMSResidualCalculator` | **Neu implementiert** |
| **Median Artifact** | `amp_median` | `MedianArtifactCalculator` | Äquivalent (inkl. Referenz-Ratio) |
| **FFT Allen** | `fft_allen` | `FFTAllenCalculator` | **Neu implementiert** |
| **FFT Niazy** | `fft_niazy` | `FFTNiazyCalculator` | **Neu implementiert** |

## Detaillierte Analyse

### 1. Signal-to-Noise Ratio (SNR)

Beide Implementierungen nutzen das Prinzip, die Varianz des korrigierten Signals (im Scanner) mit der Varianz eines "sauberen" Referenzsignals (außerhalb des Scanners) zu vergleichen.

*   **Formel:** $SNR = \frac{Var_{ref}}{Var_{residuum}}$ wobei $Var_{residuum} = Var_{korrigiert} - Var_{ref}$
*   **Matlab (`snr_residual`):**
    *   Nutzt Daten vor (`AcqPreStart`) und nach (`AcqPostEnd`) der Aufnahme als Referenz.
    *   Wendet Tiefpassfilter auf Referenzdaten an, um dem Processing der korrigierten Daten zu entsprechen.
*   **Python (`SNRCalculator`):**
    *   Nutzt ebenfalls Daten vor/nach dem Trigger-Fenster als Referenz (`ReferenceDataMixin`).
    *   Extrahiert Referenz aus dem `raw`-Objekt der aktuellen Processing-Stage.
    *   **Fazit:** Methodisch konsistent.

### 2. RMS Improvement Ratio

Misst das Verhältnis der Signalenergie vor und nach der Korrektur im Artefakt-Zeitraum. Ein höherer Wert zeigt an, dass mehr Artefakt-Energie entfernt wurde.

*   **Formel:** $Ratio = \frac{RMS_{uncorrected}}{RMS_{corrected}}$
*   **Matlab (`rms_correction`):** Nutzt `std()` (Standardabweichung).
*   **Python (`RMSCalculator`):** Nutzt explizite RMS-Berechnung.
*   **Fazit:** Identische Logik.

### 3. RMS Residual Ratio

Vergleicht das korrigierte Signal mit dem sauberen Referenzsignal. Der Wert sollte idealerweise 1.0 sein.

*   **Formel:** $Ratio = \frac{RMS_{corrected}}{RMS_{reference}}$
*   **Matlab (`rms_residual`):** Implementiert in `Eval.m`.
*   **Python (`RMSResidualCalculator`):** Neu implementiert.
*   **Fazit:** Äquivalent.

### 4. Median Artifact Amplitude

Misst die verbleibende Amplitude der Artefakte (Peak-to-Peak) und vergleicht sie mit der Referenz.

*   **Matlab (`amp_median`):** Wählt 10 Epochen aus und berechnet Median + Ratio zur Referenz.
*   **Python (`MedianArtifactCalculator`):** 
    *   Nutzt *alle* verfügbaren Epochen für stabilere Statistik.
    *   Berechnet nun auch das Verhältnis zur Referenz (`median_artifact_ratio`).
*   **Fazit:** Funktional äquivalent, Python nutzt mehr Daten (robuster).

### 5. Frequenz-Metriken (FFT)

Die zuvor fehlenden Metriken wurden implementiert:

*   **`FFTAllenCalculator` (`fft_allen`)**: 
    *   Vergleicht die spektrale Leistung in Frequenzbändern (Delta, Theta, Alpha, Beta) zwischen korrigierten Daten und Referenzdaten.
    *   Berechnet die prozentuale Abweichung.
*   **`FFTNiazyCalculator` (`fft_niazy`)**: 
    *   Analysiert spezifisch die Power bei der Volumen- und Slice-Frequenz (und deren Harmonischen).
    *   Vergleicht unkorrigierte vs. korrigierte Daten (in dB).
    *   Benötigt korrekte Slice/Volume-Informationen (werden nun im `TriggerDetector` berechnet).

## Status

Mit dem Update vom 12.01.2025 (`metrics.py`) sind alle Evaluationsmetriken der MATLAB-Edition in `facetpy` verfügbar. Die Pipeline bietet nun eine vollständige Parität in der Qualitätskontrolle.
