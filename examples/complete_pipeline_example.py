"""
Full fMRI artifact correction pipeline.

This is the reference example showing the complete, publication-quality
correction workflow. It covers every recommended step:

  Load → Crop → Filter → Upsample → Align → AAS → PCA → Downsample
  → ANC → Export → Evaluate → Plot

For shorter introductions, see:
  quickstart.py         — minimal pipeline (load, AAS, export)
  evaluation.py         — metrics and pipeline comparison
  advanced_workflows.py — conditional steps, parallel execution, factory
  batch_processing.py   — processing many files at once
  inline_steps.py       — custom def steps and the pipe operator
  synthetic_eeg.py      — generating synthetic EEG for testing
"""

from pathlib import Path

from facet import (
    ArtifactOffsetFinder,
    Pipeline,
    Loader,
    EDFExporter,
    TriggerDetector,
    TriggerAligner,
    HighPassFilter,
    LowPassFilter,
    UpSample,
    DownSample,
    Crop,
    DropChannels,
    AASCorrection,
    PCACorrection,
    SNRCalculator,
    LegacySNRCalculator,
    RMSCalculator,
    RMSResidualCalculator,
    MedianArtifactCalculator,
    FFTAllenCalculator,
    FFTNiazyCalculator,
    MetricsReport,
    RawPlotter,
    WaitForConfirmation,
    load,
)
from facet.preprocessing import TriggerExplorer

# ---------------------------------------------------------------------------
# Paths and shared settings — adjust these for your study
# ---------------------------------------------------------------------------
INPUT_FILE  = "/Volumes/JanikProSSD/DataSets/EEG Datasets/EEGfMRI_20250519_20180312_004257.mff"
OUTPUT_DIR  = Path("./output")
OUTPUT_FILE = str(OUTPUT_DIR / "corrected_full.edf")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRIGGER_REGEX    = r"^TR\s+\d+$"   # regex matching the fMRI slice trigger value
UPSAMPLE         = 10          # upsample factor for sub-sample trigger alignment
RECORDING_START  = 0        # seconds — crop start: triggers begin at ~1307 s
RECORDING_END    = None        # seconds — crop end (None keeps until the end)

# Optional: list channel names to drop before processing (non-EEG channels)
NON_EEG_CHANNELS = ["EKG", "EMG", "EOG", "ECG"]


# ---------------------------------------------------------------------------
# Try importing optional correction methods
# ---------------------------------------------------------------------------
try:
    from facet import ANCCorrection
    _has_anc = True
except ImportError:
    _has_anc = False

# ---------------------------------------------------------------------------
# Build the pipeline
# ---------------------------------------------------------------------------
steps = [
    # 1. Load
    Loader(
        path=INPUT_FILE,
        preload=True,
    ),

    # 2. Remove non-EEG channels present in the EDF file
    DropChannels(channels=NON_EEG_CHANNELS),

    # 3. Limit the analysis to the acquisition window
    # Crop(tmin=RECORDING_START, tmax=RECORDING_END),

    # 4. Detect fMRI slice-onset triggers
    TriggerExplorer(),

    ArtifactOffsetFinder(),

    # 5. High-pass filter to remove slow drifts before correction
    HighPassFilter(freq=1.0),

    # 6. Upsample for sub-sample precision in trigger alignment
    UpSample(factor=UPSAMPLE),

    # 7. Align all triggers to a shared reference using cross-correlation
    TriggerAligner(ref_trigger_index=0, upsample_for_alignment=False),

    # 8. Averaged Artifact Subtraction — the primary correction step
    AASCorrection(
        window_size=30,
        correlation_threshold=0.975,
        realign_after_averaging=True,
    ),

    # 9. PCA — remove systematic residual artifact components
    PCACorrection(n_components=0.95, hp_freq=1.0),

    # 10. Downsample back to the original recording rate
    DownSample(factor=UPSAMPLE),

    # 11. Low-pass filter to remove high-frequency noise
    LowPassFilter(freq=70.0),
]

# 12. Adaptive Noise Cancellation (requires the compiled C extension)
if _has_anc:
    steps.append(ANCCorrection(use_c_extension=True))

steps += [
    # 13. Save corrected recording
    EDFExporter(path=OUTPUT_FILE, overwrite=True),

    # 14. Compute evaluation metrics
    SNRCalculator(),
    LegacySNRCalculator(),
    RMSCalculator(),
    RMSResidualCalculator(),
    MedianArtifactCalculator(),
    FFTAllenCalculator(),
    FFTNiazyCalculator(),
    MetricsReport(),

    # 15. Plot a before/after comparison for a single channel
    RawPlotter(
        mode="matplotlib",
        channel="Fp1",
        start=25.0,
        duration=20.0,
        overlay_original=True,
        save_path=str(OUTPUT_DIR / "before_after.png"),
        show=True,
        auto_close=True,
        title="Fp1 — Before vs After Correction",
    ),
]

pipeline = Pipeline(steps, name="Full fMRI Correction Pipeline")


# ---------------------------------------------------------------------------
# Run and inspect results
# ---------------------------------------------------------------------------
result = pipeline.run(channel_sequential=True)

# One-liner summary: Done / Failed, execution time, key metric values
result.print_summary()

# Full table of every metric that was calculated
result.print_metrics()
