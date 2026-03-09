"""
Full fMRI + BCG artifact correction pipeline.

This is the reference example showing the complete, publication-quality
correction workflow. It covers every recommended step:

  Load → DropChannels → Crop → TriggerExplorer → ArtifactOffsetFinder
  → Filter → ReferenceIntervalSelector → Upsample → Align → AAS → PCA
  → Downsample → Filter → BCG(QRS + AAS) → ANC → Export → Evaluate → Plot

ReferenceIntervalSelector lets you pick a clean reference interval for
metrics. SignalIntervalSelector lets you pick the evaluated signal interval
(acquisition) when boundaries are unclear after correction. TriggerExplorer
discovers and selects trigger sources; ArtifactOffsetFinder aligns the artifact
window with the data. Use auto_select with TriggerExplorer for non-interactive
runs.

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
    ANCCorrection,
    ArtifactOffsetFinder,
    Crop,
    MagicErasor,
    Pipeline,
    Loader,
    EDFExporter,
    TriggerAligner,
    HighPassFilter,
    LowPassFilter,
    UpSample,
    DownSample,
    DropChannels,
    QRSTriggerDetector,
    RawTransform,
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
)
from facet.evaluation import ReferenceIntervalSelector, SignalIntervalSelector
from facet.preprocessing import TriggerExplorer

# ---------------------------------------------------------------------------
# Paths and shared settings — adjust these for your study
# ---------------------------------------------------------------------------
INPUT_FILE  = "./examples/datasets/NiazyFMRI.edf"
OUTPUT_DIR  = Path("./output")
OUTPUT_FILE = str(OUTPUT_DIR / "corrected_full_with_bcg.edf")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRIGGER_REGEX    = r"\b1\b"   # regex for auto_select (None = interactive TriggerExplorer)
UPSAMPLE         = 10          # upsample factor for sub-sample trigger alignment
RECORDING_START  = 0           # seconds — crop start
RECORDING_END    = 162         # seconds — crop end (None keeps until the end)

# Optional: list channel names to drop before processing (non-EEG channels).
# Keep ECG for BCG correction.
NON_EEG_CHANNELS = ["EMG", "EOG"]


def _remove_channel_from_bads(raw, channel_name):
    raw_copy = raw.copy()
    raw_copy.info["bads"] = [ch for ch in raw_copy.info["bads"] if ch != channel_name]
    return raw_copy

# ---------------------------------------------------------------------------
# Enable costly ANC correction
# ---------------------------------------------------------------------------
_has_anc = False

# ---------------------------------------------------------------------------
# Build the pipeline
# ---------------------------------------------------------------------------
steps = [
    # 1. Load
    Loader(path=INPUT_FILE, preload=True),

    # 2. Remove non-EEG channels present in the EDF file
    DropChannels(channels=NON_EEG_CHANNELS),

    # 3. Limit analysis to acquisition window
    Crop(tmin=RECORDING_START, tmax=RECORDING_END),

    # 4. Detect fMRI slice-onset triggers (use auto_select=TRIGGER_REGEX for scripted runs)
    TriggerExplorer(),

    # 5. Interactively align artifact window to trigger
    ArtifactOffsetFinder(),

    # Optional: pick evaluated reference interval manually if acquisition contains unhandled artifacts
    # ReferenceIntervalSelector(),

    RawPlotter(
        mode="mne",
        channel="Fp1",
        start=25.0,
        duration=20.0,
        overlay_original=True,
        save_path=str(OUTPUT_DIR / "before_after.png"),
        show=True,
        auto_close=True,
        title="Fp1 — Before Correction",
    ),

    # 6. High-pass filter to remove slow drifts before correction
    HighPassFilter(freq=1.0),

    # 7. Select clean reference interval for downstream metrics
    # ReferenceIntervalSelector(),

    # 8. Upsample for sub-sample precision in trigger alignment
    UpSample(factor=UPSAMPLE),

    # 9. Align all triggers to a shared reference using cross-correlation
    TriggerAligner(ref_trigger_index=0, upsample_for_alignment=False),

    # 10. Averaged Artifact Subtraction — the primary correction step
    AASCorrection(
        window_size=30,
        correlation_threshold=0.975,
        realign_after_averaging=True,
    ),

    # 11. PCA — remove systematic residual artifact components
    PCACorrection(n_components=0.95, hp_freq=1.0),

    # 12. Downsample back to the original recording rate
    DownSample(factor=UPSAMPLE),

    # 13. Low-pass filter to remove high-frequency noise
    LowPassFilter(freq=70.0),

    # 14. Ensure ECG is not excluded before BCG trigger/offset handling
    RawTransform(
        "remove_ecg_from_bads",
        lambda raw: _remove_channel_from_bads(raw, "ECG"),
    ),

    # 15. BCG correction (QRS-triggered AAS on cardiac cycle)
    #QRSTriggerDetector(),
    #ArtifactOffsetFinder(channel="ECG"),
    #AASCorrection(window_size=20),
]

# 16. Adaptive Noise Cancellation (requires the compiled C extension)
if _has_anc:
    steps.append(ANCCorrection(use_c_extension=True))

steps += [
    MagicErasor(),
    # 17. Save corrected recording
    EDFExporter(path=OUTPUT_FILE, overwrite=True),
    # Optional: pick evaluated signal interval manually if acquisition contains unhandled artifacts
    # SignalIntervalSelector(),
    # 18. Compute evaluation metrics
    SNRCalculator(),
    LegacySNRCalculator(),
    RMSCalculator(),
    RMSResidualCalculator(),
    MedianArtifactCalculator(),
    FFTAllenCalculator(),
    FFTNiazyCalculator(),
    MetricsReport(),

    # 19. Plot a before/after comparison for a single channel
    RawPlotter(
        mode="mne",
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

pipeline = Pipeline(steps, name="Full fMRI + BCG Correction Pipeline")

# ---------------------------------------------------------------------------
# Run and inspect results
# ---------------------------------------------------------------------------
result = pipeline.run(channel_sequential=False)

# result.get_raw().plot(n_channels=20)

# One-liner summary: Done / Failed, execution time, key metric values
result.print_summary()

# Full table of every metric that was calculated
result.print_metrics()
