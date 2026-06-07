"""
Full fMRI artifact correction pipeline — VOLUME-TRIGGER variant.

Copy of ``complete_pipeline_example_large_dataset.py`` adapted for recordings
whose markers are VOLUME triggers (one per fMRI volume / TR) instead of slice
triggers. It adds two MATLAB-FACET-faithful steps:

  * ``SliceTriggerGenerator`` — expands each volume trigger into N slice
    triggers (MATLAB ``GenerateSliceTriggers``). Must run right after trigger
    detection, before high-pass / upsampling.
  * ``VolumeArtifactCorrection`` — subtracts the volume-transition artifact from
    the slices bordering each inter-volume gap and interpolates the gap itself
    (MATLAB ``RARemoveVolumeArtifact`` / the ``'RemoveVolumeArt'`` RASequence
    step). Runs after alignment, before AAS — exactly MATLAB's position.

Pipeline:

  Load → DropCh → Detect (volume) → Slice-gen → Filter → Upsample → Align
  → RemoveVolumeArt → AAS → PCA → Downsample → ANC → Export → Evaluate → Plot

For shorter introductions, see:
  quickstart.py         — minimal pipeline (load, AAS, export)
  evaluation.py         — metrics and pipeline comparison
  advanced_workflows.py — conditional steps, parallel execution, factory
  batch_processing.py   — processing many files at once
  inline_steps.py       — custom def steps and the pipe operator
  synthetic_eeg.py      — generating synthetic EEG for testing
"""

from pathlib import Path

from mne import verbose

from facet import (
    ANCCorrection,
    TriggerEditor,
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
    AASCorrection,
    PCACorrection,
    VolumeArtifactCorrection,
    SNRCalculator,
    LegacySNRCalculator,
    RMSCalculator,
    RMSResidualCalculator,
    MedianArtifactCalculator,
    FFTAllenCalculator,
    FFTNiazyCalculator,
    MetricsReport,
    RawPlotter,
    load,
)
from facet.config import set_config
from facet.evaluation import ReferenceIntervalSelector
from facet.evaluation.metrics import SignalIntervalSelector
from facet.helpers.interactive import TriggerEditor
from facet.preprocessing import TriggerExplorer, SliceTriggerGenerator

import os

from facet.preprocessing.alignment import SubsampleAligner

# Ensure that per-run log files are created by setting the FACET_LOG_FILE environment variable.
os.environ["FACET_LOG_FILE"] = "1"
set_config(log_level="INFO", console_mode="modern")

# ---------------------------------------------------------------------------
# Paths and shared settings — adjust these for your study
# ---------------------------------------------------------------------------
INPUT_FILE  = "/Volumes/JanikProSSD/DataSets/EEG Datasets/EEGfMRI_20250519_20180312_004257.mff"
OUTPUT_DIR  = Path("./output")
OUTPUT_FILE = str(OUTPUT_DIR / "corrected_EEGfMRI_20250519_20180312_004257.edf")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRIGGER_REGEX    = r"^TR\s+\d+$"   # regex matching the fMRI VOLUME (TR) trigger
UPSAMPLE         = 5          # upsample factor for sub-sample trigger alignment
RECORDING_START  = 0        # seconds — crop start: triggers begin at ~1307 s
RECORDING_END    = None        # seconds — crop end (None keeps until the end)

# Optional: list channel names to drop before processing (non-EEG channels)
NON_EEG_CHANNELS = ["EKG", "EMG", "EOG", "ECG"]


# ---------------------------------------------------------------------------
# Enable costly ANC correction
# ---------------------------------------------------------------------------

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

    # 3. Remove unrelevant data
    # Crop(tmin=RECORDING_START, tmax=RECORDING_END),

    # 4. Detect fMRI VOLUME (TR) triggers
    TriggerExplorer(),
    TriggerEditor(),

    ReferenceIntervalSelector(),
    # 5. High-pass filter to remove slow drifts before correction
    HighPassFilter(freq=1.0),

    # 6. Upsample for sub-sample precision in trigger alignment
    UpSample(factor=UPSAMPLE),

    # 7. Align all triggers to a shared reference using cross-correlation
    TriggerAligner(ref_trigger_index=0, upsample_for_alignment=False),
    SubsampleAligner(),

    # 7b. Remove the volume-transition artifact (MATLAB 'RemoveVolumeArt').
    #     Runs after alignment, before AAS — exactly MATLAB's RASequence order.
    #     Self-skips when metadata.volume_gaps is False (no inter-volume gap),
    #     so it is safe to keep in unconditionally.
    VolumeArtifactCorrection(),

    # 8. Averaged Artifact Subtraction — the primary correction step
    AASCorrection(
        window_size=30,
        correlation_threshold=0.975,
        realign_after_averaging=True,
    ),

    # 9. PCA — remove systematic residual artifact components
    PCACorrection(n_components=0.95, hp_freq=70.0),

    # 10. Downsample back to the original recording rate
    DownSample(factor=UPSAMPLE),

    # 11. Low-pass filter to remove high-frequency noise
    LowPassFilter(freq=70.0),
]

# 12. Adaptive Noise Cancellation (requires the compiled C extension)
if _has_anc:
    steps.append(ANCCorrection(use_c_extension=True))

steps += [
    MagicErasor(),

    SignalIntervalSelector(),
    # 13. Save corrected recording
    EDFExporter(path=OUTPUT_FILE, overwrite=True),
    # 14. Compute evaluation metrics
    SNRCalculator(verbose=True),
    LegacySNRCalculator(verbose=True),
    RMSCalculator(verbose=True),
    RMSResidualCalculator(verbose=True),
    MedianArtifactCalculator(verbose=True),
    FFTAllenCalculator(verbose=True),
    FFTNiazyCalculator(verbose=True),
    MetricsReport(),

    # 15. Plot a before/after comparison for a single channel
    lambda ctx: ctx | RawPlotter(
           mode="mne",
           channel="Fp1",
           duration=20,  # full recording length
           overlay_original=False,
           save_path=str(OUTPUT_DIR / "before_after.png"),
           show=True,
           title="Fp1 — Before vs After Correction",
       ),
]
pipeline = Pipeline(steps, name="Full fMRI Correction Pipeline (volume→slice + RemoveVolumeArt)")


# ---------------------------------------------------------------------------
# Run and inspect results
# ---------------------------------------------------------------------------
result = pipeline.run(channel_sequential=True)

# One-liner summary: Done / Failed, execution time, key metric values
result.print_summary()

# Full table of every metric that was calculated
result.print_metrics()
