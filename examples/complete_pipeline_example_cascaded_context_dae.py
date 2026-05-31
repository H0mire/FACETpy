"""
Full fMRI artifact correction pipeline using the Cascaded Context DAE.

Same end-to-end workflow as ``complete_pipeline_example.py`` but the
classical AAS/FARM + PCA correction core is swapped for the trained
``CascadedContextDenoisingAutoencoderCorrection`` deep-learning processor:

  Load → DropChannels → Crop → TriggerExplorer → ArtifactOffsetFinder
  → HighPass → Upsample → Align → Downsample → CascadedContextDAE
  → LowPass → ANC → Export → Evaluate → Plot

The DAE adapter resamples each native epoch internally to its trained
``epoch_samples`` (512), so inference runs at the native recording rate.
The Upsample → TriggerAligner → Downsample block is kept only for
sub-sample trigger precision.
"""

from glob import glob
from pathlib import Path

from facet import (
    ANCCorrection,
    ArtifactOffsetFinder,
    Crop,
    DownSample,
    DropChannels,
    EDFExporter,
    FFTAllenCalculator,
    FFTNiazyCalculator,
    HighPassFilter,
    LegacySNRCalculator,
    Loader,
    LowPassFilter,
    MagicErasor,
    MedianArtifactCalculator,
    MetricsReport,
    Pipeline,
    RawPlotter,
    RMSCalculator,
    RMSResidualCalculator,
    SNRCalculator,
    TriggerAligner,
    UpSample,
)
from facet.config import set_config
from facet.models.cascaded_context_dae import (
    CascadedContextDenoisingAutoencoderCorrection,
)
from facet.preprocessing import TriggerExplorer

set_config(log_level="INFO", console_mode="modern")

# ---------------------------------------------------------------------------
# Paths and shared settings — adjust these for your study
# ---------------------------------------------------------------------------
INPUT_FILE  = "./examples/datasets/NiazyFMRI.edf"
OUTPUT_DIR  = Path("./output")
OUTPUT_FILE = str(OUTPUT_DIR / "corrected_cascaded_context_dae.edf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UPSAMPLE         = 10          # upsample factor for sub-sample trigger alignment
RECORDING_START  = 0           # seconds — crop start
RECORDING_END    = 162         # seconds — crop end (None keeps until the end)
NON_EEG_CHANNELS = ["EKG", "EMG", "EOG", "ECG"]
ARTIFACT_TO_TRIGGER_OFFSET = -0.005

# ---------------------------------------------------------------------------
# Cascaded Context DAE checkpoint — picks the newest Niazy proof-fit export.
# Train with:
#   uv run facet-train fit --config \
#     src/facet/models/cascaded_context_dae/training_niazy_proof_fit.yaml
# ---------------------------------------------------------------------------
_candidates = sorted(glob(
    "./training_output/cascadedcontextdenoisingautoencoderniazyprooffit_*/"
    "exports/cascaded_context_dae.ts"
))
if not _candidates:
    raise FileNotFoundError(
        "No cascaded_context_dae TorchScript checkpoint found. "
        "Train the model first (see header comment)."
    )
CHECKPOINT_PATH = _candidates[-1]
print(f"Using cascaded_context_dae checkpoint: {CHECKPOINT_PATH}")

# Toggle adaptive noise cancellation (requires the compiled C extension)
_has_anc = False

# ---------------------------------------------------------------------------
# Build the pipeline
# ---------------------------------------------------------------------------
steps = [
    # 1. Load
    Loader(path=INPUT_FILE, preload=True, artifact_to_trigger_offset=ARTIFACT_TO_TRIGGER_OFFSET),

    # 2. Remove non-EEG channels present in the EDF file
    DropChannels(channels=NON_EEG_CHANNELS),

    # 3. Limit analysis to acquisition window
    Crop(tmin=RECORDING_START, tmax=RECORDING_END),

    # 4. Detect fMRI slice-onset triggers (use auto_select=... for scripted runs)
    TriggerExplorer(),

    # 6. High-pass filter to remove slow drifts before correction
    HighPassFilter(freq=1.0),

    # 7. Upsample for sub-sample precision in trigger alignment
    UpSample(factor=UPSAMPLE // 2),

    # 8. Align all triggers to a shared reference using cross-correlation
    TriggerAligner(ref_trigger_index=0, upsample_for_alignment=False),

    # 10. Cascaded Context DAE — the primary correction step.
    #     Two-stage residual cascaded denoising autoencoder operating on a
    #     7-epoch trigger-aligned context per channel.
    CascadedContextDenoisingAutoencoderCorrection(
        checkpoint_path=CHECKPOINT_PATH,
        context_epochs=7,
        epoch_samples=512,
        device="cpu",          # change to "cuda" if a GPU is available
    ),

    # 9. Downsample back — the DAE adapter resamples each native epoch
    #    internally to ``epoch_samples`` (512).
    DownSample(factor=UPSAMPLE // 2),

    # 10b. Residual diagnostic: overlay the DAE's predicted artifact with the
    #      ORIGINAL noisy recording. Wherever the two curves diverge is exactly
    #      where the model's prediction misses part of the artifact — that
    #      gap is what survives as residual artifact in the corrected signal.
    #      ``overlay_original=True`` with ``source="prediction"`` triggers
    #      the residual-diagnostic mode.
    RawPlotter(
        mode="matplotlib",
        source="prediction",
        channel="Fp1",
        start=29.1,
        duration=0.2,
        overlay_original=True,
        save_path=str(OUTPUT_DIR / "predicted_artifact_cascaded_context_dae.png"),
        show=True,
        auto_close=False,
        title="Fp1 — Cascaded Context DAE: predicted artifact vs original noisy",
    ),

    # 11. Low-pass filter to remove high-frequency noise
    LowPassFilter(freq=70.0),

    # 12. Adaptive Noise Cancellation (requires the compiled C extension)
    *([ANCCorrection(use_c_extension=True)] if _has_anc else []),

    MagicErasor(),

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
        mode="mne",
        channel="Fp1",
        start=25.0,
        duration=20.0,
        overlay_original=True,
        save_path=str(OUTPUT_DIR / "before_after_cascaded_context_dae.png"),
        show=True,
        auto_close=False,
        title="Fp1 — Before vs After Cascaded Context DAE Correction",
    ),
]

pipeline = Pipeline(steps, name="Cascaded Context DAE Correction Pipeline")

# ---------------------------------------------------------------------------
# Run and inspect results
# ---------------------------------------------------------------------------
result = pipeline.run(channel_sequential=False)
result.print_summary()
result.print_metrics()
