"""
FARM + volume-gap correction pipeline example.

This example demonstrates the two newly added correction processors:

1. ``VolumeArtifactCorrection`` (MATLAB ``RemoveVolumeArt`` equivalent)
2. ``FARMCorrection`` (MATLAB ``AvgArtWghtFARM``-style averaging)

The sequence follows the classic FACET ordering:
Load -> Trigger detection -> Cut -> Upsample -> Align -> Volume correction
-> FARM correction -> Downsample -> Paste -> Low-pass -> Export
"""

from pathlib import Path

from facet import (
    CutAcquisitionWindow,
    DownSample,
    EDFExporter,
    FARMCorrection,
    HighPassFilter,
    Loader,
    LowPassFilter,
    PasteAcquisitionWindow,
    Pipeline,
    SliceAligner,
    SubsampleAligner,
    TriggerDetector,
    UpSample,
    VolumeArtifactCorrection,
)

# ---------------------------------------------------------------------------
# Paths and shared settings
# ---------------------------------------------------------------------------
INPUT_FILE = "./examples/datasets/NiazyFMRI.edf"
OUTPUT_DIR = Path("./output")
OUTPUT_FILE = str(OUTPUT_DIR / "corrected_farm_volume.edf")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRIGGER_REGEX = r"\b1\b"
UPSAMPLE = 10


# ---------------------------------------------------------------------------
# Build the pipeline
# ---------------------------------------------------------------------------
pipeline = Pipeline(
    [
        Loader(path=INPUT_FILE, preload=True, artifact_to_trigger_offset=-0.005),
        TriggerDetector(regex=TRIGGER_REGEX),
        CutAcquisitionWindow(),
        HighPassFilter(freq=1.0),
        UpSample(factor=UPSAMPLE),
        SliceAligner(ref_trigger_index=0),
        SubsampleAligner(ref_trigger_index=0),
        # Equivalent to MATLAB's RemoveVolumeArt step. Skips automatically
        # when no volume gaps are detected in trigger spacing.
        VolumeArtifactCorrection(template_count=5, weighting_position=0.8, weighting_slope=20.0),
        # FARM-weighted artifact subtraction.
        FARMCorrection(window_size=30, correlation_threshold=0.9),
        DownSample(factor=UPSAMPLE),
        PasteAcquisitionWindow(),
        LowPassFilter(freq=70.0),
        EDFExporter(path=OUTPUT_FILE, overwrite=True),
    ],
    name="FARM + Volume Artifact Correction",
)


# ---------------------------------------------------------------------------
# Run and inspect results
# ---------------------------------------------------------------------------
result = pipeline.run(channel_sequential=True)
result.print_summary()

ctx = result.context
if ctx is not None:
    print(f"volume_gaps_detected={ctx.metadata.volume_gaps}")
    print(f"artifact_length_samples={ctx.metadata.artifact_length}")
    print(f"output_file={OUTPUT_FILE}")
