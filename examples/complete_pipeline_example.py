"""
Complete FACETpy Pipeline Example

This example demonstrates a full end-to-end artifact correction pipeline
using the new modular architecture.

Author: FACETpy Team
Date: 2025-01-12
"""

import os
from pathlib import Path
import traceback

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MPLCONFIG_PATH = OUTPUT_DIR / "mpl_config"
MPLCONFIG_PATH.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_PATH.resolve()))

from facet.core import Pipeline, LambdaProcessor
from facet.io import EDFLoader, EDFExporter
from facet.preprocessing import (
    HighPassFilter,
    LowPassFilter,
    UpSample,
    TriggerDetector,
    TriggerAligner,
    DownSample
)
from facet.correction import (
    AASCorrection,
    ANCCorrection,
    PCACorrection
)
from facet.evaluation import (
    SNRCalculator,
    LegacySNRCalculator,
    RMSCalculator,
    MedianArtifactCalculator,
    MetricsReport,
    RawPlotter
)
from facet.helpers import WaitForConfirmation

file_path = "./examples/datasets/NiazyFMRI.edf"
# Event Regex assuming using stim channel
event_regex = r"\b1\b"
# Upsampling factor
upsample_factor = 10
# unwanted channels
unwanted_bad_channels = ["EKG", "EMG", "EOG", "ECG"]
# Add Artifact to Trigger Offset in seconds. Adjust this if the trigger events are not aligned with the artifact occurence
artifact_to_trigger_offset = -0.005
# Crop window (seconds). Adjust as needed to trim the recording before processing.
crop_tmin = 0
crop_tmax = 162

def main():
    """
    Run a complete fMRI artifact correction pipeline.

    Steps:
    1. Load EEG data and optionally crop to a time window
    2. Detect triggers
    3. Upsample for better precision
    4. Align triggers using cross-correlation
    5. Apply AAS correction
    6. Apply ANC correction (optional)
    7. Apply PCA correction (optional)
    8. Downsample back to original rate
    9. Apply final highpass filter
    10. Evaluate results
    11. Visualise before/after snippets
    12. Optionally pause for manual inspection
    13. Export corrected data
    """

    # Configuration
    input_file = "./examples/datasets/NiazyFMRI.edf"
    output_file = str(OUTPUT_DIR / "corrected.edf")
    trigger_pattern = r"\b1\b"  # Regex pattern for trigger detection

    # Build the pipeline
    pipeline = Pipeline([
        # 1. Load data
        EDFLoader(
            path=input_file,
            preload=True,
            bad_channels=unwanted_bad_channels,
            artifact_to_trigger_offset=artifact_to_trigger_offset,
        ),

        # 1b. Optionally crop the raw to a specific interval before further processing
        LambdaProcessor(
            name="crop_raw_to_interval",
            func=lambda ctx: ctx.with_raw(
                ctx.get_raw().copy().crop(tmin=crop_tmin, tmax=crop_tmax)
            )
        ),

        # 2. Detect triggers
        TriggerDetector(
            regex=trigger_pattern
        ),

        HighPassFilter(
            freq=0.5
        ),

        # 3. Upsample for precision
        UpSample(
            factor=10
        ),

        # 4. Align triggers
        TriggerAligner(
            ref_trigger_index=0,
            upsample_for_alignment=False  # Already upsampled
        ),

        # 5. AAS Correction (main algorithm)
        AASCorrection(
            window_size=30,
            correlation_threshold=0.975,
            realign_after_averaging=True
        ),

        # 6. Downsample back to original rate
        DownSample(
            factor=10
        ),

        LowPassFilter(
            freq=70
        ),

        # 9. PCA Correction (optional, for systematic artifacts)
        PCACorrection(
            n_components=0.95,  # Keep 95% of variance
            hp_freq=1.0
        ),

        # 8. ANC Correction (for residual artifacts)
        ANCCorrection(
            use_c_extension=True
        ),

        # 10. Evaluate results
        SNRCalculator(),
        LegacySNRCalculator(),
        RMSCalculator(),
        MedianArtifactCalculator(),
        MetricsReport(),

        # 11. Visualise results (configurable plotting step)
        RawPlotter(
            mode="matplotlib",  # Switch to "mne" for interactive Raw.plot()
            channel="Fp1",
            start=25.0,
            duration=120.0,
            overlay_original=True,
            save_path=str(OUTPUT_DIR / "pipeline_before_after.png"),
            show=True,
            auto_close=False,
            title="Fp1 – Before vs After (first 120s)"
        ),

        # 12. Optional confirmation pause (auto-continues in scripted runs)
        WaitForConfirmation(
            message="Review the generated plot, then press Enter to continue.",
            auto_continue=False
        ),

        # 13. Export corrected data
        EDFExporter(
            path=output_file,
            overwrite=True
        )
    ], name="fMRI Artifact Correction Pipeline")

    # Run the pipeline
    print("Starting pipeline execution...")
    result = pipeline.run()

    if result.success:
        print("\n✓ Pipeline completed successfully!")
        print(f"Execution time: {result.execution_time:.2f} seconds")

        # Access metrics
        metrics = result.context.metadata.custom.get('metrics', {})
        print("\nFinal Metrics:")
        print(f"  SNR: {metrics.get('snr', 'N/A')}")
        print(f"  Legacy SNR: {metrics.get('legacy_snr', 'N/A')}")
        print(f"  RMS Ratio: {metrics.get('rms_ratio', 'N/A')}")
        print(f"  Median Artifact: {metrics.get('median_artifact', 'N/A')}")

        plot_output = OUTPUT_DIR / "pipeline_before_after.png"
        if plot_output.exists():
            print(f"\nVisual inspection plot saved to: {plot_output}")
            print("Adjust the RawPlotter step to change channels, duration, or plotting mode.")

    else:
        print(f"\n✗ Pipeline failed: {result.error}")
        traceback.print_exception(type(result.error), result.error, result.error.__traceback__)
        if result.failed_processor:
            print(f"Failed at: {result.failed_processor}")


def example_parallel_execution():
    """
    Example of running pipeline with parallel processing.
    """
    input_file = "path/to/your/data.edf"
    output_file = "path/to/output/corrected.edf"

    pipeline = Pipeline([
        EDFLoader(path=input_file, preload=True, artifact_to_trigger_offset=-0.005),
        TriggerDetector(regex=r"\b1\b", ),
        UpSample(factor=10),
        TriggerAligner(ref_trigger_index=0),

        # AAS correction with channel-wise parallelization
        AASCorrection(
            window_size=30,
            correlation_threshold=0.975
        ),

        DownSample(factor=10),
        HighPassFilter(freq=0.5),
        EDFExporter(path=output_file, overwrite=True)
    ], name="Parallel Pipeline")

    # Run with parallel execution (automatically parallelizes compatible processors)
    result = pipeline.run(parallel=True, n_jobs=-1)  # -1 = use all cores

    print(f"Pipeline completed in {result.execution_time:.2f}s")


def example_minimal_pipeline():
    """
    Minimal pipeline for basic correction.
    """
    pipeline = Pipeline([
        EDFLoader(path="data.edf", preload=True),
        TriggerDetector(regex=r"\b1\b"),
        UpSample(factor=10),
        AASCorrection(window_size=30),
        DownSample(factor=10),
        EDFExporter(path="corrected.edf")
    ], name="Minimal Pipeline")

    return pipeline.run()


def example_custom_workflow():
    """
    Example of building a custom workflow with conditional processing.
    """
    from facet.core import ConditionalProcessor

    def should_apply_pca(context):
        """Only apply PCA if SNR is below threshold."""
        metrics = context.metadata.custom.get('metrics', {})
        snr = metrics.get('snr', float('inf'))
        return snr < 10  # Apply PCA if SNR is poor

    pipeline = Pipeline([
        EDFLoader(path="data.edf", preload=True),
        TriggerDetector(regex=r"\b1\b"),
        UpSample(factor=10),
        AASCorrection(window_size=30),

        # Calculate intermediate SNR
        SNRCalculator(),

        # Conditionally apply PCA based on SNR
        ConditionalProcessor(
            condition=should_apply_pca,
            processor=PCACorrection(n_components=0.95)
        ),

        DownSample(factor=10),
        MetricsReport(),
        EDFExporter(path="corrected.edf")
    ])

    return pipeline.run()


def example_batch_processing():
    """
    Example of processing multiple files with the same pipeline.
    """
    from facet.core import ProcessingContext

    # Define pipeline (reusable)
    correction_pipeline = Pipeline([
        TriggerDetector(regex=r"\b1\b"),
        UpSample(factor=10),
        AASCorrection(window_size=30),
        DownSample(factor=10),
        HighPassFilter(freq=0.5),
        ANCCorrection()
    ])

    # Process multiple files
    input_files = [
        "subject1.edf",
        "subject2.edf",
        "subject3.edf"
    ]

    for input_file in input_files:
        print(f"\nProcessing {input_file}...")

        # Load data
        loader = EDFLoader(path=input_file, preload=True)
        initial_context = loader.execute(ProcessingContext())

        # Run correction pipeline
        result = correction_pipeline.run(initial_context=initial_context)

        if result.success:
            # Export
            output_file = input_file.replace('.edf', '_corrected.edf')
            exporter = EDFExporter(path=output_file, overwrite=True)
            exporter.execute(result.context)
            print(f"  ✓ Saved to {output_file}")
        else:
            print(f"  ✗ Failed: {result.error}")


if __name__ == "__main__":
    # Run the main example
    # Note: Update file paths before running

    print("FACETpy Complete Pipeline Example")
    print("=" * 60)
    print("\nThis example demonstrates the full capabilities of FACETpy's")
    print("new modular architecture for fMRI artifact correction.")
    print("\nPlease update the file paths in the code before running.")
    print("\nAvailable examples:")
    print("  1. main() - Complete pipeline with all corrections")
    print("  2. example_parallel_execution() - Parallel processing")
    print("  3. example_minimal_pipeline() - Minimal correction")
    print("  4. example_custom_workflow() - Conditional processing")
    print("  5. example_batch_processing() - Batch file processing")
    print("\nTo run an example, uncomment the corresponding line below:")
    print("=" * 60)

    # Uncomment one of these to run:
    main()
    # example_parallel_execution()
    # example_minimal_pipeline()
    # example_custom_workflow()
    # example_batch_processing()
