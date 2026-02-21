"""
Complete FACETpy Pipeline Example

Demonstrates the full range of FACETpy capabilities using the modular
processor-pipeline architecture.

Author: FACETpy Team
"""

import os
from pathlib import Path
import traceback

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MPLCONFIG_PATH = OUTPUT_DIR / "mpl_config"
MPLCONFIG_PATH.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_PATH.resolve()))
os.environ["FACET_CONSOLE_MODE"] = "modern"  # "classic" or "modern"


# Everything is available directly from the top-level `facet` package
import facet
from facet import (
    Pipeline,
    LambdaProcessor,
    ConditionalProcessor,
    # I/O
    EDFLoader,
    EDFExporter,
    # Preprocessing
    HighPassFilter,
    LowPassFilter,
    UpSample,
    DownSample,
    TriggerDetector,
    TriggerAligner,
    Crop,
    RawTransform,
    # Correction
    AASCorrection,
    ANCCorrection,
    PCACorrection,
    # Evaluation
    SNRCalculator,
    LegacySNRCalculator,
    RMSCalculator,
    RMSResidualCalculator,
    MedianArtifactCalculator,
    FFTAllenCalculator,
    FFTNiazyCalculator,
    MetricsReport,
    RawPlotter,
    # Factory
    create_standard_pipeline,
)

INPUT_FILE = "./examples/datasets/NiazyFMRI.edf"
OUTPUT_FILE = str(OUTPUT_DIR / "corrected.edf")

TRIGGER_REGEX = r"\b1\b"
UPSAMPLE_FACTOR = 10
BAD_CHANNELS = ["EKG", "EMG", "EOG", "ECG"]
ARTIFACT_TO_TRIGGER_OFFSET = -0.005
CROP_TMIN = 0
CROP_TMAX = 162


# ---------------------------------------------------------------------------
# 1. Full custom pipeline
# ---------------------------------------------------------------------------
def main():
    """Complete fMRI artifact correction pipeline with all steps."""

    pipeline = Pipeline([
        # Load data
        EDFLoader(
            path=INPUT_FILE,
            preload=True,
            bad_channels=BAD_CHANNELS,
            artifact_to_trigger_offset=ARTIFACT_TO_TRIGGER_OFFSET,
        ),

        # Crop to region of interest (new Crop processor)
        Crop(tmin=CROP_TMIN, tmax=CROP_TMAX),

        # Detect fMRI slice triggers
        TriggerDetector(regex=TRIGGER_REGEX),

        HighPassFilter(freq=1.0),

        # Upsample for sub-sample trigger precision
        UpSample(factor=UPSAMPLE_FACTOR),

        # Align triggers using cross-correlation
        TriggerAligner(
            ref_trigger_index=0,
            upsample_for_alignment=False,
        ),

        # Averaged Artifact Subtraction (AAS) — primary correction
        AASCorrection(
            window_size=30,
            correlation_threshold=0.975,
            realign_after_averaging=True,
            plot_artifacts=False,
        ),

        # PCA — remove systematic residual components
        PCACorrection(
            n_components=0.95,
            hp_freq=1.0,
        ),

        # Downsample back to original rate
        DownSample(factor=UPSAMPLE_FACTOR),

        LowPassFilter(freq=70),

        # Adaptive Noise Cancellation — remove remaining gradient noise
        ANCCorrection(use_c_extension=True),

        # --- Evaluation ---
        SNRCalculator(),
        LegacySNRCalculator(),
        RMSCalculator(),
        RMSResidualCalculator(),
        MedianArtifactCalculator(),
        FFTAllenCalculator(),
        FFTNiazyCalculator(),
        MetricsReport(),

        # Visualise corrected signal
        RawPlotter(
            mode="mne",
            start=5.0,
            duration=5.0,
            save_path=str(OUTPUT_DIR / "eeg_visualization.png"),
            show=True,
            auto_close=False,
            mne_kwargs={"scalings": "auto"},
        ),

        # Export corrected recording
        EDFExporter(path=OUTPUT_FILE, overwrite=True),
    ], name="fMRI Artifact Correction Pipeline")

    result = pipeline.run()

    if result.success:
        print(f"\n✓ Pipeline completed in {result.execution_time:.2f}s")

        # Metrics are now a first-class attribute on the result
        print("\nFinal metrics:")
        for key, value in result.metrics.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_val in value.items():
                    print(f"    {sub_key}: {sub_val:.3g}")
            else:
                print(f"  {key}: {value:.4g}")

        # pandas Series (if pandas is installed)
        df = result.metrics_df
        if df is not None:
            print("\nmetrics_df:\n", df.to_string())

    else:
        print(f"\n✗ Pipeline failed: {result.error}")
        traceback.print_exception(type(result.error), result.error, result.error.__traceback__)
        if result.failed_processor:
            print(f"Failed at: {result.failed_processor}")


# ---------------------------------------------------------------------------
# 2. One-liner: factory with built-in evaluation
# ---------------------------------------------------------------------------
def example_standard_pipeline():
    """
    Simplest possible usage: factory + evaluate=True means no manual metric
    processors and no need to collect results in a side-channel dict.
    """
    pipeline = create_standard_pipeline(
        INPUT_FILE,
        OUTPUT_FILE,
        use_anc=True,
        evaluate=True,
        plot=True,
        plot_kwargs={
            "mode": "matplotlib",
            "channel": "Fp1",
            "start": 25.0,
            "duration": 20.0,
            "overlay_original": True,
            "save_path": str(OUTPUT_DIR / "pipeline_before_after.png"),
            "show": True,
            "auto_close": False,
            "title": "Fp1 – Before vs After",
        },
    )

    result = pipeline.run()
    print(result.metrics)          # {'snr': …, 'rms_ratio': …, …}
    return result


# ---------------------------------------------------------------------------
# 3. Compare two pipelines with MetricsReport.compare()
# ---------------------------------------------------------------------------
def example_compare_pipelines():
    """Compare AAS-only vs AAS+PCA using the new MetricsReport.compare() API."""

    base = [
        EDFLoader(path=INPUT_FILE, preload=True, artifact_to_trigger_offset=ARTIFACT_TO_TRIGGER_OFFSET),
        TriggerDetector(regex=TRIGGER_REGEX),
        HighPassFilter(freq=1.0),
        UpSample(factor=UPSAMPLE_FACTOR),
        TriggerAligner(ref_trigger_index=0, upsample_for_alignment=False),
        AASCorrection(window_size=30),
        DownSample(factor=UPSAMPLE_FACTOR),
        LowPassFilter(freq=70),
        SNRCalculator(),
        RMSCalculator(),
        RMSResidualCalculator(),
        MedianArtifactCalculator(),
        MetricsReport(),
    ]

    result_aas = Pipeline(base, name="AAS only").run()
    result_pca = Pipeline(
        base[:-1] + [PCACorrection(n_components=0.95)] + base[-1:],
        name="AAS + PCA",
    ).run()

    # New compare() API — accepts PipelineResult objects directly
    MetricsReport.compare(
        [result_aas, result_pca],
        labels=["AAS only", "AAS + PCA"],
        metrics=["snr", "rms_ratio", "rms_residual", "median_artifact"],
    )


# ---------------------------------------------------------------------------
# 4. Parallel execution
# ---------------------------------------------------------------------------
def example_parallel_execution():
    """Run AAS correction with channel-wise parallelisation."""

    pipeline = Pipeline([
        EDFLoader(path=INPUT_FILE, preload=True, artifact_to_trigger_offset=ARTIFACT_TO_TRIGGER_OFFSET),
        TriggerDetector(regex=TRIGGER_REGEX),
        HighPassFilter(freq=0.5),
        UpSample(factor=UPSAMPLE_FACTOR),
        TriggerAligner(ref_trigger_index=0),
        AASCorrection(window_size=30, correlation_threshold=0.975),
        DownSample(factor=UPSAMPLE_FACTOR),
        LowPassFilter(freq=70),
        SNRCalculator(),
        RMSCalculator(),
        MetricsReport(),
        RawPlotter(
            mode="matplotlib",
            channel="Fp1",
            start=25.0,
            duration=120.0,
            overlay_original=True,
            save_path=str(OUTPUT_DIR / "pipeline_before_after.png"),
            show=True,
            auto_close=True,
            title="Fp1 – Before vs After (parallel)",
        ),
        EDFExporter(path=OUTPUT_FILE, overwrite=True),
    ], name="Parallel Pipeline")

    result = pipeline.run(parallel=True, n_jobs=-1)
    print(f"Completed in {result.execution_time:.2f}s")
    print(result.metrics)


# ---------------------------------------------------------------------------
# 5. Minimal pipeline
# ---------------------------------------------------------------------------
def example_minimal_pipeline():
    """Fewest possible steps for a quick correction."""

    pipeline = Pipeline([
        EDFLoader(path=INPUT_FILE, preload=True),
        TriggerDetector(regex=TRIGGER_REGEX),
        UpSample(factor=UPSAMPLE_FACTOR),
        AASCorrection(window_size=30),
        DownSample(factor=UPSAMPLE_FACTOR),
        EDFExporter(path=OUTPUT_FILE),
    ], name="Minimal Pipeline")

    return pipeline.run()


# ---------------------------------------------------------------------------
# 6. Crop + RawTransform inline transforms
# ---------------------------------------------------------------------------
def example_inline_transforms():
    """Demonstrate Crop and RawTransform as clean alternatives to LambdaProcessor."""

    pipeline = Pipeline([
        EDFLoader(path=INPUT_FILE, preload=True),

        # Dedicated Crop processor — no lambda boilerplate
        Crop(tmin=10, tmax=120),

        # RawTransform for arbitrary one-liners
        RawTransform(
            "drop_non_eeg",
            lambda raw: raw.copy().pick_types(eeg=True, stim=True),
        ),

        TriggerDetector(regex=TRIGGER_REGEX),
        UpSample(factor=UPSAMPLE_FACTOR),
        AASCorrection(window_size=30),
        DownSample(factor=UPSAMPLE_FACTOR),
        EDFExporter(path=OUTPUT_FILE),
    ])

    return pipeline.run()


# ---------------------------------------------------------------------------
# 7. Conditional processing
# ---------------------------------------------------------------------------
def example_custom_workflow():
    """Apply PCA only when SNR is below a threshold."""

    def should_apply_pca(context):
        snr = context.metadata.custom.get('metrics', {}).get('snr', float('inf'))
        return snr < 10

    pipeline = Pipeline([
        EDFLoader(path=INPUT_FILE, preload=True),
        TriggerDetector(regex=TRIGGER_REGEX),
        UpSample(factor=UPSAMPLE_FACTOR),
        AASCorrection(window_size=30),
        SNRCalculator(),
        ConditionalProcessor(
            condition=should_apply_pca,
            processor=PCACorrection(n_components=0.95),
        ),
        DownSample(factor=UPSAMPLE_FACTOR),
        MetricsReport(),
        EDFExporter(path=OUTPUT_FILE),
    ])

    return pipeline.run()


# ---------------------------------------------------------------------------
# 8. Batch processing with Pipeline.map()
# ---------------------------------------------------------------------------
def example_batch_processing():
    """Process multiple files with Pipeline.map() — no manual context wiring."""

    correction_pipeline = Pipeline([
        TriggerDetector(regex=TRIGGER_REGEX),
        HighPassFilter(freq=1.0),
        UpSample(factor=UPSAMPLE_FACTOR),
        TriggerAligner(ref_trigger_index=0),
        AASCorrection(window_size=30),
        DownSample(factor=UPSAMPLE_FACTOR),
        LowPassFilter(freq=70),
        SNRCalculator(),
        MetricsReport(),
    ])

    input_files = [
        INPUT_FILE,
        INPUT_FILE,  # Replace with different subjects in real use
        INPUT_FILE,
    ]

    results = correction_pipeline.map(
        input_files,
        loader_factory=lambda p: EDFLoader(path=p, preload=True,
                                           artifact_to_trigger_offset=ARTIFACT_TO_TRIGGER_OFFSET),
    )

    for path, result in zip(input_files, results):
        status = "✓" if result.success else "✗"
        snr = result.metrics.get('snr', 'N/A')
        print(f"{status} {path}  SNR={snr}")


if __name__ == "__main__":
    print("FACETpy Complete Pipeline Example")
    print("=" * 60)
    print("\nAvailable examples:")
    print("  1. main()                     – Full custom pipeline")
    print("  2. example_standard_pipeline()– Factory with evaluate=True")
    print("  3. example_compare_pipelines()– MetricsReport.compare()")
    print("  4. example_parallel_execution()– Parallel AAS")
    print("  5. example_minimal_pipeline() – Fewest steps")
    print("  6. example_inline_transforms()– Crop + RawTransform")
    print("  7. example_custom_workflow()  – ConditionalProcessor")
    print("  8. example_batch_processing() – Pipeline.map()")
    print("=" * 60)

    # Uncomment one example to run:
    main()
    # example_standard_pipeline()
    # example_compare_pipelines()
    # example_parallel_execution()
    # example_minimal_pipeline()
    # example_inline_transforms()
    # example_custom_workflow()
    # example_batch_processing()
