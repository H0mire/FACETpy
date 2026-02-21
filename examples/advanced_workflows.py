"""
Advanced workflows — conditional steps, parallel execution, factory shortcut.

Three patterns for less common but important use cases:

  A. ConditionalProcessor — apply PCA only when SNR is below a threshold.
  B. Parallel execution   — channel-wise AAS using joblib.
  C. create_standard_pipeline — one-liner factory with evaluate + plot flags.
"""

from pathlib import Path

from facet import (
    Pipeline,
    EDFLoader,
    EDFExporter,
    TriggerDetector,
    TriggerAligner,
    HighPassFilter,
    LowPassFilter,
    UpSample,
    DownSample,
    AASCorrection,
    PCACorrection,
    SNRCalculator,
    RMSCalculator,
    MetricsReport,
    RawPlotter,
    ConditionalProcessor,
    create_standard_pipeline,
)

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = "./examples/datasets/NiazyFMRI.edf"
OUTPUT_FILE = str(OUTPUT_DIR / "corrected_advanced.edf")
TRIGGER_REGEX = r"\b1\b"
UPSAMPLE = 10


# ---------------------------------------------------------------------------
# A. Conditional step — apply PCA only when SNR is poor
# ---------------------------------------------------------------------------
def example_conditional():
    def snr_too_low(ctx):
        snr = ctx.metadata.custom.get("metrics", {}).get("snr", float("inf"))
        return snr < 10

    pipeline = Pipeline([
        EDFLoader(path=INPUT_FILE, preload=True, artifact_to_trigger_offset=-0.005),
        TriggerDetector(regex=TRIGGER_REGEX),
        HighPassFilter(freq=1.0),
        UpSample(factor=UPSAMPLE),
        AASCorrection(window_size=30),
        DownSample(factor=UPSAMPLE),
        LowPassFilter(freq=70),
        SNRCalculator(),
        # PCA runs only when SNR < 10 dB
        ConditionalProcessor(
            condition=snr_too_low,
            processor=PCACorrection(n_components=0.95),
        ),
        MetricsReport(),
        EDFExporter(path=OUTPUT_FILE, overwrite=True),
    ], name="Conditional PCA")

    result = pipeline.run()
    print(f"Conditional pipeline — SNR: {result.metrics.get('snr', 'N/A'):.3f}")
    return result


# ---------------------------------------------------------------------------
# B. Parallel execution
# ---------------------------------------------------------------------------
def example_parallel():
    pipeline = Pipeline([
        EDFLoader(path=INPUT_FILE, preload=True, artifact_to_trigger_offset=-0.005),
        TriggerDetector(regex=TRIGGER_REGEX),
        HighPassFilter(freq=1.0),
        UpSample(factor=UPSAMPLE),
        TriggerAligner(ref_trigger_index=0, upsample_for_alignment=False),
        AASCorrection(window_size=30, correlation_threshold=0.975),
        DownSample(factor=UPSAMPLE),
        LowPassFilter(freq=70),
        RMSCalculator(),
        MetricsReport(),
        EDFExporter(path=OUTPUT_FILE, overwrite=True),
    ], name="Parallel AAS")

    # parallel=True enables channel-wise joblib parallelism for compatible steps
    result = pipeline.run(parallel=True, n_jobs=-1)
    print(f"Parallel pipeline — {result.execution_time:.2f}s, "
          f"RMS ratio: {result.metrics.get('rms_ratio', 'N/A'):.3f}")
    return result


# ---------------------------------------------------------------------------
# C. Factory shortcut — evaluate=True adds all metric processors automatically
# ---------------------------------------------------------------------------
def example_factory():
    pipeline = create_standard_pipeline(
        INPUT_FILE,
        OUTPUT_FILE,
        use_anc=False,
        evaluate=True,
        plot=True,
        plot_kwargs={
            "mode": "matplotlib",
            "channel": "Fp1",
            "start": 25.0,
            "duration": 20.0,
            "overlay_original": True,
            "save_path": str(OUTPUT_DIR / "factory_plot.png"),
            "show": False,
            "auto_close": True,
            "title": "Fp1 – Before vs After",
        },
    )

    result = pipeline.run()
    print(f"Factory pipeline — SNR: {result.metrics.get('snr', 'N/A'):.3f}")
    return result


if __name__ == "__main__":
    print("=== A. Conditional PCA ===")
    example_conditional()

    print("\n=== B. Parallel execution ===")
    example_parallel()

    print("\n=== C. Factory shortcut ===")
    example_factory()
