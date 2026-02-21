"""
Evaluation — metrics access and pipeline comparison.

Covers:
  - All built-in metric processors (SNR, RMS, FFT, median artifact)
  - result.metrics  / result.metrics_df for direct access
  - MetricsReport.compare() for side-by-side comparison of two pipelines
"""

from facet import (
    Pipeline,
    EDFLoader,
    TriggerDetector,
    HighPassFilter,
    LowPassFilter,
    UpSample,
    DownSample,
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
from pathlib import Path

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = "./examples/datasets/NiazyFMRI.edf"
TRIGGER_REGEX = r"\b1\b"
UPSAMPLE = 10


# ---------------------------------------------------------------------------
# Shared preprocessing steps (reused in both pipelines below)
# ---------------------------------------------------------------------------
def base_steps():
    return [
        EDFLoader(path=INPUT_FILE, preload=True, artifact_to_trigger_offset=-0.005),
        TriggerDetector(regex=TRIGGER_REGEX),
        HighPassFilter(freq=1.0),
        UpSample(factor=UPSAMPLE),
        AASCorrection(window_size=30),
        DownSample(factor=UPSAMPLE),
        LowPassFilter(freq=70),
    ]


EVAL_STEPS = [
    SNRCalculator(),
    LegacySNRCalculator(),
    RMSCalculator(),
    RMSResidualCalculator(),
    MedianArtifactCalculator(),
    FFTAllenCalculator(),
    FFTNiazyCalculator(),
    MetricsReport(),
]


# ---------------------------------------------------------------------------
# 1. Single pipeline — access metrics on the result object
# ---------------------------------------------------------------------------
def example_metrics_access():
    pipeline = Pipeline(base_steps() + EVAL_STEPS, name="AAS")
    result = pipeline.run()

    if not result.success:
        print(f"Failed: {result.error}")
        return

    # result.metrics is a plain dict
    print("\n--- result.metrics ---")
    for key, val in result.metrics.items():
        if isinstance(val, dict):
            print(f"  {key}:")
            for k, v in val.items():
                print(f"    {k}: {v:.4g}")
        else:
            print(f"  {key}: {val:.4g}")

    # result.metrics_df is a flat pandas Series (scalar values only)
    df = result.metrics_df
    if df is not None:
        print("\n--- result.metrics_df ---")
        print(df.to_string())

    # Visualise corrected signal
    result.plot(
        mode="matplotlib",
        channel="Fp1",
        start=25.0,
        duration=20.0,
        overlay_original=True,
        save_path=str(OUTPUT_DIR / "evaluation_plot.png"),
        show=False,
        auto_close=True,
        title="Fp1 – Before vs After",
    )


# ---------------------------------------------------------------------------
# 2. Compare two pipelines — MetricsReport.compare()
# ---------------------------------------------------------------------------
def example_compare():
    result_aas = Pipeline(
        base_steps() + EVAL_STEPS,
        name="AAS only",
    ).run()

    result_pca = Pipeline(
        base_steps() + [PCACorrection(n_components=0.95)] + EVAL_STEPS,
        name="AAS + PCA",
    ).run()

    # compare() accepts PipelineResult objects directly
    MetricsReport.compare(
        [result_aas, result_pca],
        labels=["AAS only", "AAS + PCA"],
        metrics=["snr", "rms_ratio", "rms_residual", "median_artifact"],
        title="AAS vs AAS + PCA",
        save_path=str(OUTPUT_DIR / "comparison.png"),
        show=False,
    )

    print(f"\nAAS only  — SNR: {result_aas.metrics.get('snr', 'N/A'):.3f}")
    print(f"AAS + PCA — SNR: {result_pca.metrics.get('snr', 'N/A'):.3f}")


if __name__ == "__main__":
    example_metrics_access()
    example_compare()
