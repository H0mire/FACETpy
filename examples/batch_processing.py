"""
Batch processing — run the same pipeline on multiple files.

Pipeline.map() handles per-file loading and collects one PipelineResult per
input. Failed files are skipped by default (on_error="continue").

Call results.print_summary() for a formatted table — no manual formatting needed.
"""

from facet import (
    Pipeline,
    Loader,
    TriggerDetector,
    HighPassFilter,
    LowPassFilter,
    UpSample,
    DownSample,
    AASCorrection,
    SNRCalculator,
    MetricsReport,
)

INPUT_FILE = "./examples/datasets/NiazyFMRI.edf"

# In practice, list different subjects / sessions here.
# We reuse the same file three times just to illustrate the pattern.
INPUT_FILES = [INPUT_FILE, INPUT_FILE, INPUT_FILE]

# ---------------------------------------------------------------------------
# Define the correction pipeline without a loader — map() injects one per file
# ---------------------------------------------------------------------------
pipeline = Pipeline([
    TriggerDetector(regex=r"\b1\b"),
    HighPassFilter(freq=1.0),
    UpSample(factor=10),
    AASCorrection(window_size=30),
    DownSample(factor=10),
    LowPassFilter(freq=70),
    SNRCalculator(),
    MetricsReport(),
], name="Batch Correction")

results = pipeline.map(
    INPUT_FILES,
    loader_factory=lambda p: Loader(
        path=p,
        preload=True,
        artifact_to_trigger_offset=-0.005,
    ),
    on_error="continue",   # log failures, keep going
)

# ---------------------------------------------------------------------------
# One call prints a formatted table with status and metrics per file
# ---------------------------------------------------------------------------
results.print_summary()

# results.summary_df gives the same data as a pandas DataFrame for further analysis
df = results.summary_df
if df is not None:
    print("\nAs a DataFrame:")
    print(df.to_string(index=False))
