"""
Batch processing — run the same pipeline on multiple files.

Pipeline.map() handles per-file loading and collects one PipelineResult per
input. Failed files are skipped by default (on_error="continue").
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
    loader_factory=lambda p: EDFLoader(
        path=p,
        preload=True,
        artifact_to_trigger_offset=-0.005,
    ),
    on_error="continue",   # log failures, keep going
)

# ---------------------------------------------------------------------------
# Summarise results
# ---------------------------------------------------------------------------
print(f"\n{'File':<40} {'Status':<8} {'SNR':>8}")
print("-" * 60)
for path, result in zip(INPUT_FILES, results):
    status = "OK" if result.success else "FAIL"
    snr = f"{result.metrics.get('snr', float('nan')):.3f}" if result.success else "—"
    print(f"{path:<40} {status:<8} {snr:>8}")
