"""
Quickstart — minimal fMRI artifact correction pipeline.

The fewest steps needed to load, correct, and export an EDF recording.
Run this first to verify your installation.
"""

from facet import (
    Pipeline,
    EDFLoader,
    EDFExporter,
    TriggerDetector,
    UpSample,
    DownSample,
    AASCorrection,
)

INPUT_FILE = "./examples/datasets/NiazyFMRI.edf"
OUTPUT_FILE = "./output/corrected_quickstart.edf"

pipeline = Pipeline([
    EDFLoader(path=INPUT_FILE, preload=True),
    TriggerDetector(regex=r"\b1\b"),
    UpSample(factor=10),
    AASCorrection(window_size=30),
    DownSample(factor=10),
    EDFExporter(path=OUTPUT_FILE, overwrite=True),
], name="Quickstart")

result = pipeline.run()

if result.success:
    print(f"Done in {result.execution_time:.2f}s → {OUTPUT_FILE}")
else:
    print(f"Failed: {result.error}")
