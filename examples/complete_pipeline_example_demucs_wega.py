"""Demucs Phase 3 Weg-A correction, using the selected epoch-53 checkpoint.

Adjust the paths below, then run from the repository root:
    uv run python -m examples.complete_pipeline_example_demucs_wega
"""

from masterthesis_guide.reproduce import adapter

from facet import (
    Crop,
    DeepLearningCorrection,
    DropChannels,
    HighPassFilter,
    Loader,
    LowPassFilter,
    Pipeline,
    RawPlotter,
    RMSResidualCalculator,
    SNRCalculator,
    TriggerDetector,
)
from facet.config import set_config

from masterthesis_guide.reproduce import check_downloaded  # isort: skip

# Constants
INPUT_FILE = "examples/datasets/NiazyFMRI.edf"
EXPERIMENT = "wega_demucs_lr0_0001_ic96_sisdr3_s42"
DEVICE = "cpu"
ARTIFACT_TO_TRIGGER_OFFSET = -0.005  # Seconds: 5 ms before the trigger.
PLOT_CHANNEL = "Fp1"
PLOT_START = 28.0
PLOT_DURATION = 4.0

check_downloaded(EXPERIMENT)

set_config(log_level="INFO", console_mode="modern")

# Pipeline steps
steps = [
    Loader(path=INPUT_FILE, preload=True, artifact_to_trigger_offset=ARTIFACT_TO_TRIGGER_OFFSET),
    DropChannels(channels=["EKG", "EMG", "EOG", "ECG"], on_missing="ignore"),
    Crop(tmin=0, tmax=162),
    HighPassFilter(freq=1.0),
    TriggerDetector(regex=r"\b1\b"),
    DeepLearningCorrection(model=adapter(EXPERIMENT, device=DEVICE)),
    LowPassFilter(freq=70.0),
    SNRCalculator(),
    RMSResidualCalculator(),
    RawPlotter(
        mode="matplotlib",
        channel=PLOT_CHANNEL,
        start=PLOT_START,
        duration=PLOT_DURATION,
        overlay_original=False,
        show=True,
        title="Demucs Phase 3 Weg-A: corrected signal",
    ),
]

# Run — the existing thesis adapter requires channel_sequential=False.
result = Pipeline(steps, name="Demucs Phase 3 Weg-A").run(channel_sequential=False)

# Results
assert result.success, result.error
result.print_summary()
result.print_metrics()
