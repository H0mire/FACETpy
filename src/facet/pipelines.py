"""Pre-built pipeline factories for common FACETpy workflows."""

from __future__ import annotations

from .core import Pipeline
from .io import Loader, EDFExporter
from .preprocessing import (
    TriggerDetector,
    CutAcquisitionWindow,
    PasteAcquisitionWindow,
    HighPassFilter,
    LowPassFilter,
    UpSample,
    DownSample,
    SliceAligner,
    SubsampleAligner,
)
from .correction import AASCorrection
from .evaluation import (
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

try:
    from .correction import ANCCorrection
    _has_anc = True
except ImportError:
    _has_anc = False

try:
    from .correction import PCACorrection
    _has_pca = True
except ImportError:
    _has_pca = False


def create_standard_pipeline(
    input_path: str,
    output_path: str,
    start_sample: int = 0,
    end_sample: int = None,
    trigger_regex: str = r"\b1\b",
    artifact_to_trigger_offset: float = -0.005,
    upsample_factor: int = 10,
    use_anc: bool = True,
    use_pca: bool = True,
    evaluate: bool = False,
    plot: bool = False,
    plot_kwargs: dict | None = None,
) -> Pipeline:
    """
    Create a standard fMRI artifact correction pipeline.

    Args:
        input_path: Path to input EEG file
        output_path: Path to output corrected file
        trigger_regex: Regex pattern for trigger detection
        artifact_to_trigger_offset: Time offset between artifact and trigger (seconds)
        upsample_factor: Upsampling factor for alignment
        use_anc: Whether to apply ANC correction
        use_pca: Whether to apply PCA correction
        evaluate: Append all standard evaluation metrics (SNR, RMS, Median, FFT)
                  and a MetricsReport step automatically.
        plot: Append a RawPlotter step after evaluation. Implies *evaluate=True*.
        plot_kwargs: Extra keyword arguments forwarded to ``RawPlotter``.

    Returns:
        Configured Pipeline instance

    Example::

        # Minimal — just correction + export
        pipeline = create_standard_pipeline("data.edf", "corrected.edf")
        result = pipeline.run()
        
        # With automatic evaluation
        pipeline = create_standard_pipeline(
            "data.edf",
            "corrected.edf",
            evaluate=True,
        )
        result = pipeline.run()
        print(result.metrics)

        # With evaluation and interactive plot
        pipeline = create_standard_pipeline(
            "data.edf",
            "corrected.edf",
            evaluate=True,
            plot=True,
            plot_kwargs={"channel": "Fp1", "start": 5.0, "duration": 10.0},
        )
        result = pipeline.run()
    """
    processors = [
        Loader(path=input_path, preload=True, artifact_to_trigger_offset=artifact_to_trigger_offset),
        TriggerDetector(regex=trigger_regex),
        CutAcquisitionWindow(),
        HighPassFilter(freq=1.0),
        UpSample(factor=upsample_factor),
        SliceAligner(ref_trigger_index=0),
        SubsampleAligner(ref_trigger_index=0),
        AASCorrection(window_size=30, correlation_threshold=0.975),
    ]
    if use_pca and _has_pca:
        processors.append(PCACorrection(n_components=0.95, hp_freq=1.0))

    processors.extend([
        DownSample(factor=upsample_factor),
        PasteAcquisitionWindow(),
        LowPassFilter(freq=70.0),
    ])

    if use_anc and _has_anc:
        processors.append(ANCCorrection())

    processors.append(
        EDFExporter(path=output_path, overwrite=True),
    )

    if evaluate or plot:
        processors.extend([
            SNRCalculator(),
            LegacySNRCalculator(),
            RMSCalculator(),
            RMSResidualCalculator(),
            MedianArtifactCalculator(),
            FFTAllenCalculator(),
            FFTNiazyCalculator(),
            MetricsReport(),
        ])

    if plot:
        processors.append(RawPlotter(**(plot_kwargs or {})))

    return Pipeline(processors, name="Standard fMRI Correction Pipeline")
