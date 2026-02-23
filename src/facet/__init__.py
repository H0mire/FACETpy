"""
FACETpy - fMRI Artifact Correction and Evaluation Toolbox for Python

A comprehensive, modular toolbox for correcting fMRI-induced artifacts in EEG data.

Author: FACETpy Team
Date: 2025-01-12
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "FACETpy Team"

# Core architecture
from .core import (
    BatchResult,
    ConditionalProcessor,
    # Parallel execution
    ParallelExecutor,
    Pipeline,
    PipelineError,
    PipelineResult,
    ProcessingContext,
    ProcessingMetadata,
    # Base classes
    Processor,
    # Exceptions
    ProcessorError,
    ProcessorValidationError,
    # Composite processors
    SequenceProcessor,
    SwitchProcessor,
    get_processor,
    list_processors,
    # Registry
    register_processor,
)

# Correction processors
from .correction import (
    AASCorrection,
    AveragedArtifactSubtraction,  # Alias
)

# I/O processors
from .io import (
    BDFExporter,
    BIDSExporter,
    BIDSLoader,
    BrainVisionExporter,
    EDFExporter,
    EEGLABExporter,
    Exporter,
    FIFExporter,
    GDFExporter,
    Loader,
    MFFExporter,
)
from .logging_config import configure_logging as _configure_logging

# Preprocessing processors
from .preprocessing import (
    BandPassFilter,
    # Transforms
    Crop,
    # Acquisition window
    CutAcquisitionWindow,
    DownSample,
    DropChannels,
    # Filtering
    Filter,
    HighPassFilter,
    LowPassFilter,
    MagicErasor,
    MissingTriggerDetector,
    NotchFilter,
    PasteAcquisitionWindow,
    PickChannels,
    PrintMetric,
    QRSTriggerDetector,
    RawTransform,
    # Resampling
    Resample,
    SliceAligner,
    SubsampleAligner,
    # Alignment
    TriggerAligner,
    # Trigger detection
    TriggerDetector,
    UpSample,
)

try:
    from .correction import AdaptiveNoiseCancellation, ANCCorrection  # noqa: F401

    _has_anc = True
except ImportError:
    _has_anc = False

try:
    from .correction import PCACorrection  # noqa: F401

    _has_pca = True
except ImportError:
    _has_pca = False

# Evaluation processors
from .evaluation import (
    FFTAllenCalculator,
    FFTNiazyCalculator,
    LegacySNRCalculator,
    MedianArtifactCalculator,
    MetricsReport,
    RawPlotter,
    RMSCalculator,
    RMSResidualCalculator,
    SignalIntervalSelector,
    SNRCalculator,
)

# Interactive helpers
from .helpers import ArtifactOffsetFinder, WaitForConfirmation

# Miscellaneous utilities
from .misc import (
    ArtifactParams,
    ChannelSchema,
    EEGGenerator,
    NoiseParams,
    OscillationParams,
    generate_synthetic_eeg,
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Core
    "Processor",
    "ProcessingContext",
    "ProcessingMetadata",
    "Pipeline",
    "PipelineResult",
    "BatchResult",
    "SequenceProcessor",
    "ConditionalProcessor",
    "SwitchProcessor",
    "register_processor",
    "get_processor",
    "list_processors",
    "ParallelExecutor",
    "ProcessorError",
    "ProcessorValidationError",
    "PipelineError",
    # I/O
    "Loader",
    "BIDSLoader",
    "Exporter",
    "EDFExporter",
    "BDFExporter",
    "BrainVisionExporter",
    "EEGLABExporter",
    "FIFExporter",
    "GDFExporter",
    "MFFExporter",
    "BIDSExporter",
    # Preprocessing
    "Filter",
    "HighPassFilter",
    "LowPassFilter",
    "BandPassFilter",
    "NotchFilter",
    "Resample",
    "UpSample",
    "DownSample",
    "TriggerDetector",
    "QRSTriggerDetector",
    "MissingTriggerDetector",
    "TriggerAligner",
    "SliceAligner",
    "SubsampleAligner",
    "CutAcquisitionWindow",
    "PasteAcquisitionWindow",
    "Crop",
    "MagicErasor",
    "RawTransform",
    "PickChannels",
    "DropChannels",
    "PrintMetric",
    # Correction
    "AASCorrection",
    "AveragedArtifactSubtraction",
    # Evaluation
    "SNRCalculator",
    "LegacySNRCalculator",
    "RMSCalculator",
    "RMSResidualCalculator",
    "MedianArtifactCalculator",
    "FFTAllenCalculator",
    "FFTNiazyCalculator",
    "MetricsReport",
    "SignalIntervalSelector",
    "RawPlotter",
    # Interactive helpers
    "ArtifactOffsetFinder",
    "WaitForConfirmation",
    # Miscellaneous / Synthetic Data Generation
    "EEGGenerator",
    "ChannelSchema",
    "OscillationParams",
    "NoiseParams",
    "ArtifactParams",
    "generate_synthetic_eeg",
    "create_standard_pipeline",
]

if _has_anc:
    __all__.extend(["ANCCorrection", "AdaptiveNoiseCancellation"])

if _has_pca:
    __all__.append("PCACorrection")

# Pipeline factories
from .pipelines import create_standard_pipeline as create_standard_pipeline

_configure_logging()


def load(path: str, **kwargs) -> "ProcessingContext":
    """Load an EEG file and return a :class:`ProcessingContext`.

    Automatically detects the file format from the extension (EDF, BDF, GDF,
    BrainVision, EEGLAB, FIF).  For BIDS datasets, use :class:`BIDSLoader`.

    Parameters
    ----------
    path : str
        Path to the EEG data file.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`~facet.io.Loader` (e.g. ``preload``,
        ``artifact_to_trigger_offset``, ``bad_channels``).

    Returns
    -------
    ProcessingContext
        Context containing the loaded recording.

    Examples
    --------
    >>> import facet
    >>> ctx = facet.load("./data/subject01.edf", preload=True)
    >>> ctx = ctx | facet.HighPassFilter(1.0) | facet.TriggerDetector(r"\\b1\\b")
    """
    return Loader(path=path, **kwargs).execute(None)


__all__.append("load")


def export(context: "ProcessingContext", path: str, **kwargs) -> "ProcessingContext":
    """Export a :class:`ProcessingContext` using extension-based auto routing.

    Uses :class:`~facet.io.Exporter` to dispatch export handling from ``path``
    extension (for example ``.edf``, ``.set``, ``.vhdr``).

    Parameters
    ----------
    context : ProcessingContext
        Input context that contains the Raw data to export.
    path : str
        Destination path; extension selects the exporter implementation.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`~facet.io.Exporter` (for example ``overwrite``).

    Returns
    -------
    ProcessingContext
        The unchanged input context after export.

    Examples
    --------
    >>> import facet
    >>> ctx = facet.load("./data/subject01.edf", preload=True)
    >>> ctx = facet.export(ctx, "./data/subject01_corrected.set", overwrite=True)
    """
    return Exporter(path=path, **kwargs).execute(context)


__all__.append("export")
