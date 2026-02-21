"""
FACETpy - fMRI Artifact Correction and Evaluation Toolbox for Python

A comprehensive, modular toolbox for correcting fMRI-induced artifacts in EEG data.

Author: FACETpy Team
Date: 2025-01-12
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "FACETpy Team"

from .logging_config import configure_logging as _configure_logging

_configure_logging()

# Core architecture
from .core import (
    # Base classes
    Processor,
    ProcessingContext,
    ProcessingMetadata,
    Pipeline,
    PipelineResult,
    BatchResult,

    # Composite processors
    SequenceProcessor,
    ConditionalProcessor,
    SwitchProcessor,

    # Registry
    register_processor,
    get_processor,
    list_processors,

    # Parallel execution
    ParallelExecutor,

    # Exceptions
    ProcessorError,
    ProcessorValidationError,
    PipelineError
)

# I/O processors
from .io import (
    EDFLoader,
    BIDSLoader,
    GDFLoader,
    EDFExporter,
    BIDSExporter
)

# Preprocessing processors
from .preprocessing import (
    # Filtering
    Filter,
    HighPassFilter,
    LowPassFilter,
    BandPassFilter,
    NotchFilter,

    # Resampling
    Resample,
    UpSample,
    DownSample,

    # Trigger detection
    TriggerDetector,
    QRSTriggerDetector,
    MissingTriggerDetector,

    # Alignment
    TriggerAligner,
    SliceAligner,
    SubsampleAligner,

    # Acquisition window
    CutAcquisitionWindow,
    PasteAcquisitionWindow,

    # Transforms
    Crop,
    RawTransform,
    PickChannels,
    DropChannels,
    PrintMetric,
)

# Correction processors
from .correction import (
    AASCorrection,
    AveragedArtifactSubtraction  # Alias
)

try:
    from .correction import ANCCorrection, AdaptiveNoiseCancellation
    _has_anc = True
except ImportError:
    _has_anc = False

try:
    from .correction import PCACorrection
    _has_pca = True
except ImportError:
    _has_pca = False

# Evaluation processors
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

# Miscellaneous utilities
from .misc import (
    EEGGenerator,
    ChannelSchema,
    OscillationParams,
    NoiseParams,
    ArtifactParams,
    generate_synthetic_eeg,
)

__all__ = [
    # Version
    '__version__',
    '__author__',

    # Core
    'Processor',
    'ProcessingContext',
    'ProcessingMetadata',
    'Pipeline',
    'PipelineResult',
    'BatchResult',
    'SequenceProcessor',
    'ConditionalProcessor',
    'SwitchProcessor',
    'register_processor',
    'get_processor',
    'list_processors',
    'ParallelExecutor',
    'ProcessorError',
    'ProcessorValidationError',
    'PipelineError',

    # I/O
    'EDFLoader',
    'BIDSLoader',
    'GDFLoader',
    'EDFExporter',
    'BIDSExporter',

    # Preprocessing
    'Filter',
    'HighPassFilter',
    'LowPassFilter',
    'BandPassFilter',
    'NotchFilter',
    'Resample',
    'UpSample',
    'DownSample',
    'TriggerDetector',
    'QRSTriggerDetector',
    'MissingTriggerDetector',
    'TriggerAligner',
    'SliceAligner',
    'SubsampleAligner',
    'CutAcquisitionWindow',
    'PasteAcquisitionWindow',
    'Crop',
    'RawTransform',
    'PickChannels',
    'DropChannels',
    'PrintMetric',

    # Correction
    'AASCorrection',
    'AveragedArtifactSubtraction',

    # Evaluation
    'SNRCalculator',
    'LegacySNRCalculator',
    'RMSCalculator',
    'RMSResidualCalculator',
    'MedianArtifactCalculator',
    'FFTAllenCalculator',
    'FFTNiazyCalculator',
    'MetricsReport',
    'RawPlotter',

    # Miscellaneous / Synthetic Data Generation
    'EEGGenerator',
    'ChannelSchema',
    'OscillationParams',
    'NoiseParams',
    'ArtifactParams',
    'generate_synthetic_eeg',
]

if _has_anc:
    __all__.extend(['ANCCorrection', 'AdaptiveNoiseCancellation'])

if _has_pca:
    __all__.append('PCACorrection')

# Pipeline factories
from .pipelines import create_standard_pipeline  # noqa: E402

__all__.append('create_standard_pipeline')


def load_edf(path: str, **kwargs) -> 'ProcessingContext':
    """
    Load an EDF file and return a :class:`ProcessingContext`.

    A convenience shortcut so you can start working with a recording without
    constructing an :class:`~facet.io.EDFLoader` processor explicitly.  Any
    keyword argument accepted by :class:`~facet.io.EDFLoader` (e.g.
    ``preload``, ``artifact_to_trigger_offset``) can be passed here.

    Args:
        path: Path to the EDF file.
        **kwargs: Additional keyword arguments forwarded to
            :class:`~facet.io.EDFLoader`.

    Returns:
        :class:`ProcessingContext` containing the loaded recording.

    Example::

        import facet

        ctx = facet.load_edf("./data/subject01.edf", preload=True)
        ctx = ctx | facet.HighPassFilter(1.0) | facet.TriggerDetector(r"\\b1\\b")
    """
    return EDFLoader(path=path, **kwargs).execute(None)


__all__.append('load_edf')
