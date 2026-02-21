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
)

# Correction processors
from .correction import (
    AASCorrection,
    AveragedArtifactSubtraction  # Alias
)

# Import optional correction processors
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
    RMSCalculator,
    MedianArtifactCalculator,
    MetricsReport
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

# Build __all__ dynamically
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

    # Correction
    'AASCorrection',
    'AveragedArtifactSubtraction',

    # Evaluation
    'SNRCalculator',
    'RMSCalculator',
    'MedianArtifactCalculator',
    'MetricsReport',

    # Miscellaneous / Synthetic Data Generation
    'EEGGenerator',
    'ChannelSchema',
    'OscillationParams',
    'NoiseParams',
    'ArtifactParams',
    'generate_synthetic_eeg',
]

# Add optional exports
if _has_anc:
    __all__.extend(['ANCCorrection', 'AdaptiveNoiseCancellation'])

if _has_pca:
    __all__.append('PCACorrection')

# Pipeline factories
from .pipelines import create_standard_pipeline  # noqa: E402

__all__.append('create_standard_pipeline')
