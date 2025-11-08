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
    ProcessorValidationError
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
    'ProcessorValidationError',

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
]

# Add optional exports
if _has_anc:
    __all__.extend(['ANCCorrection', 'AdaptiveNoiseCancellation'])

if _has_pca:
    __all__.append('PCACorrection')


# Convenience function for quick pipeline creation
def create_standard_pipeline(
    input_path: str,
    output_path: str,
    trigger_regex: str = r"\b1\b",
    upsample_factor: int = 10,
    use_anc: bool = True,
    use_pca: bool = False
) -> Pipeline:
    """
    Create a standard fMRI artifact correction pipeline.

    Args:
        input_path: Path to input EEG file
        output_path: Path to output corrected file
        trigger_regex: Regex pattern for trigger detection
        upsample_factor: Upsampling factor for alignment
        use_anc: Whether to apply ANC correction
        use_pca: Whether to apply PCA correction

    Returns:
        Configured Pipeline instance

    Example:
        pipeline = facet.create_standard_pipeline(
            "data.edf",
            "corrected.edf",
            trigger_regex=r"\\b1\\b"
        )
        result = pipeline.run()
    """
    processors = [
        EDFLoader(path=input_path, preload=True),
        TriggerDetector(regex=trigger_regex),
        CutAcquisitionWindow(),
        UpSample(factor=upsample_factor),
        SliceAligner(ref_trigger_index=0),
        SubsampleAligner(ref_trigger_index=0),
        AASCorrection(window_size=30, correlation_threshold=0.975)
    ]

    if use_anc and _has_anc:
        processors.append(ANCCorrection(filter_order=5, hp_freq=1.0))

    if use_pca and _has_pca:
        processors.append(PCACorrection(n_components=0.95, hp_freq=1.0))

    processors.extend([
        DownSample(factor=upsample_factor),
        PasteAcquisitionWindow(),
        HighPassFilter(freq=0.5),
        EDFExporter(path=output_path, overwrite=True)
    ])

    return Pipeline(processors, name="Standard fMRI Correction Pipeline")


# Print info on import
def _print_import_info():
    """Print information about available features."""
    import sys
    if hasattr(sys, 'ps1'):  # Interactive mode
        return

    # Only print in non-interactive mode if explicitly requested
    import os
    if os.environ.get('FACETPY_VERBOSE_IMPORT', '0') == '1':
        print(f"FACETpy v{__version__} loaded")
        print(f"  Available processors: {len(list_processors())}")
        if not _has_anc:
            print("  Note: ANC not available (C extension not built)")
        if not _has_pca:
            print("  Note: PCA not available (missing dependencies)")


# Run import info
_print_import_info()
