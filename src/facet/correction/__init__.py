"""
Correction Module

This module contains processors for correcting EEG artifacts.

Author: FACETpy Team
Date: 2025-01-12
"""

from .aas import AASCorrection, AveragedArtifactSubtraction

__all__ = [
    # AAS
    "AASCorrection",
    "AveragedArtifactSubtraction",
]

# Import ANC if available
try:
    from .anc import AdaptiveNoiseCancellation, ANCCorrection  # noqa: F401

    __all__.extend(["AdaptiveNoiseCancellation", "ANCCorrection"])
except ImportError:
    pass

# Import PCA if available
try:
    from .pca import PCACorrection  # noqa: F401

    __all__.append("PCACorrection")
except ImportError:
    pass
