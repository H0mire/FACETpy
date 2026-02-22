"""
Evaluation Module

This module contains processors for evaluating correction quality.

Author: FACETpy Team
Date: 2025-01-12
"""

from .metrics import (
    FFTAllenCalculator,
    FFTNiazyCalculator,
    LegacySNRCalculator,
    MedianArtifactCalculator,
    MetricsReport,
    ReferenceIntervalSelector,
    RMSCalculator,
    RMSResidualCalculator,
    SignalIntervalSelector,
    SNRCalculator,
)
from .visualization import RawPlotter

__all__ = [
    "ReferenceIntervalSelector",
    "SignalIntervalSelector",
    "SNRCalculator",
    "LegacySNRCalculator",
    "RMSCalculator",
    "RMSResidualCalculator",
    "MedianArtifactCalculator",
    "FFTAllenCalculator",
    "FFTNiazyCalculator",
    "MetricsReport",
    "RawPlotter",
]
