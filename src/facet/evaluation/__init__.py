"""
Evaluation Module

This module contains processors for evaluating correction quality.

Author: FACETpy Team
Date: 2025-01-12
"""

from .metrics import (
    SNRCalculator,
    LegacySNRCalculator,
    RMSCalculator,
    RMSResidualCalculator,
    MedianArtifactCalculator,
    FFTAllenCalculator,
    FFTNiazyCalculator,
    MetricsReport,
)
from .visualization import RawPlotter

__all__ = [
    'SNRCalculator',
    'LegacySNRCalculator',
    'RMSCalculator',
    'RMSResidualCalculator',
    'MedianArtifactCalculator',
    'FFTAllenCalculator',
    'FFTNiazyCalculator',
    'MetricsReport',
    'RawPlotter',
]
