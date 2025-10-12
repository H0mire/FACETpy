"""
Evaluation Module

This module contains processors for evaluating correction quality.

Author: FACETpy Team
Date: 2025-01-12
"""

from .metrics import (
    SNRCalculator,
    RMSCalculator,
    MedianArtifactCalculator,
    MetricsReport
)

__all__ = [
    'SNRCalculator',
    'RMSCalculator',
    'MedianArtifactCalculator',
    'MetricsReport',
]
