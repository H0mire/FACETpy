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
    SpectralCoherenceCalculator,
    SpikeDetectionRateCalculator,
)
from .model_evaluation import (
    EVALUATION_SCHEMA_VERSION,
    ModelEvaluationRun,
    ModelEvaluationWriter,
)
from .visualization import RawPlotter

__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "ReferenceIntervalSelector",
    "SignalIntervalSelector",
    "SNRCalculator",
    "LegacySNRCalculator",
    "RMSCalculator",
    "RMSResidualCalculator",
    "MedianArtifactCalculator",
    "FFTAllenCalculator",
    "FFTNiazyCalculator",
    "SpectralCoherenceCalculator",
    "SpikeDetectionRateCalculator",
    "MetricsReport",
    "ModelEvaluationRun",
    "ModelEvaluationWriter",
    "RawPlotter",
]
