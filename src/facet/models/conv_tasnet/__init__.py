"""Conv-TasNet single-channel gradient-artifact source separator."""

from .processor import ConvTasNetAdapter, ConvTasNetCorrection

__all__ = [
    "ConvTasNetAdapter",
    "ConvTasNetCorrection",
]
