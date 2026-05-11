"""Dual-Pathway Autoencoder (DPAE) for gradient artifact removal."""

from .processor import (
    DualPathwayAutoencoderAdapter,
    DualPathwayAutoencoderCorrection,
)

__all__ = [
    "DualPathwayAutoencoderAdapter",
    "DualPathwayAutoencoderCorrection",
]
