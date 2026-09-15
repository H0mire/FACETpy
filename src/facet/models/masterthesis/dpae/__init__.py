"""Dual-Pathway Autoencoder (DPAE) for gradient artifact removal."""

from facet.models.masterthesis.dpae.processor import (
    DualPathwayAutoencoderAdapter,
    DualPathwayAutoencoderCorrection,
)

__all__ = [
    "DualPathwayAutoencoderAdapter",
    "DualPathwayAutoencoderCorrection",
]
