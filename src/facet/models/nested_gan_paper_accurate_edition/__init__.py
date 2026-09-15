"""Paper-accurate Nested-GAN edition.

A more Restormer-faithful re-implementation of the inner spectral generator
branch: a genuine hierarchical encoder-decoder (pixel-unshuffle/shuffle, per-
level channels/heads/depth, skip-concat + 1x1 reduce, refinement stage, global
residual) with GDFN expansion gamma=2.66 and an optional bias-free LayerNorm.
The generator-only + multi-resolution-STFT-loss recipe and the outer time-domain
refiner are preserved; the GAN/nesting structure could not be verified against
the paywalled primary paper. See README.md and documentation/.
"""

from .processor import NestedGANPaperAccurateAdapter, NestedGANPaperAccurateCorrection

__all__ = [
    "NestedGANPaperAccurateAdapter",
    "NestedGANPaperAccurateCorrection",
]
