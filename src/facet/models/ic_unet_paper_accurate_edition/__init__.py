"""Paper-accurate IC-U-Net edition.

A more faithful re-implementation of IC-U-Net (Chuang et al. 2022) for FACETpy:
a sensor-level 1-D U-Net with ReLU CBR blocks, a transposed-convolution decoder,
a normalised equal-weight four-term ensemble loss with a z-scored 1-50 Hz PSD
frequency term, per-time-series z-score normalisation, and a clean-reconstruction
target by default. See ``README.md`` and ``documentation/paper_accuracy_review.md``.
"""

from .processor import IcUNetPaperAccurateAdapter, IcUNetPaperAccurateCorrection

__all__ = [
    "IcUNetPaperAccurateAdapter",
    "IcUNetPaperAccurateCorrection",
]
