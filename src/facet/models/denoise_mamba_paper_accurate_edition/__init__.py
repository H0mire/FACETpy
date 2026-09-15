"""Paper-accurate DenoiseMamba (U-Net ConvSSD + Mamba-2 SSD) edition.

A from-scratch rebuild of the ``denoise_mamba`` model to follow the source paper
(Chen et al., "DenoiseMamba", IEEE JBHI, vol. 29 no. 9, 2025, pp. 6551-6562)
more faithfully: a U-shaped encoder/decoder, a real channel-split ConvSSD block
with a dual SSD branch + learnable r1/r2 fusion, a Mamba-2 (SSD) layer, and a
clean-signal target with MSE loss. See ``README.md`` and
``documentation/paper_accuracy_review.md``.
"""

from .processor import (
    DenoiseMambaPaperAccurateCorrection,
    PaperAccurateDenoiseMambaAdapter,
)

__all__ = [
    "PaperAccurateDenoiseMambaAdapter",
    "DenoiseMambaPaperAccurateCorrection",
]
