"""Paper-accurate Dual-Pathway Autoencoder (DPAE) edition.

A more faithful re-implementation of Xiong, Ma & Li (2023), "A general
dual-pathway network for EEG denoising" (Front. Neurosci. 17:1258024), with a
symmetric fusion module, a residual skip wrapping the fusion module, a clean-EEG
reconstruction target, paper-style k3/k5 stride1/4 pathways, asymmetric
0.45/0.75 shrinkage ratios and per-segment std/max-abs normalisation. See the
README and ``documentation/paper_accuracy_review.md`` for the discrepancy table.
"""

from .processor import (
    DPAEPaperAccurateAdapter,
    DPAEPaperAccurateCorrection,
)

__all__ = [
    "DPAEPaperAccurateAdapter",
    "DPAEPaperAccurateCorrection",
]
