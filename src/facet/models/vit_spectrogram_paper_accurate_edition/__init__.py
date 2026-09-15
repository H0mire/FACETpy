"""Paper-accurate MAE-ViT spectrogram inpainter for fMRI gradient artifact removal.

A more faithful re-implementation of the ViT (Dosovitskiy et al. 2021,
arXiv:2010.11929) encoder and the MAE (He et al. 2022, arXiv:2111.06377)
asymmetric encoder-decoder, masked-patch reconstruction loss, and fixed 2D
sin-cos position embeddings, adapted for single-/few-channel EEG-fMRI gradient
artifact inpainting. See ``README.md`` and
``documentation/paper_accuracy_review.md`` for the discrepancy table and the
documented EEG-fMRI deviations.
"""

from .processor import (
    ViTSpectrogramMAEAdapter,
    ViTSpectrogramMAECorrection,
)

__all__ = [
    "ViTSpectrogramMAEAdapter",
    "ViTSpectrogramMAECorrection",
]
