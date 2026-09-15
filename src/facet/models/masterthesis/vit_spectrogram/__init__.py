"""Vision-Transformer spectrogram inpainter for fMRI gradient artifact removal."""

from facet.models.masterthesis.vit_spectrogram.processor import (
    ViTSpectrogramInpainterAdapter,
    ViTSpectrogramInpainterCorrection,
)

__all__ = [
    "ViTSpectrogramInpainterAdapter",
    "ViTSpectrogramInpainterCorrection",
]
