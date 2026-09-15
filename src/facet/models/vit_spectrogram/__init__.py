"""Vision-Transformer spectrogram inpainter for fMRI gradient artifact removal."""

from .processor import (
    ViTSpectrogramInpainterAdapter,
    ViTSpectrogramInpainterCorrection,
)

__all__ = [
    "ViTSpectrogramInpainterAdapter",
    "ViTSpectrogramInpainterCorrection",
]
