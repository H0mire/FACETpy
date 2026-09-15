"""IC-U-Net: multichannel 1-D U-Net with ICA preprocessing for fMRI gradient artifact removal."""

from .processor import IcUnetAdapter, IcUnetCorrection

__all__ = [
    "IcUnetAdapter",
    "IcUnetCorrection",
]
