"""Time-domain Demucs adapted for fMRI gradient-artifact removal."""

from .processor import DemucsAdapter, DemucsCorrection

__all__ = [
    "DemucsAdapter",
    "DemucsCorrection",
]
