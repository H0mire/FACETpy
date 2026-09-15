"""Nested-GAN: inner spectrogram Restormer cascaded into an outer time-domain refiner."""

from .processor import NestedGANAdapter, NestedGANCorrection

__all__ = [
    "NestedGANAdapter",
    "NestedGANCorrection",
]
