"""Nested-GAN: inner spectrogram Restormer cascaded into an outer time-domain refiner."""

from facet.models.masterthesis.nested_gan.processor import NestedGANAdapter, NestedGANCorrection

__all__ = [
    "NestedGANAdapter",
    "NestedGANCorrection",
]
