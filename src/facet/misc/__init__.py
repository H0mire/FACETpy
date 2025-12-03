"""
Miscellaneous utilities for FACETpy.

This module contains utility functions and classes that don't fit
into other categories, including synthetic data generation.
"""

from .eeg_generator import (
    EEGGenerator,
    ChannelSchema,
    OscillationParams,
    NoiseParams,
    ArtifactParams,
    generate_synthetic_eeg,
    generate_pink_noise,
    generate_oscillation,
)

__all__ = [
    "EEGGenerator",
    "ChannelSchema",
    "OscillationParams",
    "NoiseParams",
    "ArtifactParams",
    "generate_synthetic_eeg",
    "generate_pink_noise",
    "generate_oscillation",
]
