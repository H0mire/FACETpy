"""
Miscellaneous utilities for FACETpy.

This module contains utility functions and classes that don't fit
into other categories, including synthetic data generation.
"""

from .eeg_generator import (
    ArtifactParams,
    ChannelSchema,
    EEGGenerator,
    NoiseParams,
    OscillationParams,
    SpikeParams,
    generate_oscillation,
    generate_pink_noise,
    generate_synthetic_eeg,
)

__all__ = [
    "EEGGenerator",
    "ChannelSchema",
    "OscillationParams",
    "NoiseParams",
    "ArtifactParams",
    "SpikeParams",
    "generate_synthetic_eeg",
    "generate_pink_noise",
    "generate_oscillation",
]
