"""
I/O Module

This module contains processors for loading and exporting EEG data.

Author: FACETpy Team
Date: 2025-01-12
"""

from .exporters import (
    SUPPORTED_EXPORT_EXTENSIONS,
    BDFExporter,
    BIDSExporter,
    BrainVisionExporter,
    EDFExporter,
    EEGLABExporter,
    Exporter,
    FIFExporter,
    GDFExporter,
    MFFExporter,
)
from .loaders import SUPPORTED_EXTENSIONS, BIDSLoader, Loader

__all__ = [
    # Loaders
    "Loader",
    "BIDSLoader",
    "SUPPORTED_EXTENSIONS",
    # Exporters
    "Exporter",
    "EDFExporter",
    "BDFExporter",
    "BrainVisionExporter",
    "EEGLABExporter",
    "FIFExporter",
    "GDFExporter",
    "MFFExporter",
    "BIDSExporter",
    "SUPPORTED_EXPORT_EXTENSIONS",
]
