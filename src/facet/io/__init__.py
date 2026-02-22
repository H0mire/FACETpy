"""
I/O Module

This module contains processors for loading and exporting EEG data.

Author: FACETpy Team
Date: 2025-01-12
"""

from .exporters import BIDSExporter, EDFExporter
from .loaders import SUPPORTED_EXTENSIONS, BIDSLoader, Loader

__all__ = [
    # Loaders
    "Loader",
    "BIDSLoader",
    "SUPPORTED_EXTENSIONS",
    # Exporters
    "EDFExporter",
    "BIDSExporter",
]
