"""
I/O Module

This module contains processors for loading and exporting EEG data.

Author: FACETpy Team
Date: 2025-01-12
"""

from .loaders import Loader, BIDSLoader, SUPPORTED_EXTENSIONS
from .exporters import EDFExporter, BIDSExporter

__all__ = [
    # Loaders
    'Loader',
    'BIDSLoader',
    'SUPPORTED_EXTENSIONS',

    # Exporters
    'EDFExporter',
    'BIDSExporter',
]
