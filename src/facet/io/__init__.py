"""
I/O Module

This module contains processors for loading and exporting EEG data.

Author: FACETpy Team
Date: 2025-01-12
"""

from .loaders import EDFLoader, BIDSLoader, GDFLoader
from .exporters import EDFExporter, BIDSExporter

__all__ = [
    # Loaders
    'EDFLoader',
    'BIDSLoader',
    'GDFLoader',

    # Exporters
    'EDFExporter',
    'BIDSExporter',
]
