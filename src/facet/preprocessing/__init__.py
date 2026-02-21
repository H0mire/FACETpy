"""
Preprocessing Module

This module contains processors for preprocessing EEG data.

Author: FACETpy Team
Date: 2025-01-12
"""

from .filtering import (
    HighPassFilter,
    LowPassFilter,
    BandPassFilter,
    NotchFilter,
    Filter
)
from .resampling import UpSample, DownSample, Resample
from .trigger_detection import (
    TriggerDetector,
    QRSTriggerDetector,
    MissingTriggerDetector
)
from .alignment import TriggerAligner, SliceAligner, SubsampleAligner
from .acquisition import CutAcquisitionWindow, PasteAcquisitionWindow
from .transforms import Crop, RawTransform

__all__ = [
    # Filtering
    'HighPassFilter',
    'LowPassFilter',
    'BandPassFilter',
    'NotchFilter',
    'Filter',

    # Resampling
    'UpSample',
    'DownSample',
    'Resample',

    # Trigger Detection
    'TriggerDetector',
    'QRSTriggerDetector',
    'MissingTriggerDetector',

    # Alignment
    'TriggerAligner',
    'SliceAligner',
    'SubsampleAligner',

    # Acquisition window
    'CutAcquisitionWindow',
    'PasteAcquisitionWindow',

    # Transforms
    'Crop',
    'RawTransform',
]
