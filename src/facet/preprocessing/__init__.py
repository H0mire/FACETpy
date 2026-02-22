"""
Preprocessing Module

This module contains processors for preprocessing EEG data.

Author: FACETpy Team
Date: 2025-01-12
"""

from .acquisition import CutAcquisitionWindow, PasteAcquisitionWindow
from .alignment import SliceAligner, SubsampleAligner, TriggerAligner
from .filtering import BandPassFilter, Filter, HighPassFilter, LowPassFilter, NotchFilter
from .resampling import DownSample, Resample, UpSample
from .transforms import Crop, DropChannels, PickChannels, PrintMetric, RawTransform
from .trigger_detection import MissingTriggerDetector, QRSTriggerDetector, TriggerDetector
from .trigger_explorer import InteractiveTriggerExplorer, TriggerExplorer

__all__ = [
    # Filtering
    "HighPassFilter",
    "LowPassFilter",
    "BandPassFilter",
    "NotchFilter",
    "Filter",
    # Resampling
    "UpSample",
    "DownSample",
    "Resample",
    # Trigger Detection
    "TriggerDetector",
    "QRSTriggerDetector",
    "MissingTriggerDetector",
    "TriggerExplorer",
    "InteractiveTriggerExplorer",
    # Alignment
    "TriggerAligner",
    "SliceAligner",
    "SubsampleAligner",
    # Acquisition window
    "CutAcquisitionWindow",
    "PasteAcquisitionWindow",
    # Transforms
    "Crop",
    "RawTransform",
    "PickChannels",
    "DropChannels",
    "PrintMetric",
]
