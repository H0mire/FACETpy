"""Paper-accurate Conv-TasNet single-channel gradient-artifact source separator.

A more faithful re-implementation of Luo & Mesgarani (2019), arXiv:1809.07454v3,
than ``facet.models.conv_tasnet`` -- linear encoder default, explicit skip width
(Sc), and an optional source-additivity penalty -- while staying compatible with
the FACETpy training and inference contracts and CPU-cheap for EEG-fMRI.
"""

from .processor import ConvTasNetPaperAccurateAdapter, ConvTasNetPaperAccurateCorrection

__all__ = [
    "ConvTasNetPaperAccurateAdapter",
    "ConvTasNetPaperAccurateCorrection",
]
