"""Paper-accurate time-domain Demucs adapted for fMRI gradient-artifact removal.

A more faithful edition of ``facet.models.demucs`` (Defossez et al. 2019,
arXiv:1911.13254): length-agnostic ``valid_length`` padding/center-trim, the
optional 2x resampling trick (Sec 4.1), the test-time shift trick (Sec 4.4),
paper-exact init weight rescaling (Sec 4.3), and principled auto-depth.
"""

from .processor import DemucsPaperAccurateAdapter, DemucsPaperAccurateCorrection

__all__ = [
    "DemucsPaperAccurateAdapter",
    "DemucsPaperAccurateCorrection",
]
