"""Paper-accurate D4PM dual-branch diffusion gradient-artifact predictor.

A more faithful re-implementation of D4PM (arXiv:2509.14302) than the original
``facet.models.d4pm``: continuous noise-level conditioning, three Transformer
blocks per path, Dual-FiLM with class conditioning, an optional second
(clean/EEG) branch, joint posterior sampling, and the true DDPM ancestral
sampler. See README.md and documentation/paper_accuracy_review.md.
"""

from .processor import D4PMPaperAccurateAdapter, D4PMPaperAccurateCorrection

__all__ = [
    "D4PMPaperAccurateAdapter",
    "D4PMPaperAccurateCorrection",
]
