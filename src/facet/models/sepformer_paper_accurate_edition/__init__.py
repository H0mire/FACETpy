"""Paper-accurate SepFormer dual-path Transformer artifact correction model.

A more paper-faithful edition of ``facet.models.sepformer`` (Subakan et al.,
ICASSP 2021, arXiv:2010.13154). See ``README.md`` and
``documentation/paper_accuracy_review.md`` for the full list of changes and
the documented deviations.
"""

from .processor import SepFormerPaperAccurateAdapter, SepFormerPaperAccurateCorrection

__all__ = [
    "SepFormerPaperAccurateAdapter",
    "SepFormerPaperAccurateCorrection",
]
