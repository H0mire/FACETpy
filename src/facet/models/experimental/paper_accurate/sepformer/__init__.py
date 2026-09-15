"""Paper-accurate SepFormer dual-path Transformer artifact correction model.

A more paper-faithful edition of ``facet.models.masterthesis.sepformer`` (Subakan et al.,
ICASSP 2021, arXiv:2010.13154). See ``README.md`` and
``documentation/paper_accuracy_review.md`` for the full list of changes and
the documented deviations.
"""

from facet.models.experimental.paper_accurate.sepformer.processor import (
    SepFormerPaperAccurateAdapter,
    SepFormerPaperAccurateCorrection,
)

__all__ = [
    "SepFormerPaperAccurateAdapter",
    "SepFormerPaperAccurateCorrection",
]
