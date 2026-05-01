"""Compatibility import for the Demo 01 epoch-context DL processor.

The implementation lives in :mod:`facet.models.demo01.processor` because it is
model-specific. Importing it from ``facet.correction`` is kept for existing
closed-beta examples and tests.
"""

from ..models.demo01.processor import EpochContextDeepLearningCorrection

__all__ = ["EpochContextDeepLearningCorrection"]
