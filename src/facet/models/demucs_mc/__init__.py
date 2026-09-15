"""Multichannel Demucs with a cross-unit attention bridge (run_6 Phase B).

See :mod:`facet.models.demucs_mc.training` for the model and the
``facet-train`` factories.
"""

from .training import (  # noqa: F401
    CrossUnitAttention,
    MultichannelDemucs,
    build_dataset,
    build_loss,
    build_model,
)

__all__ = [
    "CrossUnitAttention",
    "MultichannelDemucs",
    "build_dataset",
    "build_loss",
    "build_model",
]
