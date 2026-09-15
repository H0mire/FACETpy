"""Demucs deployment edition — scored on the EEG it hands back.

See :mod:`facet.models.masterthesis.demucs.deployment.training` for what changed against
``facet.models.masterthesis.demucs`` and why.
"""

from facet.models.masterthesis.demucs.deployment.training import (
    CORE_OUTPUT,
    PACKING,
    build_dataset,
    build_loss,
    build_model,
)

__all__ = ["CORE_OUTPUT", "PACKING", "build_dataset", "build_loss", "build_model"]
