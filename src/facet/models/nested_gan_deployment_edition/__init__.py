"""Nested-GAN deployment edition — scored on the EEG it hands back.

See :mod:`facet.models.nested_gan_deployment_edition.training` for what changed against
``facet.models.nested_gan`` and why.
"""

from .training import CORE_OUTPUT, PACKING, build_dataset, build_loss, build_model

__all__ = ["CORE_OUTPUT", "PACKING", "build_dataset", "build_loss", "build_model"]
