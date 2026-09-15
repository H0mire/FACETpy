"""ViT-Spectrogram deployment edition — scored on the EEG it hands back.

See :mod:`facet.models.vit_spectrogram_deployment_edition.training` for what changed against
``facet.models.vit_spectrogram`` and why.
"""

from .training import CORE_OUTPUT, PACKING, build_dataset, build_loss, build_model

__all__ = ["CORE_OUTPUT", "PACKING", "build_dataset", "build_loss", "build_model"]
