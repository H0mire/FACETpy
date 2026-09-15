"""ViT-Spectrogram deployment edition — scored on the EEG it hands back.

See :mod:`facet.models.masterthesis.vit_spectrogram.deployment.training` for what changed against
``facet.models.masterthesis.vit_spectrogram`` and why.
"""

from facet.models.masterthesis.vit_spectrogram.deployment.training import CORE_OUTPUT, PACKING, build_dataset, build_loss, build_model

__all__ = ["CORE_OUTPUT", "PACKING", "build_dataset", "build_loss", "build_model"]
