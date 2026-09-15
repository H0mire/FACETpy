"""DenoiseMamba deployment edition — scored on the EEG it hands back.

See :mod:`facet.models.masterthesis.denoise_mamba.deployment.training` for what changed against
``facet.models.masterthesis.denoise_mamba`` and why.
"""

from facet.models.masterthesis.denoise_mamba.deployment.training import CORE_OUTPUT, PACKING, build_dataset, build_loss, build_model

__all__ = ["CORE_OUTPUT", "PACKING", "build_dataset", "build_loss", "build_model"]
