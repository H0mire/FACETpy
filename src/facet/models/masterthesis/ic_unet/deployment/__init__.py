"""IC-U-Net deployment edition — trained against the recovered clean signal.

See :mod:`facet.models.masterthesis.ic_unet.deployment.training` for why this edition
exists and what the three changes against ``ic_unet`` are.
"""

from facet.models.masterthesis.ic_unet.deployment.training import (
    CORE_OUTPUT,
    PACKING,
    ContextIcUnetDeploymentDataset,
    SelfNormalisingIcUnet,
    build_dataset,
    build_loss,
    build_model,
)

__all__ = [
    "CORE_OUTPUT",
    "PACKING",
    "ContextIcUnetDeploymentDataset",
    "SelfNormalisingIcUnet",
    "build_dataset",
    "build_loss",
    "build_model",
]
