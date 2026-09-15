"""IC-U-Net deployment edition — trained against the recovered clean signal.

See :mod:`facet.models.ic_unet_deployment_edition.training` for why this edition
exists and what the three changes against ``ic_unet`` are.
"""

from .training import (
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
