"""D4PM deployment edition — ε-prediction kept, waveform scored alongside.

See :mod:`facet.models.masterthesis.d4pm.deployment.training` for why this edition
could not simply swap its loss the way the other twelve did.
"""

from facet.models.masterthesis.d4pm.deployment.training import (
    CORE_OUTPUT,
    PACKING,
    D4PMDeploymentDataset,
    D4PMDeploymentLoss,
    D4PMWaveformModule,
    build_dataset,
    build_loss,
    build_model,
)

__all__ = [
    "CORE_OUTPUT",
    "PACKING",
    "D4PMDeploymentDataset",
    "D4PMDeploymentLoss",
    "D4PMWaveformModule",
    "build_dataset",
    "build_loss",
    "build_model",
]
