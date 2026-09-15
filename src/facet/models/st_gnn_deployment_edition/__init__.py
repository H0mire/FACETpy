"""ST-GNN deployment edition — scored on the EEG it hands back.

See :mod:`facet.models.st_gnn_deployment_edition.training` for what changed against
``facet.models.st_gnn`` and why.
"""

from .training import CORE_OUTPUT, PACKING, build_dataset, build_loss, build_model

__all__ = ["CORE_OUTPUT", "PACKING", "build_dataset", "build_loss", "build_model"]
