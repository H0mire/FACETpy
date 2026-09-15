"""Compatibility factories for the Demo 01 context artifact model.

The implementation lives in :mod:`facet.models.experimental.examples.demo01.training`. This module is
kept so existing ``facet-train`` configs using
``facet.models.experimental.examples.demo01.training:build_model`` continue to work.
"""

from facet.models.experimental.examples.demo01.training import SevenEpochContextArtifactNet, build_dataset, build_loss, build_model

__all__ = [
    "SevenEpochContextArtifactNet",
    "build_dataset",
    "build_loss",
    "build_model",
]
