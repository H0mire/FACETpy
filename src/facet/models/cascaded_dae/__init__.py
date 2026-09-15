"""Cascaded DAE: channel-wise cascaded denoising autoencoder.

Import training factories from ``facet.models.cascaded_dae.training`` directly. The
package-level API intentionally exposes only inference integration classes so
pipeline imports do not pull in training dependencies.
"""

from .processor import CascadedDenoisingAutoencoderAdapter, CascadedDenoisingAutoencoderCorrection

__all__ = [
    "CascadedDenoisingAutoencoderAdapter",
    "CascadedDenoisingAutoencoderCorrection",
]
