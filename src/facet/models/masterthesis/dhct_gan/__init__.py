"""DHCT-GAN: dual-branch hybrid CNN-Transformer generative adversarial denoiser."""

from facet.models.masterthesis.dhct_gan.processor import DHCTGanAdapter, DHCTGanCorrection

__all__ = [
    "DHCTGanAdapter",
    "DHCTGanCorrection",
]
