"""DHCT-GAN: dual-branch hybrid CNN-Transformer generative adversarial denoiser."""

from .processor import DHCTGanAdapter, DHCTGanCorrection

__all__ = [
    "DHCTGanAdapter",
    "DHCTGanCorrection",
]
