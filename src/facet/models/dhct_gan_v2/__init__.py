"""DHCT-GAN v2: multi-epoch context variant of the dual-branch hybrid CNN-Transformer GAN."""

from .processor import DHCTGanV2Adapter, DHCTGanV2Correction

__all__ = [
    "DHCTGanV2Adapter",
    "DHCTGanV2Correction",
]
