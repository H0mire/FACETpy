"""Paper-accurate edition of DHCT-GAN.

A strictly more faithful re-implementation of DHCT-GAN (Cai, Meng & Huang, MDPI
*Sensors* 2025, 25(1):231) than ``facet.models.dhct_gan``: MSE reconstruction
(Eq. 10), LSGAN adversarial + discriminator objective (Eqs. 12-13), discriminator
feature-matching loss (Eq. 11), three discriminators (D1 clean / D2 artifact /
D3 fused, Eqs. 6-9), two independent gating heads (Eqs. 4-5), fixed-block local
attention, a configurable LGTB inner stack and a parallel CNN path. CPU-cheap and
fully compatible with the facet-train factory + TorchScript inference contracts.
"""

from .processor import DHCTGanPaperAccurateAdapter, DHCTGanPaperAccurateCorrection

__all__ = [
    "DHCTGanPaperAccurateAdapter",
    "DHCTGanPaperAccurateCorrection",
]
