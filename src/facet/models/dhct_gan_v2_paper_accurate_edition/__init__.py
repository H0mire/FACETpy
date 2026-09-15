"""Paper-accurate edition of DHCT-GAN v2.

Makes the FACETpy DHCT-GAN v2 adaptation more faithful to Cai, Meng & Huang,
"DHCT-GAN: Improving EEG Signal Quality with a Dual-Branch Hybrid CNN-Transformer
Network", MDPI Sensors 25(1) 231 (2025): LSGAN adversarial/discriminator loss,
feature-matching loss, MSE reconstruction, three discriminators, two independent
tanh gating networks, and a paper-faithful Local-Global Transformer Block
(8-block local self-attention + per-attention feedforward + repeated LGTB).
"""

from .processor import DHCTGanV2PaperAccurateAdapter, DHCTGanV2PaperAccurateCorrection

__all__ = [
    "DHCTGanV2PaperAccurateAdapter",
    "DHCTGanV2PaperAccurateCorrection",
]
