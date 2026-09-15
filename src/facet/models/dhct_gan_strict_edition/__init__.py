"""DHCT-GAN, strict edition — full three-discriminator adversarial training.

See :mod:`facet.models.dhct_gan_strict_edition.training` and the folder README for
what "strict" means here and where the FACETpy context extension departs from the
paper on purpose.
"""

from .training import (  # noqa: F401
    DHCTGanStrictDiscriminator,
    DHCTGanStrictGenerator,
    DHCTGanStrictObjective,
    build_dataset,
    build_loss,
    build_model,
    build_wrapper,
)

__all__ = [
    "DHCTGanStrictGenerator",
    "DHCTGanStrictDiscriminator",
    "DHCTGanStrictObjective",
    "build_model",
    "build_loss",
    "build_dataset",
    "build_wrapper",
]
