# DHCT-GAN v2 — experimental paper_accurate variant

Dual-branch convolution and attention model. This experimental context variant uses separate fusion gates and adversarial feature matching. It retains a shared encoder and returns artifact as noisy centre EEG minus the fused clean estimate.

## Origin and role

Origin: EEG artifact removal.

Sources: [Cai2025](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#cai2025).

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/dhct_gan.html) explains the family, adaptations and limits.

This package is experimental. Its name does not establish paper fidelity or equivalence to a thesis result.

Factories: `facet.models.experimental.paper_accurate.dhct_gan_v2.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
