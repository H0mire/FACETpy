# DHCT-GAN

Dual-branch convolution and attention model. A shared encoder and two decoders estimate clean EEG and artifact from a single-channel epoch. The exported forward path returns the artifact branch.

## Origin and role

Origin: EEG artifact removal.

Sources: [Cai2025](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#cai2025).

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/dhct_gan.html) explains the family, adaptations and limits.

Factories: `facet.models.masterthesis.dhct_gan.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
