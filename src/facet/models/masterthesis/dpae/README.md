# DPAE

Dual-pathway convolutional autoencoder. Two convolutional pathways combine fine and coarse temporal features to predict a single-channel artifact.

## Origin and role

Origin: EEG denoising.

Sources: [Xiong2024](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#xiong2024).

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/dpae.html) explains the family, adaptations and limits.

Factories: `facet.models.masterthesis.dpae.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
