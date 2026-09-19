# DPAE — experimental paper_accurate variant

Dual-pathway convolutional autoencoder. This experimental variant adds a compression-and-expansion fusion module with a residual connection. It changes path strides and normalization, and predicts clean EEG by default.

## Origin and role

Origin: EEG denoising.

Sources: [Xiong2024](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#xiong2024).

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/dpae.html) explains the family, adaptations and limits.

This package is experimental. Its name does not establish paper fidelity or equivalence to a thesis result.

Factories: `facet.models.experimental.paper_accurate.dpae.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
