# DenoiseMamba — experimental paper_accurate variant

Convolutional and state-space model. This experimental variant uses a U-shaped ConvSSD encoder-decoder and Mamba-2-style state-space processing. It predicts clean EEG by default and can be configured to predict artifacts.

## Origin and role

Origin: EEG artifact removal.

Sources: [Chen2025](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#chen2025).

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/denoise_mamba.html) explains the family, adaptations and limits.

This package is experimental. Its name does not establish paper fidelity or equivalence to a thesis result.

Factories: `facet.models.experimental.paper_accurate.denoise_mamba.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
