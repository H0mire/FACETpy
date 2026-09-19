# Conv-TasNet — experimental paper_accurate variant

Convolutional source separator. This experimental variant uses a linear encoder with sigmoid masks and a separate skip-channel width. Its default loss scores ordered EEG and artifact targets; it does not reproduce the speech training protocol.

## Origin and role

Origin: Speech separation.

Sources: [Luo2019](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#luo2019).

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/conv_tasnet.html) explains the family, adaptations and limits.

This package is experimental. Its name does not establish paper fidelity or equivalence to a thesis result.

Factories: `facet.models.experimental.paper_accurate.conv_tasnet.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
