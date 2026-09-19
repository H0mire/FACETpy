# Conv-TasNet

Convolutional source separator. A learned waveform encoder, temporal convolution network and decoder separate ordered clean-EEG and artifact targets.

## Origin and role

Origin: Speech separation.

Sources: [Luo2019](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#luo2019).

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/conv_tasnet.html) explains the family, adaptations and limits.

Factories: `facet.models.masterthesis.conv_tasnet.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
