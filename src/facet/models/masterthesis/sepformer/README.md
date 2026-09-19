# SepFormer

Dual-path attention source separator. Dual-path attention connects samples within and between learned feature chunks. A mask and decoder recover the centre-epoch artifact from one channel's context.

## Origin and role

Origin: Speech separation.

Sources: [Subakan2021](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#subakan2021).

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/sepformer.html) explains the family, adaptations and limits.

Factories: `facet.models.masterthesis.sepformer.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
