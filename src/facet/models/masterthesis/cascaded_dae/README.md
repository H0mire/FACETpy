# Cascaded DAE

Fully connected denoising autoencoder. Two fully connected stages predict an artifact from one channel and one epoch. Stage two receives the signal after the first estimate is subtracted.

## Origin and role

Origin: Duffy et al. (2020), gradient artifact correction in simultaneous EEG-fMRI; adapted in FACETpy.

Source: [Duffy et al. (2020)](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#duffy2020). The family reference explains the FACETpy adaptations.

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/cascaded_dae.html) explains the family, adaptations and limits.

Factories: `facet.models.masterthesis.cascaded_dae.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
