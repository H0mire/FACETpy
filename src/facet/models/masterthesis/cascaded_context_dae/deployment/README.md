# Cascaded Context DAE — deployment

Fully connected context autoencoder. Two fully connected stages read seven epochs from one channel and predict the centre artifact. The second stage reads context with the first centre estimate removed. The deployment wrapper normalizes input, restores output units and can remove the predicted epoch mean. Its objective scores recovered clean EEG.

## Origin and role

Origin: FACETpy extension of Cascaded DAE.

Developed within FACETpy; no dedicated source paper is assigned to this implementation.

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/context_dae.html) explains the family, adaptations and limits.

Factories: `facet.models.masterthesis.cascaded_context_dae.deployment.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
