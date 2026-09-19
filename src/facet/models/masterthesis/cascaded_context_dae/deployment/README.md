# Cascaded Context DAE — deployment

Fully connected context autoencoder. Two fully connected stages read seven epochs from one channel and predict the centre artifact. The second stage reads context with the first centre estimate removed. The deployment wrapper normalizes input, restores output units and can remove the predicted epoch mean. Its objective scores recovered clean EEG.

## Origin and role

Origin: FACETpy context extension of the Cascaded DAE family from Duffy et al. (2020).

Source: [Duffy et al. (2020)](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#duffy2020). The family reference explains the FACETpy adaptations.

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/context_dae.html) explains the family, adaptations and limits.

Factories: `facet.models.masterthesis.cascaded_context_dae.deployment.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
