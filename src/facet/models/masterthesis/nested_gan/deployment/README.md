# Nested-GAN — deployment

Spectral generator followed by temporal refinement. A spectral generator and temporal residual refiner predict the centre artifact from neighbouring epochs. This variant uses a generator-only recipe, without the paper's full nested adversarial training. The deployment wrapper normalizes input, restores output units and can remove the predicted epoch mean. Its objective scores recovered clean EEG.

## Origin and role

Origin: EEG artifact removal; image-restoration building blocks.

Sources: [Yang2025](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#yang2025) · [Zamir2022](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#zamir2022).

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/nested_gan.html) explains the family, adaptations and limits.

Factories: `facet.models.masterthesis.nested_gan.deployment.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
