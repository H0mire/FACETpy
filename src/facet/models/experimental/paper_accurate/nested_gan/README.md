# Nested-GAN — experimental paper_accurate variant

Spectral generator followed by temporal refinement. This experimental variant adds a hierarchical Restormer-style spectral generator. It keeps the temporal refiner and generator-only recipe; it is not a reproduction of the full nested GAN training method.

## Origin and role

Origin: EEG artifact removal; image-restoration building blocks.

Sources: [Yang2025](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#yang2025) · [Zamir2022](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#zamir2022).

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/nested_gan.html) explains the family, adaptations and limits.

This package is experimental. Its name does not establish paper fidelity or equivalence to a thesis result.

Factories: `facet.models.experimental.paper_accurate.nested_gan.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Results

[Recorded results](results/index.rst) lists this variant's experiments, hyperparameters, training curves and evaluations. [Manifest](results/manifest.json) records source and checkpoint associations. Unassigned results are marked explicitly.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
